"""
Inference script for FractalGen MIDI model (Temporal Fractal Network).
Supports unconditional generation, conditional generation, and inpainting.

Usage:
    # Using config file
    python inference.py --config config/inference_default.yaml --checkpoint path/to/model.ckpt
    
    # Unconditional generation
    python inference.py --checkpoint path/to/model.ckpt --mode unconditional --output_dir outputs/samples
"""

import argparse
from pathlib import Path
import torch
import symusic
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import yaml

from trainer import FractalMIDILightningModule
from visualizer import piano_roll_to_image


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """Merge YAML config with command line arguments."""
    if config is None:
        return args
    
    merged = argparse.Namespace()
    
    # Checkpoint
    merged.checkpoint = args.checkpoint if args.checkpoint else config.get('checkpoint')
    
    # Mode
    merged.mode = args.mode if args.mode != 'unconditional' else config.get('mode', 'unconditional')
    merged.save_gif = args.save_gif if args.save_gif else config.get('output', {}).get('save_gif', False)
    
    # Unconditional settings
    unconditional_cfg = config.get('unconditional', {})
    merged.num_samples = args.num_samples if args.num_samples != 4 else unconditional_cfg.get('num_samples', 4)
    merged.generation_length = args.generation_length if args.generation_length != 256 else unconditional_cfg.get('length', 256)
    merged.temperature = args.temperature if args.temperature != 1.0 else unconditional_cfg.get('temperature', 1.0)
    
    # Output
    output_cfg = config.get('output', {})
    merged.output_dir = args.output_dir if args.output_dir != 'outputs/inference' else output_cfg.get('output_dir', 'outputs/inference')
    merged.save_images = args.save_images if args.save_images else output_cfg.get('save_png', True)
    
    # Hardware
    merged.device = args.device if args.device != 'cuda:0' else config.get('device', 'cuda:0')
    
    return merged


def create_generation_gif(intermediates, output_path, fps=15):
    """
    Create a GIF from intermediate generation steps.
    """
    try:
        from visualizer import create_growth_animation
        
        sample_frames = []
        for item in intermediates:
            frame = item['output'] # (B, 3, T, 128) or (3, T, 128)
            if frame.ndim == 4: frame = frame[0]
            sample_frames.append(frame)
            
        create_growth_animation(
            sample_frames,
            save_path=output_path,
            fps=fps,
            transition_duration=0.1,
            pop_effect=True
        )
    except ImportError:
        print("Could not import create_growth_animation from visualizer")
    except Exception as e:
        print(f"Error creating GIF: {e}")


def piano_roll_to_midi(piano_roll, tempo_curve=None, ticks_per_16th=120, velocity_threshold=0.10):
    """
    Convert piano roll to MIDI file.
    
    Args:
        piano_roll: (C, T, 128) numpy array. C=2 or C=3.
        tempo_curve: Optional (T,) array of normalized tempo if C=2.
        ticks_per_16th: Ticks per 16th note
        velocity_threshold: Minimum note activation to create a note
    
    Returns:
        symusic.Score object
    """
    score = symusic.Score()
    score.ticks_per_quarter = ticks_per_16th * 4
    
    # Create a piano track
    track = symusic.Track()
    track.name = "FractalPiano"
    track.program = 0 # Acoustic Grand Piano
    track.is_drum = False
    
    # Unpack channels
    C, T, H = piano_roll.shape
    
    note_layer = piano_roll[0] # (T, 128)
    vel_layer = piano_roll[1]  # (T, 128)
    
    if C >= 3:
        tempo_layer = piano_roll[2] # (T, 128)
        tempo_curve_arr = tempo_layer.mean(axis=1)
    elif tempo_curve is not None:
        tempo_curve_arr = tempo_curve
    else:
        tempo_curve_arr = np.ones(T) * 0.5 # Default 120 BPM (normalized)
    
    # 1. Extract Notes
    for pitch in range(H):
        is_note_on = False
        note_start = 0
        current_vel_sum = 0.0
        current_vel_count = 0
        
        for t in range(T):
            activation = note_layer[t, pitch]
            velocity = vel_layer[t, pitch]
            
            if activation > 0.5: # Binary threshold
                if not is_note_on:
                    # Note On
                    is_note_on = True
                    note_start = t
                    current_vel_sum = velocity
                    current_vel_count = 1
                else:
                    # Accumulate velocity for average
                    current_vel_sum += velocity
                    current_vel_count += 1
            else:
                if is_note_on:
                    # Note Off
                    is_note_on = False
                    duration = t - note_start
                    avg_vel = current_vel_sum / max(1, current_vel_count)
                    
                    if avg_vel > velocity_threshold:
                        vel_int = int(np.clip(avg_vel * 127, 1, 127))
                        note = symusic.Note(
                            time=note_start * ticks_per_16th,
                            duration=duration * ticks_per_16th,
                            pitch=pitch,
                            velocity=vel_int
                        )
                        track.notes.append(note)
        
        # End active note
        if is_note_on:
            duration = T - note_start
            avg_vel = current_vel_sum / max(1, current_vel_count)
            if avg_vel > velocity_threshold:
                vel_int = int(np.clip(avg_vel * 127, 1, 127))
                note = symusic.Note(
                    time=note_start * ticks_per_16th,
                    duration=duration * ticks_per_16th,
                    pitch=pitch,
                    velocity=vel_int
                )
                track.notes.append(note)
    
    score.tracks.append(track)
    
    # 2. Extract Tempo
    # Convert normalized tempo back to BPM
    # Norm = (BPM - 40) / 160
    # BPM = Norm * 160 + 40
    bpm_curve = tempo_curve_arr * 160.0 + 40.0
    
    # Filter tempo events (only write if significant change)
    last_bpm = -1.0
    for t in range(T):
        current_bpm = bpm_curve[t]
        # Only write if change > 2 BPM
        if abs(current_bpm - last_bpm) > 2.0:
            score.tempos.append(symusic.Tempo(
                time=t * ticks_per_16th,
                qpm=float(current_bpm)
            ))
            last_bpm = current_bpm
            
    if not score.tempos:
        score.tempos.append(symusic.Tempo(time=0, qpm=120.0))
        
    return score

def midi_to_tensor(midi_path, target_length=256):
    """Load MIDI and convert to tensor for inpainting."""
    try:
        score = symusic.Score(midi_path)
        score = score.resample(120, min_dur=1) # 120 ticks per 16th
        
        duration_steps = target_length
        note_layer = np.zeros((duration_steps, 128), dtype=np.float32)
        vel_layer = np.zeros((duration_steps, 128), dtype=np.float32)
        
        for track in score.tracks:
            for note in track.notes:
                pitch = note.pitch
                velocity = note.velocity / 127.0
                start_step = note.time // 120
                note_duration_steps = max(1, note.duration // 120)
                end_step = min(start_step + note_duration_steps, duration_steps)
                
                if 0 <= pitch < 128 and start_step < duration_steps:
                    note_layer[start_step:end_step, pitch] = 1.0
                    vel_layer[start_step:end_step, pitch] = np.maximum(vel_layer[start_step:end_step, pitch], velocity)
        
        notes = np.stack([note_layer, vel_layer], axis=0)
        return torch.from_numpy(notes) # (2, T, 128)
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FractalGen MIDI Inference')
    
    # Config file
    parser.add_argument('-c', '--config', type=str, default=None,
                       help='Path to YAML config file (e.g., config/inference_default.yaml)')
    
    # Model
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='unconditional',
                       choices=['unconditional'], # Only supporting unconditional for v2 first pass
                       help='Generation mode')
    parser.add_argument('--save_gif', action='store_true',
                       help='Save generation process as GIF for the first sample')
    
    # Unconditional generation
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to generate')
    parser.add_argument('--generation_length', type=int, default=256,
                       help='Length of generated sequence (in 16th notes)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    
    # Output
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/inference',
                       help='Output directory')
    parser.add_argument('--save_images', action='store_true',
                       help='Save piano roll images')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    cmd_args = parse_args()
    
    # Load config file if provided
    config = None
    if cmd_args.config:
        config_path = Path(cmd_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    
    # Merge config with command line arguments
    args = merge_config_with_args(config, cmd_args)
    
    # Validate checkpoint is provided
    if not args.checkpoint:
        raise ValueError("--checkpoint is required (either in config or as command line argument)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FractalGen MIDI Inference (Temporal Fractal Network)")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if Path(args.checkpoint).exists():
        # Load from checkpoint
        model = FractalMIDILightningModule.load_from_checkpoint(args.checkpoint)
        print(f"✓ Loaded from checkpoint: {args.checkpoint}")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params/1e6:.2f}M parameters")
    
    # Generate based on mode
    print(f"\nGenerating ({args.mode} mode)...")
    
    if args.mode == 'unconditional':
        # Unconditional generation
        for i in range(args.num_samples):
            print(f"\nGenerating sample {i+1}/{args.num_samples}...")
            
            return_intermediates = (i == 0 and args.save_gif)
            
            with torch.no_grad():
                result = model.model.sample(
                    batch_size=1,
                    length=args.generation_length,
                    global_cond=None,
                    cfg=1.0,
                    temperature=args.temperature,
                    num_iter_list=[8, 4, 2],
                    return_intermediates=return_intermediates
                )
            
            intermediates = None
            if isinstance(result, tuple):
                generated, intermediates = result
            else:
                generated = result
            
            piano_roll = generated[0].cpu().numpy()  # (2, T, 128)
            
            # Extract tempo from intermediates if available
            tempo_curve = None
            if intermediates:
                # Find last structure output
                for item in reversed(intermediates):
                    if item.get('is_structure', False):
                        struct = item['output'] # (B, 2, T_low)
                        tempo_low = struct[0, 1]
                        # Upsample
                        tempo_curve = np.interp(
                            np.linspace(0, len(tempo_low), args.generation_length),
                            np.arange(len(tempo_low)),
                            tempo_low.numpy()
                        )
                        break
            
            # Save MIDI
            midi_path = output_dir / f'unconditional_{i:03d}.mid'
            score = piano_roll_to_midi(piano_roll, tempo_curve=tempo_curve, velocity_threshold=0.1)
            score.dump_midi(str(midi_path))
            print(f"✓ Saved MIDI: {midi_path}")
            
            # Save image (visualizer expects 3 channels)
            if args.save_images:
                img_path = output_dir / f'unconditional_{i:03d}.png'
                if tempo_curve is None:
                    tempo_curve = np.ones(piano_roll.shape[1]) * 0.5
                
                # Reconstruct 3-channel roll for visualization
                tempo_ch = np.tile(tempo_curve[:, None], (1, 128)).T # (128, T) -> (T, 128)
                vis_roll = np.concatenate([piano_roll, tempo_ch[None, :, :]], axis=0) # (3, T, 128)
                
                img_pil = piano_roll_to_image(
                    torch.from_numpy(vis_roll),
                    apply_colormap=True,
                    return_pil=True,
                    composite_tempo=True
                )
                img_pil.save(str(img_path))
                print(f"✓ Saved image: {img_path}")
            
            if intermediates is not None and len(intermediates) > 0:
                # Filter frames for GIF (only content)
                # And add tempo channel
                content_intermediates = []
                for item in intermediates:
                    if not item.get('is_structure', False):
                        frame = item['output'][0] # (2, T, 128)
                        # Add tempo
                        T_frame = frame.shape[1]
                        tempo_frm = torch.ones(1, T_frame, 128) * 0.5
                        full_frame = torch.cat([frame, tempo_frm], dim=0)
                        content_intermediates.append({'output': full_frame.unsqueeze(0)})
                
                print(f"\nCreating generation process GIF...")
                gif_path = output_dir / f'unconditional_{i:03d}_process.gif'
                create_generation_gif(content_intermediates, str(gif_path), fps=15)
                print(f"✓ Saved GIF: {gif_path}")

    elif args.mode == 'inpainting':
        if not args.input_midi:
            raise ValueError("--input_midi is required for inpainting mode")
        
        print(f"Loading input MIDI: {args.input_midi}")
        initial_content = midi_to_tensor(args.input_midi, target_length=args.generation_length)
        if initial_content is None:
            raise ValueError("Failed to load input MIDI")
        
        initial_content = initial_content.unsqueeze(0).to(device) # (1, 2, T, 128)
        
        # Parse mask
        # Default: Mask nothing (reconstruction)? No, that does nothing.
        # If no mask provided, mask second half?
        inpaint_mask = torch.zeros(1, args.generation_length, device=device)
        if args.mask_ranges:
            ranges = args.mask_ranges.split(',')
            for r in ranges:
                start, end = map(int, r.split('-'))
                inpaint_mask[:, start:end] = 1.0
            print(f"Masking ranges: {args.mask_ranges}")
        else:
            print("No mask provided. Masking last 50% by default.")
            inpaint_mask[:, args.generation_length//2:] = 1.0
            
        print(f"\nInpainting...")
        
        with torch.no_grad():
            result = model.model.sample(
                batch_size=1,
                length=args.generation_length,
                global_cond=None,
                cfg=1.0,
                temperature=args.temperature,
                num_iter_list=[8, 4, 2],
                initial_content=initial_content,
                inpaint_mask=inpaint_mask,
                return_intermediates=args.save_gif
            )
            
        intermediates = None
        if isinstance(result, tuple):
            generated, intermediates = result
        else:
            generated = result
            
        piano_roll = generated[0].cpu().numpy()
        
        # Save MIDI
        midi_path = output_dir / f'inpainting_result.mid'
        score = piano_roll_to_midi(piano_roll, velocity_threshold=0.1)
        score.dump_midi(str(midi_path))
        print(f"✓ Saved MIDI: {midi_path}")
        
        # Save Image
        if args.save_images:
            img_path = output_dir / f'inpainting_result.png'
            # Add dummy tempo
            tempo_ch = np.ones((1, args.generation_length, 128)) * 0.5
            vis_roll = np.concatenate([piano_roll, tempo_ch], axis=0)
            
            img_pil = piano_roll_to_image(
                torch.from_numpy(vis_roll),
                apply_colormap=True,
                return_pil=True,
                composite_tempo=True
            )
            img_pil.save(str(img_path))
            print(f"✓ Saved image: {img_path}")

    else:
        print(f"Mode {args.mode} not implemented.")
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
