"""
Inference script for FractalGen MIDI model.
Supports unconditional generation, conditional generation, and inpainting.

Usage:
    # Using config file
    python inference.py --config config/inference_default.yaml --checkpoint path/to/model.ckpt
    
    # Unconditional generation (legacy)
    python inference.py --checkpoint path/to/model.ckpt --mode unconditional --output_dir outputs/samples

    # Conditional generation (prefix-based)
    python inference.py --checkpoint path/to/model.ckpt --mode conditional \\
        --condition_midi input.mid --generation_length 512 --output_dir outputs/conditional

    # Inpainting
    python inference.py --checkpoint path/to/model.ckpt --mode inpainting \\
        --input_midi input.mid --mask_start 64 --mask_end 192 --output_dir outputs/inpainting
"""

import argparse
from pathlib import Path
import torch
import symusic
from PIL import Image
import numpy as np
import yaml

from trainer import FractalMIDILightningModule
from model import fractalmar_piano
from visualizer import piano_roll_to_image


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """Merge YAML config with command line arguments.
    
    Command line arguments take precedence over config file values.
    """
    if config is None:
        return args
    
    merged = argparse.Namespace()
    
    # Checkpoint (required)
    merged.checkpoint = args.checkpoint if args.checkpoint else config.get('checkpoint')
    
    # Mode
    merged.mode = args.mode if args.mode != 'unconditional' else config.get('mode', 'unconditional')
    merged.save_gif = args.save_gif if args.save_gif else config.get('output', {}).get('save_gif', False)
    
    # Unconditional settings
    unconditional_cfg = config.get('unconditional', {})
    merged.num_samples = args.num_samples if args.num_samples != 4 else unconditional_cfg.get('num_samples', 10)
    merged.generation_length = args.generation_length if args.generation_length != 512 else unconditional_cfg.get('length', 512)
    merged.target_width = args.target_width if args.target_width != 512 else unconditional_cfg.get('length', 512)
    merged.temperature = args.temperature if args.temperature != 1.0 else unconditional_cfg.get('temperature', 1.0)
    
    # Get sparsity_bias and velocity_threshold from config based on mode
    mode_cfg = config.get(merged.mode, {})
    merged.sparsity_bias = args.sparsity_bias if args.sparsity_bias != 0.0 else mode_cfg.get('sparsity_bias', 0.0)
    merged.velocity_threshold = args.velocity_threshold if args.velocity_threshold != 0.10 else mode_cfg.get('velocity_threshold', 0.10)
    
    # Conditional settings
    conditional_cfg = config.get('conditional', {})
    merged.condition_midi = args.condition_midi if args.condition_midi else conditional_cfg.get('condition_midi')
    merged.condition_length = args.condition_length if args.condition_length != 64 else 64
    
    # Inpainting settings
    inpainting_cfg = config.get('inpainting', {})
    merged.input_midi = args.input_midi if args.input_midi else inpainting_cfg.get('input_midi')
    merged.mask_start = args.mask_start if args.mask_start != 64 else inpainting_cfg.get('mask_start', 64)
    merged.mask_end = args.mask_end if args.mask_end != 192 else inpainting_cfg.get('mask_end', 192)
    
    # Sampling parameters
    merged.num_iter_list = args.num_iter_list if args.num_iter_list != [12, 8, 4, 1] else [12, 8, 4, 1]
    merged.cfg = args.cfg
    merged.sparsity_bias = args.sparsity_bias
    merged.velocity_threshold = args.velocity_threshold
    
    # Output
    output_cfg = config.get('output', {})
    merged.output_dir = args.output_dir if args.output_dir != 'outputs/inference' else output_cfg.get('output_dir', 'outputs/inference')
    merged.save_images = args.save_images if args.save_images else output_cfg.get('save_png', True)
    
    # Hardware
    merged.device = args.device if args.device != 'cuda:0' else config.get('device', 'cuda:0')
    
    return merged


def create_generation_gif(intermediates, output_path, fps=2):
    """
    Create a GIF from intermediate generation steps.
    
    Args:
        intermediates: List of dicts with 'output', 'level', 'iteration', 'level_name' keys
        output_path: Path to save GIF
        fps: Frames per second
    """
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    
    frames = []
    
    for frame_data in intermediates:
        # Extract piano roll (B, C, H, W)
        piano_roll = frame_data['output'][0, 0]  # (128, W)
        
        # Convert to image
        img = piano_roll_to_image(
            piano_roll,
            apply_colormap=True,
            return_pil=True
        )
        
        # Add text annotation
        draw = ImageDraw.Draw(img)
        text = frame_data.get('level_name', f"Level {frame_data['level']}, Iter {frame_data['iteration']}")
        
        # Try to use a better font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw text with background
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle([(bbox[0]-5, bbox[1]-5), (bbox[2]+5, bbox[3]+5)], fill='black')
        draw.text((10, 10), text, fill='white', font=font)
        
        frames.append(np.array(img))
    
    # Save as GIF
    imageio.mimsave(
        output_path,
        frames,
        fps=fps,
        loop=0  # Infinite loop
    )


def piano_roll_to_midi(piano_roll, ticks_per_16th=120, velocity_threshold=0.10):
    """
    Convert piano roll (H x W) to MIDI file.
    
    Args:
        piano_roll: (128, T) numpy array with values in [0, 1]
        ticks_per_16th: Ticks per 16th note
        velocity_threshold: Minimum velocity to create a note
    
    Returns:
        symusic.Score object
    """
    score = symusic.Score()
    score.ticks_per_quarter = ticks_per_16th * 4  # 4 * 16th = quarter
    
    # Create a piano track
    track = symusic.Track()
    track.name = "Piano"
    track.is_drum = False
    
    # Convert piano roll to notes
    H, W = piano_roll.shape  # (128, T)
    
    for pitch in range(H):
        is_note_on = False
        note_start = 0
        
        for t in range(W):
            velocity_norm = piano_roll[pitch, t]
            
            if velocity_norm > velocity_threshold:
                if not is_note_on:
                    # Note on
                    is_note_on = True
                    note_start = t
            else:
                if is_note_on:
                    # Note off - create note
                    is_note_on = False
                    note_duration = t - note_start
                    if note_duration > 0:
                        # Calculate average velocity during note
                        avg_velocity = piano_roll[pitch, note_start:t].mean()
                        velocity_int = int(avg_velocity * 127)
                        velocity_int = max(1, min(127, velocity_int))
                        
                        note = symusic.Note(
                            time=note_start * ticks_per_16th,
                            duration=note_duration * ticks_per_16th,
                            pitch=pitch,
                            velocity=velocity_int
                        )
                        track.notes.append(note)
        
        # Handle note that extends to the end
        if is_note_on:
            note_duration = W - note_start
            if note_duration > 0:
                avg_velocity = piano_roll[pitch, note_start:W].mean()
                velocity_int = int(avg_velocity * 127)
                velocity_int = max(1, min(127, velocity_int))
                
                note = symusic.Note(
                    time=note_start * ticks_per_16th,
                    duration=note_duration * ticks_per_16th,
                    pitch=pitch,
                    velocity=velocity_int
                )
                track.notes.append(note)
    
    score.tracks.append(track)
    return score


def midi_to_piano_roll(midi_path, ticks_per_16th=120, target_length=None):
    """
    Load MIDI file and convert to piano roll.
    
    Args:
        midi_path: Path to MIDI file
        ticks_per_16th: Ticks per 16th note
        target_length: If specified, crop/pad to this length
    
    Returns:
        (128, T) numpy array
    """
    score = symusic.Score(midi_path)
    
    # Calculate total length in 16th notes
    total_ticks = max(
        note.end
        for track in score.tracks
        for note in track.notes
    ) if any(track.notes for track in score.tracks) else ticks_per_16th * 16
    
    length = (total_ticks + ticks_per_16th - 1) // ticks_per_16th
    
    if target_length is not None:
        length = target_length
    
    # Initialize piano roll
    piano_roll = np.zeros((128, length), dtype=np.float32)
    
    # Fill in notes
    for track in score.tracks:
        if track.is_drum:
            continue
        
        for note in track.notes:
            start_16th = note.time // ticks_per_16th
            end_16th = note.end // ticks_per_16th
            
            if start_16th >= length:
                continue
            
            end_16th = min(end_16th, length)
            
            velocity_norm = note.velocity / 127.0
            piano_roll[note.pitch, start_16th:end_16th] = max(
                piano_roll[note.pitch, start_16th:end_16th].max(),
                velocity_norm
            )
    
    return piano_roll


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FractalGen MIDI Inference')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (e.g., config/inference_default.yaml)')
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='unconditional',
                       choices=['unconditional', 'conditional', 'inpainting'],
                       help='Generation mode')
    parser.add_argument('--save_gif', action='store_true',
                       help='Save generation process as GIF for the first sample')
    
    # Unconditional generation
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to generate (unconditional mode)')
    parser.add_argument('--generation_length', type=int, default=512,
                       help='Length of generated sequence (in 16th notes, default 512 for 128x512)')
    parser.add_argument('--target_width', type=int, default=512,
                       help='Target width for generation (128, 256, 512, etc.). Default: 512 (for 128x512), 256 (for 128x256), 128 (for 128x128)')
    
    # Conditional generation
    parser.add_argument('--condition_midi', type=str,
                       help='Condition MIDI file (for conditional mode)')
    parser.add_argument('--condition_length', type=int, default=64,
                       help='Length of conditioning (in 16th notes)')
    
    # Inpainting
    parser.add_argument('--input_midi', type=str,
                       help='Input MIDI file (for inpainting mode)')
    parser.add_argument('--mask_start', type=int, default=64,
                       help='Start of mask region (in 16th notes)')
    parser.add_argument('--mask_end', type=int, default=192,
                       help='End of mask region (in 16th notes)')
    
    # Sampling parameters
    parser.add_argument('--num_iter_list', type=int, nargs='+', default=[12, 8, 4, 1],
                       help='Number of iterations per level (4 levels: 128→16→4→1)')
    parser.add_argument('--cfg', type=float, default=1.0,
                       help='Classifier-free guidance strength')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--sparsity_bias', type=float, default=0.0,
                       help='Bias to encourage sparsity (higher = sparser, 0 = no bias)')
    parser.add_argument('--velocity_threshold', type=float, default=0.10,
                       help='Minimum velocity to create a note (0-1)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
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
    print(f"FractalGen MIDI Inference")
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
        # Create new model (for testing)
        print(f"⚠️  Checkpoint not found, creating new model")
        from trainer import FractalTrainerConfig
        config = FractalTrainerConfig()
        model = FractalMIDILightningModule(config)
    
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
            
            # Enable intermediate recording for first sample if GIF requested
            return_intermediates = (i == 0 and args.save_gif)
            
            with torch.no_grad():
                # Generate using model's sample method
                result = model.model.sample(
                    batch_size=1,
                    cond_list=None,
                    num_iter_list=args.num_iter_list,
                    cfg=args.cfg,
                    cfg_schedule='constant',
                    temperature=args.temperature,
                    filter_threshold=0.0,
                    fractal_level=0,
                    target_width=args.target_width,
                    return_intermediates=return_intermediates
                )
            
            # Extract tensor and intermediates
            intermediates = None
            if isinstance(result, tuple):
                generated, intermediates = result
            else:
                generated = result
            
            # Apply sparsity bias if specified (default is 0 = no bias, same as training)
            if args.sparsity_bias != 0.0:
                generated = generated - args.sparsity_bias
                if args.sparsity_bias < 1.0:
                    generated = generated / (1.0 - args.sparsity_bias)
                generated = torch.clamp(generated, min=0.0, max=1.0)
            
            # Convert to piano roll
            piano_roll = generated[0, 0].cpu().numpy()  # (128, T)
            
            # Trim to desired length
            piano_roll = piano_roll[:, :args.generation_length]
            
            # Save MIDI
            midi_path = output_dir / f'unconditional_{i:03d}.mid'
            score = piano_roll_to_midi(piano_roll, velocity_threshold=args.velocity_threshold)
            score.dump_midi(str(midi_path))
            print(f"✓ Saved MIDI: {midi_path}")
            
            # Always save image (like training does)
            img_path = output_dir / f'unconditional_{i:03d}.png'
            # Use the same visualization as training
            img_pil = piano_roll_to_image(
                torch.from_numpy(piano_roll),
                apply_colormap=True,
                return_pil=True
            )
            img_pil.save(str(img_path))
            print(f"✓ Saved image: {img_path}")
            
            # Save GIF if intermediates recorded
            if intermediates is not None and len(intermediates) > 0:
                print(f"\nCreating generation process GIF...")
                gif_path = output_dir / f'unconditional_{i:03d}_process.gif'
                create_generation_gif(intermediates, gif_path, fps=2)
                print(f"✓ Saved GIF: {gif_path}")
                print(f"  Total frames: {len(intermediates)}")
    
    elif args.mode == 'conditional':
        # Conditional generation (prefix-based)
        if not args.condition_midi:
            print("Error: --condition_midi is required for conditional mode")
            return
        
        print(f"Loading condition from: {args.condition_midi}")
        condition_roll = midi_to_piano_roll(
            args.condition_midi,
            target_length=args.condition_length
        )
        
        condition_tensor = torch.from_numpy(condition_roll).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            # Use model's conditional generation
            from model import conditional_generation
            generated = conditional_generation(
                model.model,
                condition_tensor,
                args.generation_length,
                args.num_iter_list,
                args.cfg,
                args.temperature,
                0.0
            )
        
        # Extract tensor
        if isinstance(generated, tuple):
            generated = generated[0]
        
        # Apply sparsity bias if specified (default is 0 = no bias, same as training)
        if args.sparsity_bias != 0.0:
            generated = generated - args.sparsity_bias
            if args.sparsity_bias < 1.0:
                generated = generated / (1.0 - args.sparsity_bias)
            generated = torch.clamp(generated, min=0.0, max=1.0)
        
        piano_roll = generated[0, 0].cpu().numpy()
        
        # Save MIDI
        midi_path = output_dir / 'conditional_output.mid'
        score = piano_roll_to_midi(piano_roll, velocity_threshold=args.velocity_threshold)
        score.dump_midi(str(midi_path))
        print(f"✓ Saved MIDI: {midi_path}")
        
        # Always save image
        img_path = output_dir / 'conditional_output.png'
        img_pil = piano_roll_to_image(
            torch.from_numpy(piano_roll),
            apply_colormap=True,
            return_pil=True
        )
        img_pil.save(str(img_path))
        print(f"✓ Saved image: {img_path}")
    
    elif args.mode == 'inpainting':
        # Inpainting
        if not args.input_midi:
            print("Error: --input_midi is required for inpainting mode")
            return
        
        print(f"Loading input from: {args.input_midi}")
        input_roll = midi_to_piano_roll(args.input_midi)
        
        input_tensor = torch.from_numpy(input_roll).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            # Use model's inpainting
            from model import inpainting_generation
            generated = inpainting_generation(
                model.model,
                input_tensor,
                args.mask_start,
                args.mask_end,
                args.num_iter_list,
                args.cfg,
                args.temperature,
                0.0
            )
        
        # Extract tensor
        if isinstance(generated, tuple):
            generated = generated[0]
        
        # Apply sparsity bias if specified (default is 0 = no bias, same as training)
        if args.sparsity_bias != 0.0:
            generated = generated - args.sparsity_bias
            if args.sparsity_bias < 1.0:
                generated = generated / (1.0 - args.sparsity_bias)
            generated = torch.clamp(generated, min=0.0, max=1.0)
        
        piano_roll = generated[0, 0].cpu().numpy()
        
        # Save MIDI
        midi_path = output_dir / 'inpainting_output.mid'
        score = piano_roll_to_midi(piano_roll, velocity_threshold=args.velocity_threshold)
        score.dump_midi(str(midi_path))
        print(f"✓ Saved MIDI: {midi_path}")
        
        # Always save image
        img_path = output_dir / 'inpainting_output.png'
        img_pil = piano_roll_to_image(
            torch.from_numpy(piano_roll),
            apply_colormap=True,
            return_pil=True
        )
        img_pil.save(str(img_path))
        print(f"✓ Saved image: {img_path}")
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

