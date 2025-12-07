"""
Inference script for FractalGen MIDI model (Temporal Fractal Network).
Supports unconditional generation, conditional generation, and inpainting.

Usage:
    # Using config file
    python src/inference/inference.py --config config/inference_default.yaml --checkpoint path/to/model.ckpt
    
    # Unconditional generation
    python src/inference/inference.py --checkpoint path/to/model.ckpt --mode unconditional --output_dir outputs/samples
"""

import argparse
from pathlib import Path
import torch
import symusic
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import yaml
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add code folder to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent)) 

from src.training.trainer import FractalMIDILightningModule
from src.visualization.visualizer import piano_roll_to_image
from src.inference.metrics import MusicMetrics, MetricsConfig

# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class UnconditionalConfig:
    """Configuration for unconditional generation."""
    num_samples: int = 4
    length: int = 256
    temperature: float = 1.0

@dataclass
class OutputConfig:
    """Configuration for output handling."""
    output_dir: str = "outputs/inference"
    save_png: bool = True
    save_gif: bool = False

@dataclass
class InferenceConfig:
    """Master inference configuration."""
    checkpoint: Optional[str] = None
    mode: str = "unconditional"
    device: str = "cuda:0"
    unconditional: UnconditionalConfig = field(default_factory=UnconditionalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluate: bool = False

    @classmethod
    def from_dict(cls, cfg: dict):
        """Create configuration from dictionary."""
        return cls(
            checkpoint=cfg.get('checkpoint'),
            mode=cfg.get('mode', 'unconditional'),
            device=cfg.get('device', 'cuda:0'),
            unconditional=UnconditionalConfig(**cfg.get('unconditional', {})),
            output=OutputConfig(**cfg.get('output', {})),
            evaluate=cfg.get('evaluate', False)
        )

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_config_with_args(config_dict, args):
    """Merge YAML config with command line arguments using strict Dataclasses."""
    
    if config_dict is None:
        config_dict = {}

    # 1. Load into Dataclass
    try:
        full_config = InferenceConfig.from_dict(config_dict)
    except TypeError as e:
        print(f"Error loading configuration: {e}")
        raise

    # 2. Override with Command Line Arguments
    if args.checkpoint:
        full_config.checkpoint = args.checkpoint
    
    if args.mode != 'unconditional': # Default arg value check
        full_config.mode = args.mode
    
    if args.device != 'cuda:0':
        full_config.device = args.device
        
    if args.evaluate:
        full_config.evaluate = True

    # Unconditional overrides
    if args.num_samples != 4:
        full_config.unconditional.num_samples = args.num_samples
    if args.generation_length != 256:
        full_config.unconditional.length = args.generation_length
    if args.temperature != 1.0:
        full_config.unconditional.temperature = args.temperature

    # Output overrides
    if args.output_dir != 'outputs/inference':
        full_config.output.output_dir = args.output_dir
    if args.save_images: # Action store_true, default False usually, but here default is implied True in config
        # The arg is 'save_images', config is 'save_png'. 
        # If user explicitly sets flag, we enable it? 
        # Wait, the original arg definition was: parser.add_argument('--save_images', action='store_true', help='Save piano roll images')
        # So default is False. If passed, it becomes True.
        # The config default is True. 
        # This logic is tricky. If user doesn't pass flag (False), do we overwrite config (True)?
        # Probably we should respect config if arg is default.
        # But here arg default is False. 
        # Let's assume if arg is True, we force True. If arg is False, we keep config.
        full_config.output.save_png = full_config.output.save_png or args.save_images

    if args.save_gif:
        full_config.output.save_gif = True

    # 3. Flatten to Namespace for compatibility
    merged = argparse.Namespace()
    merged.checkpoint = full_config.checkpoint
    merged.mode = full_config.mode
    merged.device = full_config.device
    merged.evaluate = full_config.evaluate
    
    merged.num_samples = full_config.unconditional.num_samples
    merged.generation_length = full_config.unconditional.length
    merged.temperature = full_config.unconditional.temperature
    
    merged.output_dir = full_config.output.output_dir
    merged.save_images = full_config.output.save_png
    merged.save_gif = full_config.output.save_gif
    
    return merged

def create_generation_gif(intermediates, output_path, fps=15):
    try:
        from src.visualization.visualizer import create_growth_animation
        
        sample_frames = []
        for item in intermediates:
            # Check for non-visual items (like influence data)
            if 'output' not in item: continue
            
            frame = item['output'] 
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
    score = symusic.Score()
    score.ticks_per_quarter = ticks_per_16th * 4
    
    track = symusic.Track()
    track.name = "FractalPiano"
    track.program = 0 
    track.is_drum = False
    
    C, T, H = piano_roll.shape
    
    note_layer = piano_roll[0] 
    vel_layer = piano_roll[1]  
    
    if C >= 3:
        tempo_layer = piano_roll[2] 
        tempo_curve_arr = tempo_layer.mean(axis=1)
    elif tempo_curve is not None:
        tempo_curve_arr = tempo_curve
    else:
        tempo_curve_arr = np.ones(T) * 0.5 
    
    for pitch in range(H):
        is_note_on = False
        note_start = 0
        current_vel_sum = 0.0
        current_vel_count = 0
        
        for t in range(T):
            activation = note_layer[t, pitch]
            velocity = vel_layer[t, pitch]
            
            if activation > 0.5: 
                if not is_note_on:
                    is_note_on = True
                    note_start = t
                    current_vel_sum = velocity
                    current_vel_count = 1
                else:
                    current_vel_sum += velocity
                    current_vel_count += 1
            else:
                if is_note_on:
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
    
    bpm_curve = tempo_curve_arr * 160.0 + 40.0
    
    last_bpm = -1.0
    for t in range(T):
        current_bpm = bpm_curve[t]
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
    try:
        score = symusic.Score(midi_path)
        score = score.resample(120, min_dur=1)
        
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
        return torch.from_numpy(notes) 
    except Exception as e:
        print(f"Error loading MIDI {midi_path}: {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='FractalGen MIDI Inference')
    parser.add_argument('-c', '--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='unconditional', choices=['unconditional', 'inpainting'], help='Generation mode')
    parser.add_argument('--save_gif', action='store_true', help='Save generation process as GIF')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to generate')
    parser.add_argument('--generation_length', type=int, default=256, help='Length of generated sequence')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/inference', help='Output directory')
    parser.add_argument('--save_images', action='store_true', help='Save piano roll images')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--evaluate', action='store_true', help='Calculate objective metrics for generated samples')
    return parser.parse_args()

def main():
    cmd_args = parse_args()
    config = None
    if cmd_args.config:
        config_path = Path(cmd_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: {config_path}")
        config = load_config(config_path)
    
    args = merge_config_with_args(config, cmd_args)
    
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"FractalGen MIDI Inference (Temporal Fractal Network)")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if Path(args.checkpoint).exists():
        model = FractalMIDILightningModule.load_from_checkpoint(args.checkpoint)
        print(f"✓ Loaded from checkpoint: {args.checkpoint}")
    else:
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    model = model.to(device)
    model.eval()
    
    generated_rolls = []
    
    if args.mode == 'unconditional':
        for i in range(args.num_samples):
            print(f"\nGenerating sample {i+1}/{args.num_samples}...")
            
            return_intermediates = (i == 0 and args.save_gif) or (i == 0) # Always return intermediates for influence vis if first sample
            
            with torch.no_grad():
                bar_pos = (torch.arange(args.generation_length, device=device) % 16).unsqueeze(0)
                
                result = model.model.sample(
                    batch_size=1,
                    length=args.generation_length,
                    global_cond=None,
                    cfg=1.0,
                    temperature=args.temperature,
                    num_iter_list=[8, 4, 2],
                    return_intermediates=return_intermediates,
                    bar_pos=bar_pos
                )
            
            intermediates = None
            if isinstance(result, tuple):
                generated, intermediates = result
            else:
                generated = result
            
            piano_roll = generated[0].cpu().numpy() 
            generated_rolls.append(piano_roll)
            
            # Extract tempo from intermediates
            tempo_curve = None
            if intermediates:
                for item in reversed(intermediates):
                    if isinstance(item, dict) and item.get('is_structure', False):
                        struct = item['output'] 
                        tempo_low = struct[0, 1]
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
            
            # Save Image
            if args.save_images:
                img_path = output_dir / f'unconditional_{i:03d}.png'
                if tempo_curve is None:
                    tempo_curve = np.ones(piano_roll.shape[1]) * 0.5
                
                tempo_ch = np.tile(tempo_curve[:, None], (1, 128))
                vis_roll = np.concatenate([piano_roll, tempo_ch[None, :, :]], axis=0)
                
                img_pil = piano_roll_to_image(
                    torch.from_numpy(vis_roll),
                    apply_colormap=True,
                    return_pil=True,
                    composite_tempo=True
                )
                img_pil.save(str(img_path))
            
            # Visualize Condition Influence (New)
            if intermediates and i == 0:
                influence_data = None
                for item in intermediates:
                    if isinstance(item, dict) and item.get('type') == 'condition_influence':
                        influence_data = item['data']
                        break
                
                if influence_data:
                    print("Saved condition influence data to JSON")
                    # Just save raw values for now or could visualize heatmaps
                    infl_path = output_dir / f'unconditional_{i:03d}_influence.json'
                    # Convert tensors to lists
                    serializable_infl = []
                    for level_inf in influence_data:
                        serializable_infl.append({
                            'level': level_inf['level'],
                            'cond_map_shape': list(level_inf['cond_map'].shape)
                        })
                    with open(infl_path, 'w') as f:
                        json.dump(serializable_infl, f, indent=2)

            if intermediates and args.save_gif and i == 0:
                content_intermediates = []
                for item in intermediates:
                    if isinstance(item, dict) and 'output' in item and not item.get('is_structure', False):
                        frame = item['output'][0] 
                        T_frame = frame.shape[1]
                        tempo_frm = torch.ones(1, T_frame, 128) * 0.5
                        full_frame = torch.cat([frame, tempo_frm], dim=0)
                        content_intermediates.append({'output': full_frame.unsqueeze(0)})
                
                gif_path = output_dir / f'unconditional_{i:03d}_process.gif'
                create_generation_gif(content_intermediates, str(gif_path), fps=15)

    # Evaluate
    if args.evaluate and generated_rolls:
        print("\nCalculating objective metrics...")
        metrics = MusicMetrics()
        results = metrics.evaluate_batch(generated_rolls)
        
        print("-" * 40)
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"  {k:<20}: {v:.4f}")
        print("-" * 40)
        
        # Save metrics
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
