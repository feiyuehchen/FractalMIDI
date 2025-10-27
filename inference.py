"""
Inference script for FractalGen MIDI model.
Supports unconditional generation, conditional generation, and inpainting.

Usage:
    # Unconditional generation
    python inference_fractalgen.py --checkpoint path/to/model.ckpt --mode unconditional --output_dir outputs/samples

    # Conditional generation (prefix-based)
    python inference_fractalgen.py --checkpoint path/to/model.ckpt --mode conditional \\
        --condition_midi input.mid --generation_length 256 --output_dir outputs/conditional

    # Inpainting
    python inference_fractalgen.py --checkpoint path/to/model.ckpt --mode inpainting \\
        --input_midi input.mid --mask_start 64 --mask_end 192 --output_dir outputs/inpainting
"""

import argparse
from pathlib import Path
import torch
import symusic
from PIL import Image
import numpy as np

from trainer import FractalMIDILightningModule
from model import fractalmar_piano
from visualizer import piano_roll_to_image


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
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Generation mode
    parser.add_argument('--mode', type=str, default='unconditional',
                       choices=['unconditional', 'conditional', 'inpainting'],
                       help='Generation mode')
    
    # Unconditional generation
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples to generate (unconditional mode)')
    parser.add_argument('--generation_length', type=int, default=256,
                       help='Length of generated sequence (in 16th notes)')
    
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
    parser.add_argument('--sparsity_bias', type=float, default=2.0,
                       help='Bias to encourage sparsity (higher = sparser)')
    
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
    args = parse_args()
    
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
            
            with torch.no_grad():
                # Generate using model's sample method
                generated = model.model.sample(
                    batch_size=1,
                    cond_list=None,
                    num_iter_list=args.num_iter_list,
                    cfg=args.cfg,
                    cfg_schedule='constant',
                    temperature=args.temperature,
                    filter_threshold=0.0,
                    fractal_level=0
                )
            
            # Apply sparsity bias
            generated = torch.sigmoid(generated - args.sparsity_bias)
            
            # Convert to piano roll
            piano_roll = generated[0, 0].cpu().numpy()  # (128, T)
            
            # Trim to desired length
            piano_roll = piano_roll[:, :args.generation_length]
            
            # Save MIDI
            midi_path = output_dir / f'unconditional_{i:03d}.mid'
            score = piano_roll_to_midi(piano_roll)
            score.dump_midi(str(midi_path))
            print(f"✓ Saved MIDI: {midi_path}")
            
            # Save image if requested
            if args.save_images:
                img_path = output_dir / f'unconditional_{i:03d}.png'
                img_tensor = piano_roll_to_image(piano_roll)
                # Convert tensor to PIL Image if needed
                if torch.is_tensor(img_tensor):
                    from torchvision.utils import save_image
                    save_image(img_tensor, str(img_path))
                else:
                    img_tensor.save(str(img_path))
                print(f"✓ Saved image: {img_path}")
    
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
        
        # Apply sparsity bias
        generated = torch.sigmoid(generated - args.sparsity_bias)
        
        piano_roll = generated[0, 0].cpu().numpy()
        
        # Save MIDI
        midi_path = output_dir / 'conditional_output.mid'
        score = piano_roll_to_midi(piano_roll)
        score.dump_midi(str(midi_path))
        print(f"✓ Saved MIDI: {midi_path}")
        
        # Save image
        if args.save_images:
            img_path = output_dir / 'conditional_output.png'
            img_tensor = piano_roll_to_image(piano_roll)
            # Convert tensor to PIL Image if needed
            if torch.is_tensor(img_tensor):
                from torchvision.utils import save_image
                save_image(img_tensor, str(img_path))
            else:
                img_tensor.save(str(img_path))
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
        
        # Apply sparsity bias
        generated = torch.sigmoid(generated - args.sparsity_bias)
        
        piano_roll = generated[0, 0].cpu().numpy()
        
        # Save MIDI
        midi_path = output_dir / 'inpainting_output.mid'
        score = piano_roll_to_midi(piano_roll)
        score.dump_midi(str(midi_path))
        print(f"✓ Saved MIDI: {midi_path}")
        
        # Save image
        if args.save_images:
            img_path = output_dir / 'inpainting_output.png'
            img_tensor = piano_roll_to_image(piano_roll)
            # Convert tensor to PIL Image if needed
            if torch.is_tensor(img_tensor):
                from torchvision.utils import save_image
                save_image(img_tensor, str(img_path))
            else:
                img_tensor.save(str(img_path))
            print(f"✓ Saved image: {img_path}")
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

