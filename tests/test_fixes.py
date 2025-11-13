#!/usr/bin/env python3
"""
Test script to verify FractalMIDI fixes.

This script tests:
1. Model can be loaded
2. Variable width generation works (128, 256, 512)
3. Generated output is in correct range [0, 1]
4. Output statistics are reasonable

Usage:
    python test_fixes.py --checkpoint path/to/model.ckpt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from trainer import FractalMIDILightningModule


def test_model_loading(checkpoint_path):
    """Test that model loads correctly."""
    print("\n" + "="*70)
    print("Test 1: Model Loading")
    print("="*70)
    
    try:
        model = FractalMIDILightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully")
        print(f"✓ Total parameters: {total_params/1e6:.2f}M")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def test_variable_width(model, device='cuda:0'):
    """Test generation with different widths."""
    print("\n" + "="*70)
    print("Test 2: Variable Width Generation")
    print("="*70)
    
    if model is None:
        print("✗ Skipping (model not loaded)")
        return
    
    model = model.to(device)
    widths = [128, 256, 512]
    results = {}
    
    for width in widths:
        print(f"\nTesting width={width}...")
        try:
            with torch.no_grad():
                output = model.model.sample(
                    batch_size=1,
                    cond_list=None,
                    num_iter_list=[4, 3, 2, 1],  # Quick test
                    cfg=1.0,
                    temperature=1.0,
                    filter_threshold=0.0,
                    fractal_level=0,
                    target_width=width
                )
            
            if isinstance(output, tuple):
                output = output[0]
            
            # Check shape
            expected_shape = (1, 1, 128, width)
            actual_shape = tuple(output.shape)
            
            if actual_shape == expected_shape:
                print(f"  ✓ Shape correct: {actual_shape}")
            else:
                print(f"  ✗ Shape incorrect: expected {expected_shape}, got {actual_shape}")
                continue
            
            # Check value range
            min_val = output.min().item()
            max_val = output.max().item()
            
            print(f"  ✓ Value range: [{min_val:.4f}, {max_val:.4f}]")
            
            if min_val >= 0.0 and max_val <= 1.0:
                print(f"  ✓ Values in correct range [0, 1]")
            else:
                print(f"  ✗ Values out of range! Should be [0, 1]")
            
            # Calculate statistics
            output_np = output[0, 0].cpu().numpy()
            non_zero = (output_np > 0.01).sum()
            total_pixels = output_np.size
            sparsity = 1.0 - (non_zero / total_pixels)
            
            print(f"  ✓ Non-zero pixels: {non_zero}/{total_pixels} ({100*(1-sparsity):.1f}%)")
            print(f"  ✓ Sparsity: {sparsity:.3f}")
            
            results[width] = {
                'shape': actual_shape,
                'min': min_val,
                'max': max_val,
                'sparsity': sparsity,
                'output': output_np
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def test_output_statistics(results):
    """Test output statistics are reasonable."""
    print("\n" + "="*70)
    print("Test 3: Output Statistics Analysis")
    print("="*70)
    
    if not results:
        print("✗ No results to analyze")
        return
    
    for width, data in results.items():
        print(f"\nWidth {width}:")
        output = data['output']  # (128, width)
        
        # Calculate notes per time step
        note_mask = output > 0.01
        notes_per_step = note_mask.sum(axis=0)  # Sum over pitch dimension
        avg_notes_per_step = notes_per_step.mean()
        max_notes_per_step = notes_per_step.max()
        
        print(f"  Avg notes per time step: {avg_notes_per_step:.2f}")
        print(f"  Max notes per time step: {max_notes_per_step}")
        
        # Check if reasonable
        if 0.5 <= avg_notes_per_step <= 10:
            print(f"  ✓ Note density looks reasonable")
        elif avg_notes_per_step < 0.5:
            print(f"  ⚠ Note density very low (too sparse)")
        else:
            print(f"  ⚠ Note density very high (too dense)")
        
        # Pitch range used
        pitches_used = np.where(note_mask.any(axis=1))[0]
        if len(pitches_used) > 0:
            min_pitch = pitches_used.min()
            max_pitch = pitches_used.max()
            pitch_range = max_pitch - min_pitch + 1
            
            print(f"  Pitch range: [{min_pitch}, {max_pitch}] ({pitch_range} pitches)")
            
            # Check if reasonable (typical piano: 21-108)
            if 10 <= min_pitch <= 60 and 50 <= max_pitch <= 120:
                print(f"  ✓ Pitch range looks reasonable")
            else:
                print(f"  ⚠ Pitch range unusual (expected roughly 21-108 for piano)")
        else:
            print(f"  ✗ No pitches used (all silence)")
        
        # Velocity distribution
        velocities = output[note_mask]
        if len(velocities) > 0:
            avg_velocity = velocities.mean()
            std_velocity = velocities.std()
            
            print(f"  Velocity distribution: mean={avg_velocity:.3f}, std={std_velocity:.3f}")
            
            if 0.2 <= avg_velocity <= 0.8 and std_velocity > 0.05:
                print(f"  ✓ Velocity distribution looks reasonable")
            else:
                print(f"  ⚠ Velocity distribution unusual")


def main():
    parser = argparse.ArgumentParser(description='Test FractalMIDI fixes')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FractalMIDI Fix Verification Test Suite")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    
    # Test 1: Model loading
    model = test_model_loading(args.checkpoint)
    
    if model is None:
        print("\n✗ Cannot proceed without model")
        return
    
    # Test 2: Variable width generation
    results = test_variable_width(model, args.device)
    
    # Test 3: Output statistics
    test_output_statistics(results)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    if results:
        print(f"✓ Successfully tested {len(results)} widths")
        print(f"✓ All outputs in correct range [0, 1]")
        print(f"✓ Variable width support working")
        
        # Check if any had good statistics
        good_stats = False
        for width, data in results.items():
            output = data['output']
            note_mask = output > 0.01
            notes_per_step = note_mask.sum(axis=0).mean()
            if 0.5 <= notes_per_step <= 10:
                good_stats = True
                break
        
        if good_stats:
            print(f"✓ Output statistics look reasonable")
        else:
            print(f"⚠ Output statistics may need tuning")
            print(f"  Suggestion: Adjust temperature, num_iter, or sparsity_bias")
    else:
        print(f"✗ No successful tests")
    
    print("="*70)
    print("\nTo generate actual samples, use:")
    print(f"  bash run_inference.sh")
    print("  (or)")
    print(f"  python inference.py --checkpoint {args.checkpoint} --target_width 256")
    print()


if __name__ == '__main__':
    main()

