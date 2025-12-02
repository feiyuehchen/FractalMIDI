"""
Test script to verify AR generator fixes.
This script trains a small AR model to verify the fixes work correctly.

Usage:
    python test_ar_fixes.py --quick  # Quick test with tiny model
    python test_ar_fixes.py --full   # Full test with small model
"""

import argparse
import torch
import pytorch_lightning as pl
from pathlib import Path

from trainer import FractalMIDILightningModule, FractalTrainerConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from dataset import create_dataloader, DataLoaderConfig, MIDIDatasetConfig


def test_ar_quick():
    """Quick test with minimal model to verify basic functionality."""
    print("=" * 70)
    print("Quick AR Test - Minimal Model")
    print("=" * 70)
    
    # Tiny model configuration for quick testing
    model_config = ModelConfig(
        # Very small architecture
        img_size_list=(128, 16, 4, 1),
        embed_dim_list=(64, 32, 16, 8),  # Much smaller
        num_blocks_list=(2, 1, 1, 1),    # Fewer blocks
        num_heads_list=(2, 2, 1, 1),
        
        # All AR
        generator_type_list=("ar", "ar", "ar", "ar"),
        scan_order='row_major',
        
        # Training settings
        attn_dropout=0.0,
        proj_dropout=0.0,
        grad_checkpointing=False,
        
        # AR-specific
        ar_prefix_mask_ratio=1.0,
    )
    
    optimizer_config = OptimizerConfig(
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,  # Lower for small model
    )
    
    scheduler_config = SchedulerConfig(
        schedule_type="cosine",
        warmup_steps=100,  # Short warmup for quick test
        min_lr=1e-6,
    )
    
    trainer_config = FractalTrainerConfig(
        max_steps=500,  # Just 500 steps for quick test
        grad_clip=1.0,
        accumulate_grad_batches=1,
        precision="32",
        log_every_n_steps=10,
        val_check_interval_steps=100,
        checkpoint_every_n_steps=100,
        log_images_every_n_steps=100,
        num_images_to_log=1,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        model=model_config,
    )
    
    # Create model
    model = FractalMIDILightningModule(trainer_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Test forward pass with dummy data
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 1, 128, 128)  # Small batch, small width
    try:
        with torch.no_grad():
            loss, stats = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Stats: {list(stats.keys())[:5]}...")  # Show first 5 stat keys
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation
    print("\nTesting generation...")
    try:
        with torch.no_grad():
            generated = model.model.sample(
                batch_size=1,
                cond_list=None,
                num_iter_list=[2, 2, 1, 1],  # Very few iterations for quick test
                cfg=1.0,
                temperature=1.0,
                filter_threshold=0.0,
                target_width=128,  # Small width
            )
        print(f"✓ Generation successful")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Value range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
        
        # Check if generation is not all zeros or all same value
        if generated.std().item() < 1e-6:
            print(f"⚠ Warning: Generated output has very low variance (std={generated.std().item():.6f})")
        else:
            print(f"✓ Generated output has reasonable variance (std={generated.std().item():.3f})")
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("Quick test completed successfully!")
    print("=" * 70)
    return True


def test_ar_full():
    """Full test with small model and actual training."""
    print("=" * 70)
    print("Full AR Test - Small Model Training")
    print("=" * 70)
    
    # Check if dataset exists
    train_file = Path("dataset/train.txt")
    if not train_file.exists():
        print(f"✗ Training data not found: {train_file}")
        print("  Please prepare dataset first.")
        return False
    
    # Small model configuration
    model_config = ModelConfig(
        img_size_list=(128, 16, 4, 1),
        embed_dim_list=(128, 64, 32, 16),  # Small but reasonable
        num_blocks_list=(4, 2, 1, 1),
        num_heads_list=(4, 2, 1, 1),
        
        # All AR
        generator_type_list=("ar", "ar", "ar", "ar"),
        scan_order='row_major',
        
        # Training settings
        attn_dropout=0.0,
        proj_dropout=0.0,
        grad_checkpointing=False,
    )
    
    optimizer_config = OptimizerConfig(
        lr=5e-5,  # Lower LR for AR
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    
    scheduler_config = SchedulerConfig(
        schedule_type="cosine",
        warmup_steps=1000,
        min_lr=1e-6,
    )
    
    trainer_config = FractalTrainerConfig(
        max_steps=5000,  # 5k steps for full test
        grad_clip=1.0,
        accumulate_grad_batches=1,
        precision="32",
        log_every_n_steps=50,
        val_check_interval_steps=500,
        checkpoint_every_n_steps=500,
        log_images_every_n_steps=500,
        num_images_to_log=2,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        model=model_config,
    )
    
    # Create model
    model = FractalMIDILightningModule(trainer_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        train_loader_config = DataLoaderConfig.training_default(
            file_list_path="dataset/train.txt"
        )
        train_loader_config.dataset.random_crop = True
        train_loader_config.dataset.crop_length = 128  # Small crops for testing
        train_loader_config.sampler.batch_size = 4  # Small batch
        
        val_loader_config = DataLoaderConfig.validation_default(
            file_list_path="dataset/valid.txt"
        )
        val_loader_config.dataset.random_crop = True
        val_loader_config.dataset.crop_length = 128
        val_loader_config.sampler.batch_size = 4
        
        train_loader = create_dataloader(train_loader_config)
        val_loader = create_dataloader(val_loader_config)
        
        print(f"✓ Dataloaders created")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"✗ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create trainer
    print("\nCreating trainer...")
    from trainer import create_trainer
    
    output_dir = Path("outputs/ar_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="ar_test",
        version="v1"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="ar-test-{step:06d}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        every_n_train_steps=500,
    )
    
    trainer = create_trainer(
        trainer_config,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=500,
    )
    
    print(f"✓ Trainer created")
    
    # Train
    print("\nStarting training...")
    print("Monitor with: tensorboard --logdir outputs/ar_test")
    print("-" * 70)
    
    try:
        trainer.fit(model, train_loader, val_loader)
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test final generation
    print("\nTesting final generation...")
    model.eval()
    try:
        with torch.no_grad():
            generated = model.model.sample(
                batch_size=1,
                cond_list=None,
                num_iter_list=[8, 4, 2, 1],
                cfg=1.0,
                temperature=1.0,
                filter_threshold=0.0,
                target_width=256,
            )
        print(f"✓ Final generation successful")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Value range: [{generated.min().item():.3f}, {generated.max().item():.3f}]")
        print(f"  Std: {generated.std().item():.3f}")
        
        # Save generated sample
        from visualizer import piano_roll_to_image
        from inference import piano_roll_to_midi
        
        piano_roll = generated[0, 0].cpu().numpy()
        
        # Save image
        img_path = output_dir / "test_generation.png"
        img = piano_roll_to_image(torch.from_numpy(piano_roll), apply_colormap=True, return_pil=True)
        img.save(str(img_path))
        print(f"  Saved image: {img_path}")
        
        # Save MIDI
        midi_path = output_dir / "test_generation.mid"
        score = piano_roll_to_midi(piano_roll, velocity_threshold=0.1)
        score.dump_midi(str(midi_path))
        print(f"  Saved MIDI: {midi_path}")
        
    except Exception as e:
        print(f"✗ Final generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("Full test completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Test AR generator fixes")
    parser.add_argument("--quick", action="store_true", help="Quick test with minimal model")
    parser.add_argument("--full", action="store_true", help="Full test with training")
    args = parser.parse_args()
    
    if not args.quick and not args.full:
        print("Please specify --quick or --full")
        parser.print_help()
        return
    
    if args.quick:
        success = test_ar_quick()
    elif args.full:
        success = test_ar_full()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        exit(1)


if __name__ == "__main__":
    main()

