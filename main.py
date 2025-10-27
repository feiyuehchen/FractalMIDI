"""
Main training script for FractalGen MIDI model.
Simplified version using the new FractalTrainerConfig.

Usage:
    python main_fractalgen.py --model_size small --batch_size 8 --max_epochs 50
    python main_fractalgen.py --model_size base --batch_size 4 --max_epochs 100 --devices 0,1
"""

import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import create_dataloader
from trainer import FractalMIDILightningModule, FractalTrainerConfig, create_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FractalGen MIDI model')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default='dataset/train.txt',
                       help='Path to training data list')
    parser.add_argument('--val_data', type=str, default='dataset/valid.txt',
                       help='Path to validation data list')
    
    # Model configuration (single architecture: 128→16→4→1)
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                       help='Gradient clipping value')
    
    # Hardware configuration
    parser.add_argument('--devices', type=str, default='0,1',
                       help='Comma-separated GPU indices (e.g., "0,1")')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--precision', type=str, default='32',
                       choices=['32', '16', 'bf16'],
                       help='Training precision')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/fractalgen',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Save top K checkpoints')
    
    # Debugging
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick debug pass')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Parse GPU indices
    gpu_indices = [int(x) for x in args.devices.split(',')]
    print(f"\n{'='*70}")
    print(f"FractalGen MIDI Training")
    print(f"{'='*70}")
    print(f"Model: FractalGen (128→16→4→1, 13.4M params)")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"GPUs: {gpu_indices}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer configuration
    print("Creating configuration...")
    config = FractalTrainerConfig(
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        log_images_every_n_epochs=1,
        num_images_to_log=4
    )
    
    # Override optimizer/scheduler if needed
    config.optimizer.lr = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.scheduler.warmup_epochs = args.warmup_epochs
    
    print(f"✓ Model: FractalGen (128→16→4→1)")
    print(f"✓ Learning rate: {args.lr}")
    print(f"✓ Warmup epochs: {args.warmup_epochs}")
    print(f"✓ Grad clip: {args.grad_clip}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = create_dataloader(
        file_list_path=args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = create_dataloader(
        file_list_path=args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = FractalMIDILightningModule(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params/1e6:.2f}M")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='epoch_{epoch:03d}-val_loss_{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs'
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = create_trainer(
        config,
        devices=gpu_indices,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=output_dir,
        log_every_n_steps=args.log_every_n_steps,
        fast_dev_run=args.fast_dev_run,
        precision=args.precision
    )
    
    print(f"✓ Trainer configured")
    print(f"✓ Strategy: {trainer.strategy}")
    print(f"✓ Precision: {args.precision}")
    
    # Start training
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        print(f"\n{'='*70}")
        print("Training completed successfully!")
        print(f"{'='*70}")
        print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
        print(f"TensorBoard logs: {output_dir / 'logs'}")
        print(f"\nView logs with: tensorboard --logdir {output_dir}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

