"""
Main training script for the FractalGen MIDI model (single configuration).

Usage:
    python main_fractalgen.py --train_batch_size 8 --max_steps 200000 --devices 0,1
"""

import argparse
import math
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import create_dataloader, DataLoaderConfig
from trainer import FractalMIDILightningModule, FractalTrainerConfig, create_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FractalGen MIDI model')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default='dataset/train.txt',
                       help='Path to training data list')
    parser.add_argument('--val_data', type=str, default='dataset/valid.txt',
                       help='Path to validation data list')
    
    # Training configuration
    parser.add_argument('--train_batch_size', type=int, default=8,
                       help='Training batch size per device')
    parser.add_argument('--val_batch_size', type=int, default=8,
                       help='Validation batch size per device')
    parser.add_argument('--augment_factor', type=int, default=1,
                       help='Random crop augmentation factor for training data')
    parser.add_argument('--pitch_shift_min', type=int, default=-3,
                       help='Minimum semitone shift for pitch augmentation (inclusive)')
    parser.add_argument('--pitch_shift_max', type=int, default=3,
                       help='Maximum semitone shift for pitch augmentation (inclusive)')
    parser.add_argument('--max_steps', type=int, default=200000,
                       help='Maximum number of optimizer update steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='Number of learning-rate warmup steps')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                       help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Hardware configuration
    parser.add_argument('--devices', type=str, default='0,1',
                       help='Comma-separated GPU indices (e.g., "0,1")')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='Prefetch batches per worker (set 0 to disable)')
    parser.add_argument('--disable_persistent_workers', action='store_true',
                       help='Disable persistent workers in DataLoader')
    parser.add_argument('--precision', type=str, default='32',
                       choices=['32', '16', 'bf16'],
                       help='Training precision')
    parser.add_argument('--grad_checkpoint', action='store_true',
                       help='Enable gradient checkpointing to save memory (slower)')
    parser.add_argument('--no_cache_in_memory', action='store_true',
                       help='Disable in-memory caching of decoded piano rolls')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Optional directory for on-disk piano-roll cache')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable pin_memory in dataloaders')
    parser.add_argument('--use_bucket_sampler', action='store_true',
                       help='Enable bucket sampler for evaluation dataloaders')
    parser.add_argument('--generator_types', type=str, default='mar,mar,mar,mar',
                       help='Comma-separated generator types per level (choices: mar or ar)')
    parser.add_argument('--scan_order', type=str, default='row_major',
                       choices=['row_major', 'column_major'],
                       help='Scanning order for AR generation: row_major (left-to-right, top-to-bottom) or column_major (top-to-bottom, left-to-right)')
    parser.add_argument('--mask_ratio_loc', type=float, default=1.0,
                       help='Mean mask ratio for MAR (1.0 = mask 100%%, 0.5 = mask 50%%)')
    parser.add_argument('--mask_ratio_scale', type=float, default=0.5,
                       help='Standard deviation of mask ratio for MAR')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/fractalgen',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Frequency (in steps) for logging to progress bar/logger')
    parser.add_argument('--val_check_interval_steps', type=int, default=2000,
                       help='Run validation every N training steps')
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=2000,
                       help='Checkpoint frequency in training steps')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Number of best checkpoints to keep (-1 keeps all)')
    parser.add_argument('--log_images_every_n_steps', type=int, default=5000,
                       help='Frequency (in steps) to log generated samples (0 to disable)')
    parser.add_argument('--num_images_to_log', type=int, default=4,
                       help='How many ground-truth samples to log alongside generations')
    
    # Debugging
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick debug pass')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    if args.pitch_shift_min > args.pitch_shift_max:
        raise ValueError('pitch_shift_min must be <= pitch_shift_max')
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Parse GPU indices
    gpu_indices = [int(x) for x in args.devices.split(',')]
    print(f"\n{'='*70}")
    print(f"FractalGen MIDI Training")
    print(f"{'='*70}")
    print(f"Model: FractalGen (128→16→4→1)")
    print(f"Train batch size: {args.train_batch_size}")
    print(f"Val batch size: {args.val_batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Val interval: every {args.val_check_interval_steps} steps")
    print(f"Checkpoint interval: every {args.checkpoint_every_n_steps} steps")
    print(f"Grad accumulation: {args.accumulate_grad_batches}")
    print(f"GPUs: {gpu_indices}")
    print(f"Output dir: {args.output_dir}")
    print(f"Generator types: {args.generator_types}")
    print(f"Pitch shift augmentation: [{args.pitch_shift_min}, {args.pitch_shift_max}] semitones")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer configuration
    print("Creating configuration...")
    config = FractalTrainerConfig(
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval_steps=args.val_check_interval_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        log_images_every_n_steps=args.log_images_every_n_steps,
        num_images_to_log=args.num_images_to_log,
        save_top_k=args.save_top_k
    )
    
    config.precision = args.precision
    config.optimizer.lr = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.scheduler.warmup_steps = args.warmup_steps
    config.model.grad_checkpointing = args.grad_checkpoint
    
    print(f"✓ Model: FractalGen (128→16→4→1)")
    print(f"✓ Learning rate: {args.lr}")
    print(f"✓ Warmup steps: {args.warmup_steps}")
    print(f"✓ Grad clip: {args.grad_clip}")
    print(f"✓ Accumulate grad batches: {args.accumulate_grad_batches}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    pin_memory = not args.no_pin_memory
    persistent_workers = not args.disable_persistent_workers and args.num_workers > 0
    prefetch_factor = max(0, args.prefetch_factor if args.num_workers > 0 else 0)

    train_loader_cfg = DataLoaderConfig.training_default(args.train_data)
    train_loader_cfg.dataset.augment_factor = args.augment_factor
    train_loader_cfg.dataset.cache_in_memory = not args.no_cache_in_memory
    train_loader_cfg.dataset.cache_dir = args.cache_dir
    train_loader_cfg.dataset.pitch_shift_range = (args.pitch_shift_min, args.pitch_shift_max)
    train_loader_cfg.num_workers = args.num_workers
    train_loader_cfg.pin_memory = pin_memory
    train_loader_cfg.prefetch_factor = prefetch_factor if args.num_workers > 0 else 0
    train_loader_cfg.persistent_workers = persistent_workers
    train_loader_cfg.sampler.batch_size = args.train_batch_size
    train_loader_cfg.use_bucket_sampler = False

    val_loader_cfg = DataLoaderConfig.validation_default(args.val_data)
    val_loader_cfg.dataset.random_crop = False
    val_loader_cfg.dataset.augment_factor = 1
    val_loader_cfg.dataset.cache_in_memory = not args.no_cache_in_memory
    val_loader_cfg.dataset.cache_dir = args.cache_dir
    val_loader_cfg.dataset.pitch_shift_range = (0, 0)
    val_loader_cfg.num_workers = args.num_workers
    val_loader_cfg.pin_memory = pin_memory
    val_loader_cfg.prefetch_factor = prefetch_factor if args.num_workers > 0 else 0
    val_loader_cfg.persistent_workers = persistent_workers
    val_loader_cfg.sampler.batch_size = args.val_batch_size
    val_loader_cfg.use_bucket_sampler = args.use_bucket_sampler
    if not args.use_bucket_sampler:
        val_loader_cfg.sampler.shuffle = False

    generator_types = tuple(type_str.strip() for type_str in args.generator_types.split(','))
    if len(generator_types) != 4:
        raise ValueError('generator_types must contain exactly 4 comma-separated values (e.g., "mar,mar,mar,mar")')
    for idx, g in enumerate(generator_types):
        if g not in {'mar', 'ar'}:
            raise ValueError(f'generator_types entry {idx} must be "mar" or "ar", got "{g}"')

    config.model.generator_type_list = generator_types
    config.model.scan_order = args.scan_order
    config.model.mask_ratio_loc = args.mask_ratio_loc
    config.model.mask_ratio_scale = args.mask_ratio_scale

    train_loader = create_dataloader(config=train_loader_cfg)
    val_loader = create_dataloader(config=val_loader_cfg)
    
    train_batches = max(1, len(train_loader))
    val_interval_steps = max(1, args.val_check_interval_steps)
    quotient, remainder = divmod(val_interval_steps, train_batches)
    if remainder == 0:
        trainer_val_check_interval = 1.0
        trainer_check_val_every_n_epoch = max(1, quotient)
    else:
        trainer_val_check_interval = remainder / train_batches
        trainer_check_val_every_n_epoch = quotient + 1 if quotient > 0 else 1
    
    print(f"✓ Train batches: {train_batches}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Validation schedule: every {val_interval_steps} steps -> val_check_interval={trainer_val_check_interval:.4f}, check_val_every_n_epoch={trainer_check_val_every_n_epoch}")
    
    # Create model
    print("\nCreating model...")
    model = FractalMIDILightningModule(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params/1e6:.2f}M")
    
    # Callbacks
    checkpoint_steps = max(1, config.checkpoint_every_n_steps)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='step_{step:08d}-val_loss_{val_loss:.4f}',
            monitor=None,
            mode='min',
            save_top_k=-1,
            save_last=False,
            verbose=True,
            every_n_train_steps=checkpoint_steps,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False
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
        fast_dev_run=args.fast_dev_run,
        val_check_interval=trainer_val_check_interval,
        check_val_every_n_epoch=trainer_check_val_every_n_epoch,
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

