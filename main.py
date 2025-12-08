"""
Main training script for the FractalGen MIDI model.

Usage:
    # Using config file
    python main.py --config config/train_default.yaml
    
    # Using command line arguments (legacy)
    python main.py --train_batch_size 8 --max_steps 200000 --devices 0,1
    
    # Override config with command line arguments
    python main.py --config config/train_default.yaml --max_steps 100000
"""

import argparse
from pathlib import Path
import sys
import torch

# Add code to path
sys.path.append(str(Path(__file__).parent))

# Optimize for RTX 4090/Ampere+
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.training.trainer import FractalMIDILightningModule, create_trainer, DualProgressBar
from src.training.train_utils import (
    load_config,
    merge_config_with_args,
    apply_config_defaults,
    save_config_to_checkpoint_dir,
    print_training_summary,
    print_model_info,
    setup_trainer_config,
    setup_dataloaders,
    compute_validation_schedule
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FractalGen MIDI model')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (e.g., config/train_default.yaml)')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default=None,
                       help='Path to training data list')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation data list')
    
    # Training configuration
    parser.add_argument('--train_batch_size', type=int, default=None,
                       help='Training batch size per device')
    parser.add_argument('--val_batch_size', type=int, default=None,
                       help='Validation batch size per device')
    parser.add_argument('--crop_length', type=int, default=None,
                       help='Crop length for training data (128 for 128x128, 256 for 128x256, 512 for 128x512)')
    parser.add_argument('--augment_factor', type=int, default=None,
                       help='Random crop augmentation factor for training data')
    parser.add_argument('--pitch_shift_min', type=int, default=None,
                       help='Minimum semitone shift for pitch augmentation (inclusive)')
    parser.add_argument('--pitch_shift_max', type=int, default=None,
                       help='Maximum semitone shift for pitch augmentation (inclusive)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Maximum number of optimizer update steps')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=None,
                       help='Number of learning-rate warmup steps')
    parser.add_argument('--grad_clip', type=float, default=None,
                       help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=None,
                       help='Gradient accumulation steps')
    
    # Hardware configuration
    parser.add_argument('--devices', type=str, default=None,
                       help='Comma-separated GPU indices (e.g., "0,1")')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers')
    parser.add_argument('--prefetch_factor', type=int, default=None,
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
    parser.add_argument('--full_mask_prob', type=float, default=0.1,
                       help='Probability of forcing 100%% mask during training')
    
    # New Model Architecture Configs
    parser.add_argument('--max_bar_len', type=int, default=64,
                       help='Maximum bar length for position embedding')
    parser.add_argument('--compressed_dim', type=int, default=32,
                       help='Dimension of harmonic compression bottleneck')
    parser.add_argument('--compression_act', type=str, default='relu',
                       choices=['relu', 'gelu', 'identity'],
                       help='Activation function for harmonic compression')
    
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
    # Parse and merge configuration
    cmd_args = parse_args()
    config_yaml = None
    if cmd_args.config:
        config_path = Path(cmd_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        print(f"Loading config from: {config_path}")
        config_yaml = load_config(config_path)
        args = merge_config_with_args(config_yaml, cmd_args)
    else:
        args = apply_config_defaults(cmd_args)
    
    # Validate arguments
    if args.pitch_shift_min > args.pitch_shift_max:
        raise ValueError('pitch_shift_min must be <= pitch_shift_max')
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Parse GPU indices
    gpu_indices = [int(x) for x in args.devices.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup trainer configuration
    config, generator_types = setup_trainer_config(args)
    
    # Create dataloaders
    train_loader, val_loader = setup_dataloaders(args)
    
    # Compute validation schedule
    train_batches = len(train_loader)
    val_batches = len(val_loader)
    val_interval_steps = args.val_check_interval_steps
    trainer_val_check_interval, trainer_check_val_every_n_epoch = compute_validation_schedule(
        train_batches, val_interval_steps, args.accumulate_grad_batches
    )
    
    # Print training summary
    print_training_summary(
        args, gpu_indices, train_batches, val_batches, val_interval_steps,
        trainer_val_check_interval, trainer_check_val_every_n_epoch
    )
    
    # Create model
    print("\nCreating model...")
    model = FractalMIDILightningModule(config)
    print_model_info(model)
    
    # Logger (create first to get version number)
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='logs'
    )
    
    # Use the same version as logger for checkpoints
    version = logger.version
    checkpoint_dir = output_dir / 'checkpoints' / f'version_{version}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Experiment version: {version}")
    print(f"✓ Checkpoint dir: {checkpoint_dir}")
    
    # Save config to checkpoint directory
    config_save_path = save_config_to_checkpoint_dir(
        checkpoint_dir, cmd_args.config, args, config, generator_types, gpu_indices
    )
    
    # Callbacks
    checkpoint_steps = max(1, config.checkpoint_every_n_steps)
    callbacks = [
        DualProgressBar(),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
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
        print(f"Experiment version: {version}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print(f"Config saved to: {config_save_path}")
        print(f"TensorBoard logs: {output_dir / 'logs' / f'version_{version}'}")
        print(f"\nView logs with: tensorboard --logdir {output_dir / 'logs'}")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
