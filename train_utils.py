"""
Training utilities for FractalGen MIDI model.
Contains configuration loading, merging, and saving functions.
"""

import argparse
import shutil
from pathlib import Path
import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config, args):
    """
    Merge YAML config with command line arguments.
    
    Command line arguments take precedence over config file values.
    Only non-None command line arguments will override config values.
    
    Args:
        config: Dictionary loaded from YAML config file (or None)
        args: Namespace from argparse
        
    Returns:
        merged: Namespace with merged configuration
    """
    if config is None:
        return args
    
    # Create a namespace with config defaults
    merged = argparse.Namespace()
    
    # Data paths
    merged.train_data = args.train_data if args.train_data is not None else config.get('data', {}).get('train_data', 'dataset/train.txt')
    merged.val_data = args.val_data if args.val_data is not None else config.get('data', {}).get('val_data', 'dataset/valid.txt')
    
    # Training configuration
    merged.train_batch_size = args.train_batch_size if args.train_batch_size is not None else config.get('training', {}).get('train_batch_size', 8)
    merged.val_batch_size = args.val_batch_size if args.val_batch_size is not None else config.get('training', {}).get('val_batch_size', 8)
    merged.crop_length = args.crop_length if args.crop_length is not None else config.get('data', {}).get('crop_length', 256)
    merged.augment_factor = args.augment_factor if args.augment_factor is not None else config.get('data', {}).get('augment_factor', 1)
    merged.pitch_shift_min = args.pitch_shift_min if args.pitch_shift_min is not None else config.get('data', {}).get('pitch_shift_min', -3)
    merged.pitch_shift_max = args.pitch_shift_max if args.pitch_shift_max is not None else config.get('data', {}).get('pitch_shift_max', 3)
    merged.max_steps = args.max_steps if args.max_steps is not None else config.get('training', {}).get('max_steps', 200000)
    merged.lr = args.lr if args.lr is not None else config.get('training', {}).get('learning_rate', 1e-4)
    merged.weight_decay = args.weight_decay if args.weight_decay is not None else config.get('training', {}).get('weight_decay', 0.05)
    merged.warmup_steps = args.warmup_steps if args.warmup_steps is not None else config.get('training', {}).get('warmup_steps', 2000)
    merged.grad_clip = args.grad_clip if args.grad_clip is not None else config.get('training', {}).get('grad_clip', 3.0)
    merged.accumulate_grad_batches = args.accumulate_grad_batches if args.accumulate_grad_batches is not None else config.get('training', {}).get('accumulate_grad_batches', 1)
    
    # Hardware configuration
    merged.devices = args.devices if args.devices is not None else ','.join(map(str, config.get('hardware', {}).get('devices', [0, 1])))
    merged.num_workers = args.num_workers if args.num_workers is not None else config.get('hardware', {}).get('num_workers', 4)
    merged.prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else config.get('hardware', {}).get('prefetch_factor', 2)
    merged.disable_persistent_workers = args.disable_persistent_workers if args.disable_persistent_workers else not config.get('hardware', {}).get('persistent_workers', True)
    merged.precision = args.precision if args.precision != '32' else config.get('hardware', {}).get('precision', '32')
    merged.grad_checkpoint = args.grad_checkpoint if args.grad_checkpoint else config.get('model', {}).get('grad_checkpointing', False)
    merged.no_cache_in_memory = args.no_cache_in_memory if args.no_cache_in_memory else not config.get('data', {}).get('cache_in_memory', True)
    merged.cache_dir = args.cache_dir if args.cache_dir is not None else config.get('data', {}).get('cache_dir', None)
    merged.no_pin_memory = args.no_pin_memory if args.no_pin_memory else not config.get('hardware', {}).get('pin_memory', True)
    merged.use_bucket_sampler = args.use_bucket_sampler
    
    # Model configuration
    model_cfg = config.get('model', {})
    gen_types = model_cfg.get('generator_types', ['mar', 'mar', 'mar', 'mar'])
    merged.generator_types = args.generator_types if args.generator_types != 'mar,mar,mar,mar' else ','.join(gen_types)
    merged.scan_order = args.scan_order if args.scan_order != 'row_major' else model_cfg.get('scan_order', 'row_major')
    merged.mask_ratio_loc = args.mask_ratio_loc if args.mask_ratio_loc != 1.0 else model_cfg.get('mask_ratio_loc', 1.0)
    merged.mask_ratio_scale = args.mask_ratio_scale if args.mask_ratio_scale != 0.5 else model_cfg.get('mask_ratio_scale', 0.5)
    
    # Model architecture (read from config only, not overridable via command line)
    merged.img_size_list = tuple(model_cfg.get('img_size_list', [128, 16, 4, 1]))
    merged.embed_dim_list = tuple(model_cfg.get('embed_dim_list', [512, 256, 128, 64]))
    merged.num_blocks_list = tuple(model_cfg.get('num_blocks_list', [12, 3, 2, 1]))
    merged.num_heads_list = tuple(model_cfg.get('num_heads_list', [8, 4, 2, 2]))
    merged.attn_dropout = model_cfg.get('attn_dropout', 0.0)
    merged.proj_dropout = model_cfg.get('proj_dropout', 0.0)
    merged.v_weight = model_cfg.get('v_weight', 1.0)
    merged.num_conds = model_cfg.get('num_conds', 5)
    
    # Logging and checkpointing
    merged.output_dir = args.output_dir if args.output_dir != 'outputs/fractalgen' else config.get('logging', {}).get('output_dir', 'outputs/fractalgen')
    merged.log_every_n_steps = args.log_every_n_steps if args.log_every_n_steps != 50 else config.get('logging', {}).get('log_every_n_steps', 50)
    merged.val_check_interval_steps = args.val_check_interval_steps if args.val_check_interval_steps != 2000 else config.get('logging', {}).get('val_check_interval_steps', 2000)
    merged.checkpoint_every_n_steps = args.checkpoint_every_n_steps if args.checkpoint_every_n_steps != 2000 else config.get('logging', {}).get('checkpoint_every_n_steps', 2000)
    merged.save_top_k = args.save_top_k if args.save_top_k != 3 else config.get('logging', {}).get('save_top_k', 3)
    merged.log_images_every_n_steps = args.log_images_every_n_steps if args.log_images_every_n_steps != 5000 else config.get('logging', {}).get('log_images_every_n_steps', 5000)
    merged.num_images_to_log = args.num_images_to_log if args.num_images_to_log != 4 else config.get('logging', {}).get('num_images_to_log', 4)
    
    # Debugging
    merged.fast_dev_run = args.fast_dev_run if args.fast_dev_run else config.get('fast_dev_run', False)
    merged.seed = args.seed if args.seed != 42 else config.get('seed', 42)
    
    return merged


def apply_config_defaults(args):
    """
    Apply default values to any None fields in args.
    
    Args:
        args: Namespace with potentially None values
        
    Returns:
        args: Namespace with defaults applied
    """
    if args.train_data is None:
        args.train_data = 'dataset/train.txt'
    if args.val_data is None:
        args.val_data = 'dataset/valid.txt'
    if args.train_batch_size is None:
        args.train_batch_size = 8
    if args.val_batch_size is None:
        args.val_batch_size = 8
    if args.crop_length is None:
        args.crop_length = 256
    if args.augment_factor is None:
        args.augment_factor = 1
    if args.pitch_shift_min is None:
        args.pitch_shift_min = -3
    if args.pitch_shift_max is None:
        args.pitch_shift_max = 3
    if args.max_steps is None:
        args.max_steps = 200000
    if args.lr is None:
        args.lr = 1e-4
    if args.weight_decay is None:
        args.weight_decay = 0.05
    if args.warmup_steps is None:
        args.warmup_steps = 2000
    if args.grad_clip is None:
        args.grad_clip = 3.0
    if args.accumulate_grad_batches is None:
        args.accumulate_grad_batches = 1
    if args.devices is None:
        args.devices = '0,1'
    if args.num_workers is None:
        args.num_workers = 4
    if args.prefetch_factor is None:
        args.prefetch_factor = 2
    
    # Model architecture defaults
    if not hasattr(args, 'img_size_list'):
        args.img_size_list = (128, 16, 4, 1)
    if not hasattr(args, 'embed_dim_list'):
        args.embed_dim_list = (512, 256, 128, 64)
    if not hasattr(args, 'num_blocks_list'):
        args.num_blocks_list = (12, 3, 2, 1)
    if not hasattr(args, 'num_heads_list'):
        args.num_heads_list = (8, 4, 2, 2)
    if not hasattr(args, 'attn_dropout'):
        args.attn_dropout = 0.0
    if not hasattr(args, 'proj_dropout'):
        args.proj_dropout = 0.0
    if not hasattr(args, 'v_weight'):
        args.v_weight = 1.0
    if not hasattr(args, 'num_conds'):
        args.num_conds = 5

    return args


def save_config_to_checkpoint_dir(checkpoint_dir, config_source_path, args, config, generator_types, gpu_indices):
    """
    Save training configuration to checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        config_source_path: Path to original config file (if used), or None
        args: Merged arguments namespace
        config: Trainer config object
        generator_types: Tuple of generator types
        gpu_indices: List of GPU indices
        
    Returns:
        config_save_path: Path where config was saved
    """
    config_save_path = checkpoint_dir / 'config.yaml'
    
    if config_source_path:
        # Copy original config file
        shutil.copy2(config_source_path, config_save_path)
        print(f"✓ Saved config from: {config_source_path}")
    else:
        # Generate config from current settings
        config_dict = {
            'model': {
                'img_size_list': list(args.img_size_list),
                'embed_dim_list': list(args.embed_dim_list),
                'num_blocks_list': list(args.num_blocks_list),
                'num_heads_list': list(args.num_heads_list),
                'generator_types': list(generator_types),
                'scan_order': args.scan_order,
                'mask_ratio_loc': args.mask_ratio_loc,
                'mask_ratio_scale': args.mask_ratio_scale,
                'attn_dropout': config.model.attn_dropout,
                'proj_dropout': config.model.proj_dropout,
                'grad_checkpointing': args.grad_checkpoint,
            },
            'training': {
                'max_steps': args.max_steps,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'warmup_steps': args.warmup_steps,
                'grad_clip': args.grad_clip,
                'accumulate_grad_batches': args.accumulate_grad_batches,
                'train_batch_size': args.train_batch_size,
                'val_batch_size': args.val_batch_size,
            },
            'data': {
                'train_data': args.train_data,
                'val_data': args.val_data,
                'crop_length': args.crop_length,
                'augment_factor': args.augment_factor,
                'pitch_shift_min': args.pitch_shift_min,
                'pitch_shift_max': args.pitch_shift_max,
            },
            'hardware': {
                'devices': gpu_indices,
                'num_workers': args.num_workers,
                'precision': args.precision,
            },
            'logging': {
                'output_dir': str(args.output_dir),
                'log_every_n_steps': args.log_every_n_steps,
                'val_check_interval_steps': args.val_check_interval_steps,
                'checkpoint_every_n_steps': args.checkpoint_every_n_steps,
                'save_top_k': args.save_top_k,
            },
            'seed': args.seed,
        }
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Generated config from command line args")
    
    print(f"✓ Config saved to: {config_save_path}")
    return config_save_path


def print_training_summary(args, gpu_indices, train_batches, val_batches, val_interval_steps, 
                          trainer_val_check_interval, trainer_check_val_every_n_epoch):
    """Print training configuration summary."""
    print(f"\n{'='*70}")
    print(f"FractalGen MIDI Training")
    print(f"{'='*70}")
    arch_str = '→'.join(str(x) for x in args.img_size_list)
    print(f"Model: FractalGen ({arch_str})")
    print(f"Train batch size: {args.train_batch_size}")
    print(f"Val batch size: {args.val_batch_size}")
    print(f"Crop length: {args.crop_length} (training on 128×{args.crop_length} piano rolls)")
    print(f"Max steps: {args.max_steps}")
    print(f"Val interval: every {args.val_check_interval_steps} steps")
    print(f"Checkpoint interval: every {args.checkpoint_every_n_steps} steps")
    print(f"Grad accumulation: {args.accumulate_grad_batches}")
    print(f"GPUs: {gpu_indices}")
    print(f"Output dir: {args.output_dir}")
    print(f"Generator types: {args.generator_types}")
    print(f"Pitch shift augmentation: [{args.pitch_shift_min}, {args.pitch_shift_max}] semitones")
    print(f"{'='*70}\n")
    
    print("Creating configuration...")
    print(f"✓ Embedding dims: {args.embed_dim_list}")
    print(f"✓ Transformer blocks: {args.num_blocks_list}")
    print(f"✓ Attention heads: {args.num_heads_list}")
    print(f"✓ Learning rate: {args.lr}")
    print(f"✓ Warmup steps: {args.warmup_steps}")
    print(f"✓ Grad clip: {args.grad_clip}")
    print(f"✓ Accumulate grad batches: {args.accumulate_grad_batches}")
    
    print(f"\n✓ Train batches: {train_batches}")
    print(f"✓ Val batches: {val_batches}")
    print(f"✓ Validation schedule: every {val_interval_steps} steps -> val_check_interval={trainer_val_check_interval:.4f}, check_val_every_n_epoch={trainer_check_val_every_n_epoch}")


def print_model_info(model):
    """Print model information."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created")
    print(f"✓ Total parameters: {total_params/1e6:.2f}M")


def setup_trainer_config(args):
    """
    Create and configure FractalTrainerConfig from arguments.
    
    Args:
        args: Merged arguments namespace
        
    Returns:
        config: Configured FractalTrainerConfig
        generator_types: Tuple of generator types
    """
    from trainer import FractalTrainerConfig
    
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
    
    # Parse and validate generator types
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
    
    # Set model architecture from config
    config.model.img_size_list = args.img_size_list
    config.model.embed_dim_list = args.embed_dim_list
    config.model.num_blocks_list = args.num_blocks_list
    config.model.num_heads_list = args.num_heads_list
    config.model.attn_dropout = args.attn_dropout
    config.model.proj_dropout = args.proj_dropout
    config.model.v_weight = args.v_weight
    config.model.num_conds = args.num_conds
    
    return config, generator_types


def setup_dataloaders(args):
    """
    Create training and validation dataloaders.
    
    Args:
        args: Merged arguments namespace
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    from dataset import DataLoaderConfig, create_dataloader
    
    print("\nCreating dataloaders...")
    pin_memory = not args.no_pin_memory
    persistent_workers = not args.disable_persistent_workers and args.num_workers > 0
    prefetch_factor = max(0, args.prefetch_factor if args.num_workers > 0 else 0)

    # Training dataloader
    train_loader_cfg = DataLoaderConfig.training_default(args.train_data)
    train_loader_cfg.dataset.crop_length = args.crop_length
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

    # Validation dataloader
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

    train_loader = create_dataloader(config=train_loader_cfg)
    val_loader = create_dataloader(config=val_loader_cfg)
    
    return train_loader, val_loader


def compute_validation_schedule(train_batches, val_interval_steps):
    """
    Compute PyTorch Lightning validation schedule parameters.
    
    Args:
        train_batches: Number of training batches per epoch
        val_interval_steps: Desired validation interval in steps
        
    Returns:
        trainer_val_check_interval: val_check_interval for Trainer
        trainer_check_val_every_n_epoch: check_val_every_n_epoch for Trainer
    """
    train_batches = max(1, train_batches)
    val_interval_steps = max(1, val_interval_steps)
    quotient, remainder = divmod(val_interval_steps, train_batches)
    if remainder == 0:
        trainer_val_check_interval = 1.0
        trainer_check_val_every_n_epoch = max(1, quotient)
    else:
        trainer_val_check_interval = remainder / train_batches
        trainer_check_val_every_n_epoch = quotient + 1 if quotient > 0 else 1
    
    return trainer_val_check_interval, trainer_check_val_every_n_epoch

