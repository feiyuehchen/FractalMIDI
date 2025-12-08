"""
Training utilities for FractalGen MIDI model.
Contains configuration loading, merging, and saving functions.
"""

import argparse
import shutil
from pathlib import Path
import sys
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Any

# Add code folder to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_config import FractalModelConfig

# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class DataConfig:
    """Data loading configuration."""
    train_data: str
    val_data: str
    crop_length: int = 256
    augment_factor: int = 1
    pitch_shift_min: int = -3
    pitch_shift_max: int = 3
    cache_in_memory: bool = True
    cache_dir: Optional[str] = None

@dataclass
class TrainingLoopConfig:
    """Training loop configuration (top-level training params)."""
    max_steps: int = 200000
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    grad_clip: float = 3.0
    accumulate_grad_batches: int = 1
    train_batch_size: int = 8
    val_batch_size: int = 8

@dataclass
class HardwareConfig:
    """Hardware and compute configuration."""
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    precision: str = "32"

@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    output_dir: str = "outputs/fractalgen"
    log_every_n_steps: int = 50
    val_check_interval_steps: int = 2000
    checkpoint_every_n_steps: int = 2000
    save_top_k: int = 3
    log_images_every_n_steps: int = 5000
    num_images_to_log: int = 4

@dataclass
class FullConfig:
    """Master configuration container."""
    data: DataConfig
    training: TrainingLoopConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    model: FractalModelConfig
    # Debugging/Misc
    seed: int = 42
    fast_dev_run: bool = False

    @classmethod
    def from_dict(cls, cfg: dict):
        """Create configuration from dictionary."""
        # Helper to safely create sub-configs
        def create_subconfig(cls_type, section_name):
            data = cfg.get(section_name, {})
            # Strict check: if data is empty and class has non-optional fields, it will raise TypeError
            return cls_type(**data)

        return cls(
            data=create_subconfig(DataConfig, 'data'),
            training=create_subconfig(TrainingLoopConfig, 'training'),
            hardware=create_subconfig(HardwareConfig, 'hardware'),
            logging=create_subconfig(LoggingConfig, 'logging'),
            model=FractalModelConfig.from_dict(cfg.get('model', {})),
            seed=cfg.get('seed', 42),
            fast_dev_run=cfg.get('fast_dev_run', False)
        )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config_dict, args):
    """
    Merge YAML config with command line arguments using strict Dataclasses.
    Returns an argparse.Namespace for compatibility.
    """
    if config_dict is None:
        # If no config file, args must provide everything required
        # But since we want to avoid hardcoding defaults here, we rely on Dataclass defaults
        # However, DataConfig requires train_data/val_data which have no defaults.
        # We'll create a dummy dict and hope args fill the gaps or let it fail.
        config_dict = {}

    # 1. Load into Dataclasses (Applying defaults and strict validation)
    try:
        full_config = FullConfig.from_dict(config_dict)
    except TypeError as e:
        print(f"Error loading configuration: {e}")
        print("Ensure your config file contains all required fields (e.g., data.train_data).")
        raise

    # 2. Override with Command Line Arguments
    # Helper to override if arg is present
    def override(obj, attr, arg_val):
        if arg_val is not None:
            setattr(obj, attr, arg_val)

    # Data
    override(full_config.data, 'train_data', args.train_data)
    override(full_config.data, 'val_data', args.val_data)
    override(full_config.data, 'crop_length', args.crop_length)
    override(full_config.data, 'augment_factor', args.augment_factor)
    override(full_config.data, 'pitch_shift_min', args.pitch_shift_min)
    override(full_config.data, 'pitch_shift_max', args.pitch_shift_max)
    
    # Cache logic: Arg flag is 'no_cache_in_memory' (negation)
    if args.no_cache_in_memory:
        full_config.data.cache_in_memory = False
    override(full_config.data, 'cache_dir', args.cache_dir)

    # Training
    override(full_config.training, 'train_batch_size', args.train_batch_size)
    override(full_config.training, 'val_batch_size', args.val_batch_size)
    override(full_config.training, 'max_steps', args.max_steps)
    override(full_config.training, 'learning_rate', args.lr)
    override(full_config.training, 'weight_decay', args.weight_decay)
    override(full_config.training, 'warmup_steps', args.warmup_steps)
    override(full_config.training, 'grad_clip', args.grad_clip)
    override(full_config.training, 'accumulate_grad_batches', args.accumulate_grad_batches)

    # Hardware
    if args.devices is not None:
        full_config.hardware.devices = [int(x) for x in args.devices.split(',')]
    
    override(full_config.hardware, 'num_workers', args.num_workers)
    override(full_config.hardware, 'prefetch_factor', args.prefetch_factor)
    if args.disable_persistent_workers:
        full_config.hardware.persistent_workers = False
    if args.precision != '32':
        full_config.hardware.precision = args.precision
    if args.no_pin_memory:
        full_config.hardware.pin_memory = False

    # Logging
    if args.output_dir != 'outputs/fractalgen':
        full_config.logging.output_dir = args.output_dir
    
    # Check for non-default arg values before overriding
    if args.log_every_n_steps != 50: full_config.logging.log_every_n_steps = args.log_every_n_steps
    if args.val_check_interval_steps != 2000: full_config.logging.val_check_interval_steps = args.val_check_interval_steps
    if args.checkpoint_every_n_steps != 2000: full_config.logging.checkpoint_every_n_steps = args.checkpoint_every_n_steps
    if args.save_top_k != 3: full_config.logging.save_top_k = args.save_top_k
    if args.log_images_every_n_steps != 5000: full_config.logging.log_images_every_n_steps = args.log_images_every_n_steps
    if args.num_images_to_log != 4: full_config.logging.num_images_to_log = args.num_images_to_log

    # Model (Overrides)
    # Architecture
    if hasattr(args, 'seq_len_list') and args.seq_len_list is not None: full_config.model.architecture.seq_len_list = args.seq_len_list
    if hasattr(args, 'img_size_list') and args.img_size_list is not None: full_config.model.architecture.seq_len_list = args.img_size_list # Backwards compatibility
    if hasattr(args, 'embed_dim_list') and args.embed_dim_list is not None: full_config.model.architecture.embed_dim_list = args.embed_dim_list
    if hasattr(args, 'num_blocks_list') and args.num_blocks_list is not None: full_config.model.architecture.num_blocks_list = args.num_blocks_list
    if hasattr(args, 'num_heads_list') and args.num_heads_list is not None: full_config.model.architecture.num_heads_list = args.num_heads_list
    if hasattr(args, 'max_bar_len') and args.max_bar_len != 64: full_config.model.architecture.max_bar_len = args.max_bar_len
    if hasattr(args, 'compressed_dim') and args.compressed_dim != 32: full_config.model.architecture.compressed_dim = args.compressed_dim
    if hasattr(args, 'compression_act') and args.compression_act != 'relu': full_config.model.architecture.compression_act = args.compression_act
    
    # Generator
    if args.generator_types != 'mar,mar,mar,mar':
        full_config.model.generator.generator_type_list = tuple(args.generator_types.split(','))
    if args.scan_order != 'row_major': full_config.model.generator.scan_order = args.scan_order
    if args.mask_ratio_loc != 1.0: full_config.model.generator.mask_ratio_loc = args.mask_ratio_loc
    if args.mask_ratio_scale != 0.5: full_config.model.generator.mask_ratio_scale = args.mask_ratio_scale

    # Training (Model)
    if args.grad_checkpoint: full_config.model.training.grad_checkpointing = True
    
    # Misc
    if args.seed != 42: full_config.seed = args.seed
    if args.fast_dev_run: full_config.fast_dev_run = True
    if args.use_bucket_sampler: 
        # This is an arg-only flag, not in config usually, but we can pass it through
        pass

    # 3. Convert back to Namespace for compatibility
    merged = argparse.Namespace()
    
    # Flatten Data
    for k, v in asdict(full_config.data).items(): setattr(merged, k, v)
    merged.no_cache_in_memory = not full_config.data.cache_in_memory

    # Flatten Training
    for k, v in asdict(full_config.training).items(): setattr(merged, k, v)
    # Rename learning_rate to lr for compatibility
    merged.lr = full_config.training.learning_rate

    # Flatten Hardware
    for k, v in asdict(full_config.hardware).items(): setattr(merged, k, v)
    merged.devices = ','.join(map(str, full_config.hardware.devices))
    merged.disable_persistent_workers = not full_config.hardware.persistent_workers
    merged.no_pin_memory = not full_config.hardware.pin_memory

    # Flatten Logging
    for k, v in asdict(full_config.logging).items(): setattr(merged, k, v)

    # Flatten Model (for those parts main.py or trainer setup might read directly from args)
    # Most model config is passed as the struct, but some legacy code might look at args
    merged.seq_len_list = full_config.model.architecture.seq_len_list
    merged.img_size_list = full_config.model.architecture.seq_len_list # Backwards compatibility
    merged.embed_dim_list = full_config.model.architecture.embed_dim_list
    merged.num_blocks_list = full_config.model.architecture.num_blocks_list
    merged.num_heads_list = full_config.model.architecture.num_heads_list
    merged.attn_dropout = full_config.model.architecture.attn_dropout
    merged.proj_dropout = full_config.model.architecture.proj_dropout
    merged.max_bar_len = full_config.model.architecture.max_bar_len
    merged.compressed_dim = full_config.model.architecture.compressed_dim
    merged.compression_act = full_config.model.architecture.compression_act
    
    merged.grad_checkpoint = full_config.model.training.grad_checkpointing
    merged.v_weight = full_config.model.training.v_weight
    
    merged.generator_types = ','.join(full_config.model.generator.generator_type_list)
    merged.pitch_generator_type = full_config.model.generator.pitch_generator_type
    merged.scan_order = full_config.model.generator.scan_order
    merged.mask_ratio_loc = full_config.model.generator.mask_ratio_loc
    merged.mask_ratio_scale = full_config.model.generator.mask_ratio_scale
    merged.num_conds = full_config.model.generator.num_conds
    merged.full_mask_prob = full_config.model.generator.full_mask_prob

    # Misc
    merged.seed = full_config.seed
    merged.fast_dev_run = full_config.fast_dev_run
    merged.use_bucket_sampler = args.use_bucket_sampler
    
    return merged


def apply_config_defaults(args):
    """
    Legacy function for args-only usage. 
    Now we just create a default FullConfig and merge args into it.
    """
    # Create default config (will fail if required fields like train_data aren't in args)
    # We'll handle required fields by checking args first
    
    # Construct a minimal config dict from args to satisfy required fields of FullConfig
    minimal_config = {
        'data': {
            'train_data': args.train_data if args.train_data else "data/lists/train.txt",
            'val_data': args.val_data if args.val_data else "data/lists/valid.txt"
        }
    }
    
    return merge_config_with_args(minimal_config, args)


def save_config_to_checkpoint_dir(checkpoint_dir, config_source_path, args, config, generator_types, gpu_indices):
    """
    Save training configuration to checkpoint directory.
    """
    config_save_path = checkpoint_dir / 'config.yaml'
    
    if config_source_path:
        # Copy original config file
        shutil.copy2(config_source_path, config_save_path)
        print(f"✓ Saved config from: {config_source_path}")
    else:
        # Generate config from current settings
        # Since we now have FullConfig logic, we should ideally rely on that, 
        # but 'config' passed here is FractalTrainerConfig, not FullConfig.
        # So we reconstruct the dictionary manually as before, but cleaner.
        
        config_dict = {
            'model': config.model.to_dict(), # FractalModelConfig has to_dict
            'training': {
                'max_steps': config.max_steps,
                'learning_rate': config.optimizer.lr,
                'weight_decay': config.optimizer.weight_decay,
                'warmup_steps': config.scheduler.warmup_steps,
                'grad_clip': config.grad_clip,
                'accumulate_grad_batches': config.accumulate_grad_batches,
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
                'cache_in_memory': not args.no_cache_in_memory,
                'cache_dir': args.cache_dir,
            },
            'hardware': {
                'devices': gpu_indices,
                'num_workers': args.num_workers,
                'precision': args.precision,
                'prefetch_factor': args.prefetch_factor,
                'persistent_workers': not args.disable_persistent_workers,
                'pin_memory': not args.no_pin_memory,
            },
            'logging': {
                'output_dir': str(args.output_dir),
                'log_every_n_steps': config.log_every_n_steps,
                'val_check_interval_steps': config.val_check_interval_steps,
                'checkpoint_every_n_steps': config.checkpoint_every_n_steps,
                'save_top_k': config.save_top_k,
                'log_images_every_n_steps': config.log_images_every_n_steps,
                'num_images_to_log': config.num_images_to_log,
            },
            'seed': args.seed,
            'fast_dev_run': args.fast_dev_run
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
    arch_str = '→'.join(str(x) for x in args.seq_len_list)
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
    """
    from src.training.trainer import FractalTrainerConfig, OptimizerConfig, SchedulerConfig
    
    # Create trainer configuration
    print("Creating configuration...")
    
    # Since args is now a Namespace with all values merged, we can just use them.
    # Note: FractalTrainerConfig expects FractalModelConfig structure in .model
    # But we flattened it into args in merge_config_with_args.
    # Ideally we should pass the FullConfig object instead of Namespace, 
    # but this requires changing the signature and main.py.
    # For now, we reconstruct from Namespace.
    
    # ... Wait, we have the Model config parts in args, but FractalTrainerConfig needs the object.
    # Let's create a fresh one from args values.
    
    config = FractalTrainerConfig(
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval_steps=args.val_check_interval_steps,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        log_images_every_n_steps=args.log_images_every_n_steps,
        num_images_to_log=args.num_images_to_log,
        save_top_k=args.save_top_k,
        optimizer=OptimizerConfig(
            lr=args.lr,
            weight_decay=args.weight_decay
        ),
        scheduler=SchedulerConfig(
            warmup_steps=args.warmup_steps
        )
    )
    
    config.precision = args.precision
    config.model.training.grad_checkpointing = args.grad_checkpoint
    
    # Parse and validate generator types
    generator_types = tuple(type_str.strip() for type_str in args.generator_types.split(','))
    if len(generator_types) != 4:
        raise ValueError('generator_types must contain exactly 4 comma-separated values (e.g., "mar,mar,mar,mar")')
    for idx, g in enumerate(generator_types):
        if g not in {'mar', 'ar'}:
            raise ValueError(f'generator_types entry {idx} must be "mar" or "ar", got "{g}"')
    
    config.model.generator.generator_type_list = generator_types
    config.model.generator.scan_order = args.scan_order
    config.model.generator.mask_ratio_loc = args.mask_ratio_loc
    config.model.generator.mask_ratio_scale = args.mask_ratio_scale
    config.model.generator.full_mask_prob = args.full_mask_prob
    
    # Set pitch generator type from args if available
    if hasattr(args, 'pitch_generator_type'):
        config.model.generator.pitch_generator_type = args.pitch_generator_type
    
    # Set model architecture from config
    config.model.architecture.seq_len_list = args.seq_len_list
    config.model.architecture.embed_dim_list = args.embed_dim_list
    config.model.architecture.num_blocks_list = args.num_blocks_list
    config.model.architecture.num_heads_list = args.num_heads_list
    config.model.architecture.attn_dropout = args.attn_dropout
    config.model.architecture.proj_dropout = args.proj_dropout
    config.model.training.v_weight = args.v_weight
    config.model.generator.num_conds = args.num_conds
    
    # New config fields
    config.model.architecture.max_bar_len = args.max_bar_len
    config.model.architecture.compressed_dim = args.compressed_dim
    config.model.architecture.compression_act = args.compression_act
    
    return config, generator_types


def setup_dataloaders(args):
    """
    Create training and validation dataloaders.
    """
    from src.dataset.dataset import DataLoaderConfig, create_dataloader, MIDIDatasetConfig, BucketSamplerConfig
    
    print("\nCreating dataloaders...")
    pin_memory = not args.no_pin_memory
    persistent_workers = not args.disable_persistent_workers and args.num_workers > 0
    prefetch_factor = max(0, args.prefetch_factor if args.num_workers > 0 else 0)

    # Training dataloader
    train_dataset_cfg = MIDIDatasetConfig(
        file_list_path=args.train_data,
        crop_length=args.crop_length,
        augment_factor=args.augment_factor,
        cache_in_memory=not args.no_cache_in_memory,
        cache_dir=args.cache_dir,
        pitch_shift_range=(args.pitch_shift_min, args.pitch_shift_max),
        max_bar_len=args.max_bar_len
    )
    
    train_loader_cfg = DataLoaderConfig(
        dataset=train_dataset_cfg,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        sampler=BucketSamplerConfig(
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=True
        ),
        use_bucket_sampler=False # Default for training usually false if random crop
    )

    # Validation dataloader
    val_dataset_cfg = MIDIDatasetConfig(
        file_list_path=args.val_data,
        random_crop=False,
        crop_length=args.crop_length,
        augment_factor=1,
        cache_in_memory=not args.no_cache_in_memory,
        cache_dir=args.cache_dir,
        pitch_shift_range=(0, 0),
        max_bar_len=args.max_bar_len
    )
    
    val_loader_cfg = DataLoaderConfig(
        dataset=val_dataset_cfg,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        sampler=BucketSamplerConfig(
            batch_size=args.val_batch_size,
            shuffle=False,
            drop_last=False
        ),
        use_bucket_sampler=args.use_bucket_sampler
    )
        
    if not args.use_bucket_sampler:
        val_loader_cfg.sampler.shuffle = False

    train_loader = create_dataloader(config=train_loader_cfg)
    val_loader = create_dataloader(config=val_loader_cfg)
    
    return train_loader, val_loader


def compute_validation_schedule(train_batches, val_interval_steps, accumulate_grad_batches=1):
    """
    Compute PyTorch Lightning validation schedule parameters.
    val_interval_steps: Number of global optimization steps between validations.
    """
    train_batches = max(1, train_batches)
    val_interval_steps = max(1, val_interval_steps)
    accumulate_grad_batches = max(1, accumulate_grad_batches)
    
    # Convert global steps to batches
    val_interval_batches = val_interval_steps * accumulate_grad_batches
    
    quotient, remainder = divmod(val_interval_batches, train_batches)
    if remainder == 0:
        trainer_val_check_interval = 1.0
        trainer_check_val_every_n_epoch = max(1, quotient)
    else:
        trainer_val_check_interval = remainder / train_batches
        trainer_check_val_every_n_epoch = quotient + 1 if quotient > 0 else 1
    
    return trainer_val_check_interval, trainer_check_val_every_n_epoch
