"""
PyTorch Lightning trainer for FractalGen Piano Roll model.
Simplified version that uses the complete hierarchical FractalGen architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import new model
from models.temporal_fractal import TemporalFractalNetwork
from visualizer import log_piano_roll_to_tensorboard


# ==============================================================================
# Configuration Dataclasses
# ==============================================================================

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.05


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    schedule_type: str = "cosine"  # cosine, linear, or constant
    warmup_steps: int = 2000
    min_lr: float = 1e-6


@dataclass
class ModelConfig:
    """Configuration for the FractalGen piano model."""
    # Model architecture - Temporal Fractal Network
    embed_dims: Tuple[int, ...] = field(default_factory=lambda: (512, 256, 128))
    num_heads: Tuple[int, ...] = field(default_factory=lambda: (8, 4, 2))
    num_blocks: Tuple[int, ...] = field(default_factory=lambda: (6, 4, 2))
    input_channels: int = 3
    cond_dim: int = 0 # Condition dimension (e.g. for genre embedding)
    
    # Legacy parameters (kept for compatibility if needed, but unused by TFN)
    img_size_list: Tuple[int, ...] = field(default_factory=lambda: (128, 16, 4, 1))
    
    # Training configuration
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    
    def __post_init__(self):
        # Validate architecture lists
        if len(self.embed_dims) != 3:
            raise ValueError("embed_dims must contain exactly 3 entries")
        if len(self.num_heads) != 3:
            raise ValueError("num_heads must contain exactly 3 entries")
        if len(self.num_blocks) != 3:
            raise ValueError("num_blocks must contain exactly 3 entries")


@dataclass
class FractalTrainerConfig:
    """Main trainer configuration."""
    # Training hyperparameters
    max_steps: int = 200000
    grad_clip: float = 3.0
    accumulate_grad_batches: int = 1
    
    # Trainer runtime options
    precision: str = "32"
    strategy: Optional[str] = None
    log_every_n_steps: int = 50
    val_check_interval_steps: int = 2000
    checkpoint_every_n_steps: int = 2000
    save_top_k: int = 3
    
    # Visualization
    log_images_every_n_steps: int = 5000
    num_images_to_log: int = 4
    
    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.max_steps > 0, "max_steps must be positive"
        assert self.grad_clip >= 0, "grad_clip must be non-negative"
        assert self.accumulate_grad_batches >= 1, "accumulate_grad_batches must be >= 1"
        assert self.precision in {"16", "32", "bf16"}, "precision must be one of {'16', '32', 'bf16'}"
        if self.log_images_every_n_steps is not None:
            assert self.log_images_every_n_steps >= 0, "log_images_every_n_steps must be >= 0"


# ==============================================================================
# Lightning Module
# ==============================================================================

class FractalMIDILightningModule(pl.LightningModule):
    """PyTorch Lightning module for FractalGen piano roll generation."""
    
    def __init__(self, config: FractalTrainerConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Build model
        self.model = self._build_model()
        self._last_logged_image_step = -1
        
    def _build_model(self):
        """Build FractalGen model with configurable architecture."""
        model_cfg = self.config.model
        
        model = TemporalFractalNetwork(
            input_channels=model_cfg.input_channels,
            embed_dims=model_cfg.embed_dims,
            num_heads=model_cfg.num_heads,
            num_blocks=model_cfg.num_blocks,
            cond_dim=model_cfg.cond_dim
        )
        return model
    
    def forward(self, notes, tempo, density, global_cond=None):
        """
        Forward pass through FractalGen.
        
        Args:
            notes: (B, 2, T, 128) Note/Vel tensors
            tempo: (B, T) Tempo curve
            density: (B, T) Density curve
            global_cond: (B, C) condition vector
        
        Returns:
            loss: Hierarchical loss
            stats: Dictionary of statistics
        """
        loss, stats = self.model(notes, tempo, density, global_cond)
        return loss, stats
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Batch: (notes, tempo, density, durations, shifts)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 5:
                notes, tempo, density, durations, pitch_shifts = batch
            elif len(batch) == 4:
                notes, tempo, density, durations = batch
                pitch_shifts = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            raise ValueError("Batch must be a tuple")
        
        # Label Dropout (CFG)
        global_cond = None
        
        # Forward pass
        loss, stats = self(notes, tempo, density, global_cond)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self._log_metrics(stats, split='train', on_step=True, on_epoch=False)

        if pitch_shifts is not None:
            pitch_shifts = pitch_shifts.to(notes.device)
            self.log('train/pitch_shift_mean', pitch_shifts.float().mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False)
        
        try:
            if hasattr(self, '_trainer') and self._trainer is not None:
                self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        except:
            pass 
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 5:
                notes, tempo, density, durations, pitch_shifts = batch
            elif len(batch) == 4:
                notes, tempo, density, durations = batch
                pitch_shifts = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        # Forward pass
        loss, stats = self(notes, tempo, density)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._log_metrics(stats, split='val', on_step=False, on_epoch=True)

        # Log images
        if batch_idx == 0:
            should_log_images = (
                self.config.log_images_every_n_steps is not None
                and self.config.log_images_every_n_steps > 0
                and self.global_step > 0
                and self.global_step % self.config.log_images_every_n_steps == 0
            )
            
            if should_log_images:
                self._log_images(batch, batch_idx, 'val', force_sample=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 5:
                notes, tempo, density, durations, pitch_shifts = batch
            elif len(batch) == 4:
                notes, tempo, density, durations = batch
                pitch_shifts = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        loss, stats = self(notes, tempo, density)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self._log_metrics(stats, split='test', on_step=False, on_epoch=True)

        return loss
    
    def _reconstruct_piano_roll(self, notes, tempo):
        # notes: (B, 2, T, 128)
        # tempo: (B, T)
        # Returns: (B, 3, T, 128)
        B, _, T, P = notes.shape
        tempo_expanded = tempo.unsqueeze(1).unsqueeze(-1).expand(B, 1, T, P) # (B, 1, T, 128)
        return torch.cat([notes, tempo_expanded], dim=1)

    def _log_images(self, batch, batch_idx, split='train', force_sample=False):
        """Log piano rolls and generated samples during validation."""
        if self.trainer is None or self.logger is None:
            return
        if self.trainer.sanity_checking:
            return
        
        # Only log during validation
        if split != 'val':
            return
        
        # Unpack batch
        if len(batch) >= 4:
            notes, tempo, density, durations = batch[0], batch[1], batch[2], batch[3]
        
        num_to_log = min(self.config.num_images_to_log, notes.size(0))
        if num_to_log <= 0:
            return
        
        # Check if we should log this step
        if not force_sample:
            return
        if self.config.log_images_every_n_steps is None or self.config.log_images_every_n_steps <= 0:
            return
        
        # Avoid logging images multiple times at the same step
        if self._last_logged_image_step == self.global_step:
            return
        
        self._last_logged_image_step = self.global_step
        
        # Reconstruct GT piano roll for logging
        gt_rolls = self._reconstruct_piano_roll(notes, tempo) # (B, 3, T, 128)
        
        # Log ground truth
        if not hasattr(self, '_ground_truth_logged') or not self._ground_truth_logged:
            for i in range(num_to_log):
                roll = gt_rolls[i]
                try:
                    log_piano_roll_to_tensorboard(
                        self.logger.experiment,
                        f'{split}_images/0_ground_truth/sample_{i}', 
                        roll.detach().cpu(),
                        self.global_step,
                        apply_colormap=True
                    )
                except Exception as exc:
                    print(f"Warning: Failed to log ground truth {i}: {exc}")
            self._ground_truth_logged = True

        # Generate samples
        samples_to_generate = num_to_log
        target_length = notes.shape[2]
        
        try:
            with torch.no_grad():
                generated, intermediates = self.model.sample(
                    batch_size=samples_to_generate,
                    length=target_length,
                    global_cond=None,
                    num_iter_list=[8, 4, 2], 
                    cfg=1.0,
                    temperature=1.0,
                    return_intermediates=True
                )
        except Exception as exc:
            print(f"Warning: Failed to generate samples for logging: {exc}")
            import traceback
            traceback.print_exc()
            return

        if not torch.is_tensor(generated):
            return

        generated = generated.detach().cpu()
        # generated is (B, 2, T, 128) [Note, Vel]
        # We need to add Tempo channel for visualization
        # Note: model.sample returns Content (Note, Vel). Tempo is implicitly in Structure level.
        # But `intermediates` has structure output.
        # Or we can extract tempo from intermediates[0]?
        # Actually, `generated` is just content.
        # Let's use a dummy tempo for visualization if we don't have it easily, 
        # OR extract it from intermediates.
        
        # Extract tempo from Level 0 output
        # intermediate structure: {'level': 0, 'output': (B, 2, T_low), 'is_structure': True}
        # We need final Level 0 output.
        # Let's iterate intermediates to find last step of Level 0
        tempo_curve = None
        for item in reversed(intermediates):
            if item.get('is_structure', False):
                # This is structure (B, 2, T_low)
                # We need to upsample it to T
                struct = item['output'] # (B, 2, T_low)
                tempo_low = struct[:, 1] # (B, T_low)
                # Upsample
                tempo_curve = F.interpolate(tempo_low.unsqueeze(1), size=target_length, mode='linear', align_corners=False).squeeze(1)
                break
        
        if tempo_curve is None:
            tempo_curve = torch.ones(samples_to_generate, target_length) * 0.5

        # Reconstruct generated roll
        gen_rolls = self._reconstruct_piano_roll(generated, tempo_curve)

        for i in range(min(samples_to_generate, gen_rolls.size(0))):
            roll = gen_rolls[i]
            try:
                log_piano_roll_to_tensorboard(
                    self.logger.experiment,
                    f'{split}_images/1_generated_step_{self.global_step:07d}/sample_{i}', 
                    roll,
                    self.global_step,
                    apply_colormap=True
                )
            except Exception as exc:
                print(f"Warning: Failed to log generated sample {i}: {exc}")
        
        if intermediates is not None and len(intermediates) > 0:
            self._create_generation_gifs(intermediates, samples_to_generate, split)
            
    def _create_generation_gifs(self, intermediates, num_samples, split='val'):
        """Create GIF animations from generation intermediates."""
        try:
            from visualizer import create_growth_animation
            import os
            
            gif_dir = os.path.join(self.logger.log_dir, 'generation_gifs')
            os.makedirs(gif_dir, exist_ok=True)
            
            # We need to standardise frames to (3, T, 128)
            # Structure frames: (B, 2, T_low) -> Need to map to visualization
            # Content frames: (B, 2, T, 128) -> Add tempo
            
            sample_frames = []
            
            # Use final tempo for all content frames
            # For structure frames, maybe visualize density/tempo as heatmap?
            # Or just skip structure frames in GIF for now to keep it simple?
            # Critique asked for "Level 0 -> Chord". It would be cool to see it.
            # But visualization expects piano roll.
            # Let's just visualize Content frames (Level 1+).
            
            # Find final tempo for sample 0
            final_tempo = torch.ones(128) * 0.5 # Fallback
            # ... (logic to find tempo, same as above) ...
            
            for item in intermediates:
                if not item.get('is_structure', False):
                    # Content frame (B, 2, T, 128)
                    frame = item['output'][0] # (2, T, 128)
                    # Add dummy tempo channel
                    T = frame.shape[1]
                    tempo_ch = torch.ones(1, T, 128) * 0.5
                    full_frame = torch.cat([frame, tempo_ch], dim=0) # (3, T, 128)
                    sample_frames.append(full_frame)

            if not sample_frames:
                return

            # Generate GIF for sample 0
            gif_path = os.path.join(gif_dir, f'step_{self.global_step:07d}_sample_0.gif')
            
            create_growth_animation(
                sample_frames,
                save_path=gif_path,
                fps=24,
                transition_duration=0.15,
                min_height=512, 
                easing="ease_in_out_cubic",
                pop_effect=True,
                optimize=True,
                quality=90
            )
            
            if frames is None: # create_growth_animation returns None on success (saves file)
                 print(f"Saved generation GIF: {gif_path}")
                
        except Exception as exc:
            print(f"Warning: Failed to create generation GIFs: {exc}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_cfg = self.config.optimizer
        sched_cfg = self.config.scheduler
        
        # Create optimizer with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            weight_decay=opt_cfg.weight_decay
        )
        
        # Create learning rate scheduler
        total_steps = max(1, self.config.max_steps)
        warmup_steps = max(0, sched_cfg.warmup_steps)

        def lr_lambda(current_step: int):
            """Learning rate schedule function based on global step."""
            step = max(current_step, 0)

            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            
            if sched_cfg.schedule_type == "constant":
                return 1.0
            if sched_cfg.schedule_type == "linear":
                return 1.0 - progress * (1.0 - sched_cfg.min_lr / opt_cfg.lr)
            if sched_cfg.schedule_type == "cosine":
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return sched_cfg.min_lr / opt_cfg.lr + (1.0 - sched_cfg.min_lr / opt_cfg.lr) * cosine_decay
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _log_metrics(self, stats: Dict[str, torch.Tensor], split: str, on_step: bool, on_epoch: bool):
        """Helper to log hierarchical statistics returned by the model."""
        if stats is None:
            return

        for key, value in stats.items():
            metric_name = f"{split}/{key}"
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(float(value), device=self.device)
            elif torch.is_tensor(value):
                tensor_value = value.detach()
                if tensor_value.numel() != 1:
                    tensor_value = tensor_value.mean()
                tensor_value = tensor_value.to(self.device)
            else:
                continue

            tensor_value = tensor_value.float()
            sync_dist = split != 'train'
            self.log(metric_name, tensor_value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=sync_dist)
    
    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping before optimizer step."""
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_trainer(config: FractalTrainerConfig, *, val_check_interval: float | None = None,
                  check_val_every_n_epoch: int | None = None, **trainer_kwargs) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer.
    
    Args:
        config: Trainer configuration
        **trainer_kwargs: Additional arguments for pl.Trainer (overrides config values)
    
    Returns:
        PyTorch Lightning Trainer
    """
    # Default trainer arguments
    strategy = config.strategy if config.strategy is not None else ('ddp' if torch.cuda.device_count() > 1 else 'auto')

    default_args = {
        'max_steps': config.max_steps,
        'max_epochs': 1_000_000,  # Effectively rely on max_steps for termination
        'accelerator': 'auto',
        'devices': 'auto',
        'strategy': strategy,
        'precision': config.precision,
        'gradient_clip_val': 0,  # Handled in on_before_optimizer_step
        'accumulate_grad_batches': config.accumulate_grad_batches,
        'log_every_n_steps': config.log_every_n_steps,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': False,
        'benchmark': True,
        'num_sanity_val_steps': 0,
    }

    if val_check_interval is None:
        # Convert global steps to batches for Lightning
        # val_check_interval in int means batches
        # global_step = batch_idx // accumulate_grad_batches
        # So we need val_check_interval_steps * accumulate_grad_batches batches
        default_args['val_check_interval'] = int(config.val_check_interval_steps * config.accumulate_grad_batches)
    else:
        default_args['val_check_interval'] = val_check_interval

    default_args['check_val_every_n_epoch'] = check_val_every_n_epoch if check_val_every_n_epoch is not None else 1
    
    # Merge with user-provided arguments
    trainer_args = {**default_args, **trainer_kwargs}
    
    # Configure ModelCheckpoint callback if not present
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_top_k=config.save_top_k,
        monitor='val_loss',
        mode='min',
        filename='step_{step:08d}-val_loss_{val_loss:.4f}',
        save_last=True,
        save_on_train_epoch_end=False  # Crucial for saving based on steps not epochs
    )
    
    callbacks = trainer_args.get('callbacks', [])
    if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        callbacks.append(checkpoint_callback)
    trainer_args['callbacks'] = callbacks
    
    # Disable distributed sampler (we use custom BucketBatchSampler)
    trainer_args['use_distributed_sampler'] = False
    
    # Create trainer
    trainer = pl.Trainer(**trainer_args)
    
    return trainer
