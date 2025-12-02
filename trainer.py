"""
PyTorch Lightning trainer for FractalGen Piano Roll model.
Simplified version that uses the complete hierarchical FractalGen architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import fractalmar_piano
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
    # Model architecture
    img_size_list: Tuple[int, ...] = field(default_factory=lambda: (128, 16, 4, 1))
    embed_dim_list: Tuple[int, ...] = field(default_factory=lambda: (512, 256, 128, 64))
    num_blocks_list: Tuple[int, ...] = field(default_factory=lambda: (12, 3, 2, 1))
    num_heads_list: Tuple[int, ...] = field(default_factory=lambda: (8, 4, 2, 2))
    
    # Training configuration
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    guiding_pixel: bool = False
    num_conds: int = 5
    v_weight: float = 1.0
    grad_checkpointing: bool = False
    generator_type_list: Tuple[str, ...] = field(default_factory=lambda: ("mar", "mar", "mar", "mar"))
    scan_order: str = 'row_major'
    mask_ratio_loc: float = 1.0  # Mean mask ratio for MAR (1.0 = mask 100%, 0.5 = mask 50%)
    mask_ratio_scale: float = 0.5  # Standard deviation of mask ratio
    ar_prefix_mask_ratio: float = 1.0  # Maximum prefix mask ratio for AR training (1.0 = can mask up to 100%)

    def __post_init__(self):
        allowed = {"mar", "ar"}
        if len(self.generator_type_list) != 4:
            raise ValueError("generator_type_list must contain exactly 4 entries for levels 0-3")
        for idx, g in enumerate(self.generator_type_list):
            if g not in allowed:
                raise ValueError(f"generator_type_list[{idx}] must be one of {allowed}, got '{g}'")
        if self.scan_order not in {'row_major', 'column_major'}:
            raise ValueError(f"scan_order must be 'row_major' or 'column_major', got '{self.scan_order}'")
        
        # Validate architecture lists
        if len(self.img_size_list) != 4:
            raise ValueError("img_size_list must contain exactly 4 entries")
        if len(self.embed_dim_list) != 4:
            raise ValueError("embed_dim_list must contain exactly 4 entries")
        if len(self.num_blocks_list) != 4:
            raise ValueError("num_blocks_list must contain exactly 4 entries")
        if len(self.num_heads_list) != 4:
            raise ValueError("num_heads_list must contain exactly 4 entries")


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
        model = fractalmar_piano(
            img_size_list=model_cfg.img_size_list,
            embed_dim_list=model_cfg.embed_dim_list,
            num_blocks_list=model_cfg.num_blocks_list,
            num_heads_list=model_cfg.num_heads_list,
            attn_dropout=model_cfg.attn_dropout,
            proj_dropout=model_cfg.proj_dropout,
            guiding_pixel=model_cfg.guiding_pixel,
            num_conds=model_cfg.num_conds,
            v_weight=model_cfg.v_weight,
            grad_checkpointing=model_cfg.grad_checkpointing,
            generator_type_list=model_cfg.generator_type_list,
            scan_order=model_cfg.scan_order,
            mask_ratio_loc=model_cfg.mask_ratio_loc,
            mask_ratio_scale=model_cfg.mask_ratio_scale,
            ar_prefix_mask_ratio=model_cfg.ar_prefix_mask_ratio
        )
        return model
    
    def forward(self, piano_rolls, durations=None):
        """
        Forward pass through FractalGen.
        
        Args:
            piano_rolls: (B, 128, T) piano roll tensors
            durations: (B,) original durations (not used by FractalGen)
        
        Returns:
            loss: Hierarchical loss from FractalGen
        """
        # Add channel dimension: (B, 128, T) -> (B, 1, 128, T)
        if piano_rolls.dim() == 3:
            piano_rolls = piano_rolls.unsqueeze(1)
        
        # FractalGen forward computes loss recursively through all levels
        # Note on OOM: The batch size expands at each level of the hierarchy.
        # Level 0: B
        # Level 1: B * (128/16)^2 = B * 64
        # Level 2: B * 64 * (16/4)^2 = B * 1024
        # Level 3: B * 1024 * (4/1)^2 = B * 16384
        # With B=8, Level 3 processes ~131k items.
        # Use grad_checkpointing=True in config to save memory at cost of compute.
        loss, stats = self.model(piano_rolls)
        
        return loss, stats
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        pitch_shifts = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            piano_rolls, durations, pitch_shifts = batch
        else:
            piano_rolls, durations = batch
        
        # Forward pass (already computes loss)
        loss, stats = self(piano_rolls, durations)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self._log_metrics(stats, split='train', on_step=True, on_epoch=False)

        # Pitch shift statistics (optional, only in TensorBoard)
        if pitch_shifts is not None:
            pitch_shifts = pitch_shifts.to(piano_rolls.device)
            self.log('train/pitch_shift_mean', pitch_shifts.float().mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False)
        
        # Log learning rate (only if trainer is attached)
        try:
            if hasattr(self, '_trainer') and self._trainer is not None:
                self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        except:
            pass  # Skip if optimizer/trainer not available yet
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pitch_shifts = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            piano_rolls, durations, pitch_shifts = batch
        else:
            piano_rolls, durations = batch
        
        # Forward pass
        loss, stats = self(piano_rolls, durations)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._log_metrics(stats, split='val', on_step=False, on_epoch=True)

        # Note: Validation typically doesn't use pitch shift augmentation
        # (only training does), so this is usually not logged
        
        # Log images periodically (only on first batch to avoid redundancy)
        if batch_idx == 0:
            should_log_images = (
                self.config.log_images_every_n_steps is not None
                and self.config.log_images_every_n_steps > 0
                and self.global_step > 0
                and self.global_step % self.config.log_images_every_n_steps == 0
            )
            
            # Also log if this is a validation step triggered by val_check_interval_steps
            # Because validation happens based on batches (via val_check_interval), 
            # but global_step is updated based on accumulated grad batches.
            # If val_check_interval_steps (batches) is consistent with log_images_every_n_steps (steps),
            # we should ensure we log when validation happens.
            
            if should_log_images:
                self._log_images(batch, batch_idx, 'val', force_sample=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        pitch_shifts = None
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            piano_rolls, durations, pitch_shifts = batch
        else:
            piano_rolls, durations = batch
        
        # Forward pass
        loss, stats = self(piano_rolls, durations)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self._log_metrics(stats, split='test', on_step=False, on_epoch=True)

        return loss
    
    def _log_images(self, batch, batch_idx, split='train', force_sample=False):
        """Log piano rolls and generated samples during validation."""
        if self.trainer is None or self.logger is None:
            return
        if self.trainer.sanity_checking:
            return
        
        # Only log during validation
        if split != 'val':
            return
        
        if isinstance(batch, (list, tuple)):
            piano_rolls = batch[0]
        else:
            piano_rolls = batch
        num_to_log = min(self.config.num_images_to_log, piano_rolls.size(0))
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
        
        # Log ground truth (only once, not every validation)
        # Check if ground truth has been logged before
        if not hasattr(self, '_ground_truth_logged') or not self._ground_truth_logged:
            for i in range(num_to_log):
                roll = piano_rolls[i]
                if roll.dim() == 3:
                    roll = roll.squeeze(0)
                try:
                    log_piano_roll_to_tensorboard(
                        self.logger.experiment,
                        f'{split}_images/0_ground_truth/sample_{i}',  # Better organization, no step number
                        roll.detach().cpu(),
                        self.global_step,
                        apply_colormap=True
                    )
                except Exception as exc:
                    print(f"Warning: Failed to log ground truth {i}: {exc}")
            
            # Mark as logged
            self._ground_truth_logged = True

        # Generate and log samples with intermediates for GIF
        samples_to_generate = num_to_log
        try:
            with torch.no_grad():
                generated, intermediates = self.model.sample(
                    batch_size=samples_to_generate,
                    cond_list=None,
                    num_iter_list=[8, 4, 2, 1],
                    cfg=1.0,
                    cfg_schedule="constant",
                    temperature=1.0,
                    filter_threshold=0,
                    return_intermediates=True
                )
        except Exception as exc:
            print(f"Warning: Failed to generate samples for logging: {exc}")
            import traceback
            traceback.print_exc()
            return

        if torch.is_tensor(generated) and generated.dim() == 4:
            generated = generated.squeeze(1)

        if not torch.is_tensor(generated):
            return

        generated = generated.detach().cpu()
        if generated.dim() == 2:
            generated = generated.unsqueeze(0)

        # Log final generated samples
        for i in range(min(samples_to_generate, generated.size(0))):
            roll = generated[i]
            if roll.dim() == 3:
                roll = roll.squeeze(0)
            try:
                log_piano_roll_to_tensorboard(
                    self.logger.experiment,
                    f'{split}_images/1_generated_step_{self.global_step:07d}/sample_{i}',  # Group by step
                    roll,
                    self.global_step,
                    apply_colormap=True
                )
            except Exception as exc:
                print(f"Warning: Failed to log generated sample {i}: {exc}")
        
        # Create and log GIFs from intermediates
        if intermediates is not None and len(intermediates) > 0:
            self._create_generation_gifs(intermediates, samples_to_generate, split)

    def _create_generation_gifs(self, intermediates, num_samples, split='val'):
        """Create GIF animations from generation intermediates using growth animation."""
        try:
            from visualizer import create_growth_animation
            import os
            import numpy as np
            
            gif_dir = os.path.join(self.logger.log_dir, 'generation_gifs')
            os.makedirs(gif_dir, exist_ok=True)
            
            # Note: AR generator only records the first sample (sample_idx=0)
            # So we can only visualize sample 0
            
            sample_frames = []
            
            # Extract frames for sample 0
            for item in intermediates:
                if isinstance(item, dict) and 'output' in item:
                    # output is (1, C, H, W) or similar for AR
                    frame = item['output']
                    if isinstance(frame, torch.Tensor):
                        if frame.ndim == 4: frame = frame[0] # (C, H, W)
                        if frame.ndim == 3: frame = frame[0] # (H, W) if C=1
                        sample_frames.append(frame)
                elif isinstance(item, torch.Tensor):
                    # Legacy/other models
                    frame = item
                    if frame.ndim == 4: frame = frame[0] # Take first sample
                    sample_frames.append(frame)

            if not sample_frames:
                return

            # Generate GIF for sample 0
            gif_path = os.path.join(gif_dir, f'step_{self.global_step:07d}_sample_0.gif')
            
            # Create smooth animation with improved quality and easing
            frames = create_growth_animation(
                sample_frames,
                save_path=gif_path,
                fps=24,  # Slightly higher FPS for smoother animation
                transition_duration=0.15,  # Longer transitions for smoother effect
                min_height=512,  # Higher resolution
                easing="ease_in_out_cubic",  # Smooth easing
                pop_effect=True,  # Enable note-popping effect
                optimize=True,
                quality=90
            )
            
            if frames is None:
                # Saved successfully
                    print(f"Saved generation GIF: {gif_path}")
                
        except Exception as exc:
            print(f"Warning: Failed to create generation GIFs: {exc}")
            import traceback
            traceback.print_exc()

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
    
    def _create_hierarchical_gif(self, intermediates, temperature):
        """Create GIF showing progressive pixel filling (like visual.gif)."""
        from visualizer import piano_roll_to_image
        from PIL import Image
        import numpy as np
        
        frames = []
        
        try:
            # Process each iteration (all same size, progressively filled)
            for inter in intermediates:
                output = inter['output']
                
                # Extract first sample
                if output.dim() == 4:
                    output = output[0]
                if output.dim() == 3:
                    output = output[0]
                
                # Convert to image
                img_tensor = piano_roll_to_image(output.cpu())
                
                # To PIL
                if torch.is_tensor(img_tensor):
                    img_np = img_tensor.squeeze().cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = img_np.transpose(1, 2, 0)
                    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                    if img_np.ndim == 2:
                        pil_img = Image.fromarray(img_np, mode='L').convert('RGB')
                    else:
                        pil_img = Image.fromarray(img_np, mode='RGB')
                else:
                    pil_img = img_tensor.convert('RGB')
                
                # No text label, just the image
                frames.append(pil_img)
            
            return frames
        except Exception as e:
            print(f"    Error creating GIF: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_with_visualization(self, batch_size, temperature, device):
        """
        Generate a sample while capturing intermediate steps for GIF visualization.
        
        Returns:
            final_output: The final generated piano roll
            intermediate_frames: List of PIL Images showing generation progress
        """
        from visualizer import piano_roll_to_image
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        intermediate_frames = []
        
        # Modified sample method that captures intermediate states
        # We'll do a simplified version focusing on the first level
        num_iter_list = [8, 4, 2, 1]  # 4 levels
        
        # Start with zeros
        H, W = 128, 256  # Fixed size for visualization
        current_roll = torch.zeros(batch_size, 1, H, W, device=device)
        
        # Add initial state (all zeros)
        frame = self._piano_roll_to_pil_image(current_roll[0], "Initial (all zeros)")
        intermediate_frames.append(frame)
        
        # Generate step by step (simplified for visualization)
        try:
            # Do actual generation
            final_output = self.model.sample(
                batch_size=batch_size,
                cond_list=None,
                num_iter_list=num_iter_list,
                cfg=1.0,
                temperature=temperature,
                visualize=False  # We handle visualization ourselves
            )
            
            # Create intermediate frames by gradually revealing the result
            # This simulates the iterative generation process
            num_steps = 8
            for step in range(1, num_steps + 1):
                ratio = step / num_steps
                # Mix zeros and final result to simulate progressive generation
                mixed = current_roll * (1 - ratio) + final_output * ratio
                
                frame = self._piano_roll_to_pil_image(
                    mixed[0], 
                    f"Step {step}/{num_steps} ({ratio*100:.0f}%)"
                )
                intermediate_frames.append(frame)
            
            # Add final result
            frame = self._piano_roll_to_pil_image(final_output[0], f"Final Result")
            intermediate_frames.append(frame)
            
        except Exception as e:
            print(f"    Error during generation: {e}")
            import traceback
            traceback.print_exc()
            # Return what we have
            return current_roll, intermediate_frames
        
        return final_output, intermediate_frames
    
    def _piano_roll_to_pil_image(self, piano_roll, label=""):
        """Convert a piano roll tensor to PIL Image with label."""
        from visualizer import piano_roll_to_image
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Remove channel dimension if present
        if piano_roll.dim() == 4:
            piano_roll = piano_roll.squeeze(0)
        if piano_roll.dim() == 3:
            piano_roll = piano_roll.squeeze(0)
        
        # Convert to image using visualizer
        frame_tensor = piano_roll_to_image(piano_roll.cpu())
        
        # Convert to PIL Image
        if torch.is_tensor(frame_tensor):
            frame_np = frame_tensor.squeeze().cpu().numpy()
            if frame_np.ndim == 3:  # (C, H, W)
                frame_np = frame_np.transpose(1, 2, 0)
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
            if frame_np.ndim == 2:
                frame_pil = Image.fromarray(frame_np, mode='L')
            else:
                frame_pil = Image.fromarray(frame_np, mode='RGB')
        else:
            frame_pil = frame_tensor
        
        # Add label text
        if label:
            draw = ImageDraw.Draw(frame_pil)

            font = ImageFont.load_default()
            
            # Add background for text
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.rectangle([(5, 5), (text_width + 15, text_height + 15)], fill='black')
            draw.text((10, 10), label, fill='white', font=font)
        
        return frame_pil


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
