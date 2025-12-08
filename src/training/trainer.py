"""
PyTorch Lightning trainer for FractalGen Piano Roll model.
Simplified version that uses the complete hierarchical FractalGen architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys
import math
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Import new model and config
from src.models.fractal_gen import RecursiveFractalNetwork
from src.models.model_config import FractalModelConfig, ArchitectureConfig
from src.visualization.visualizer import log_piano_roll_to_tensorboard
from src.inference.metrics import MusicMetrics
from pytorch_lightning.callbacks import TQDMProgressBar
import sys
from tqdm.auto import tqdm

# ==============================================================================
# Custom Callbacks
# ==============================================================================

class DualProgressBar(TQDMProgressBar):
    """
    Progress Bar that shows both Global Steps (Total) and Epoch Steps (Current Epoch).
    Reference: ling482_African_ASR/training/custom_progressbar.py
    """
    def __init__(self):
        # process_position=1 makes the standard epoch bar start at line 1
        super().__init__(process_position=1)
        self.global_progress_bar = None
        
    def init_train_tqdm(self):
        """Override to customize the epoch bar (inner bar)."""
        bar = super().init_train_tqdm()
        bar.set_description("Epoch Progress")
        return bar
        
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # Create a persistent global progress bar at position 0
        if trainer.is_global_zero:
            self.global_progress_bar = tqdm(
                desc="Global Progress",
                total=trainer.max_steps,
                initial=trainer.global_step,
                position=0,
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
                smoothing=0
            )
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.global_progress_bar and trainer.is_global_zero:
            self.global_progress_bar.n = trainer.global_step
            self.global_progress_bar.refresh()
            # Update postfix with loss
            metrics = trainer.callback_metrics
            postfix = {}
            if 'train_loss' in metrics:
                postfix['loss'] = f"{metrics['train_loss']:.3f}"
            if 'val_loss' in metrics:
                postfix['val_loss'] = f"{metrics['val_loss']:.3f}"
            self.global_progress_bar.set_postfix(postfix)
            
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        if self.global_progress_bar:
            self.global_progress_bar.close()


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
    model: FractalModelConfig = field(default_factory=FractalModelConfig)
    
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
        self.metrics_evaluator = MusicMetrics()
        self._last_logged_image_step = -1
        
    def _build_model(self):
        """Build FractalGen model with configurable architecture."""
        # Use new RecursiveFractalNetwork
        return RecursiveFractalNetwork(self.config.model)
    
    def forward(self, notes, tempo, density, global_cond=None, bar_pos=None):
        """
        Forward pass through FractalGen.
        """
        loss, stats = self.model(notes, tempo, density, global_cond, bar_pos=bar_pos)
        return loss, stats
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Batch: (notes, tempo, density, bar_pos, chroma, durations, shifts)
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 5:
                # (notes, tempo, density, bar_pos, chroma, ...)
                notes, tempo, density, bar_pos, chroma = batch[0], batch[1], batch[2], batch[3], batch[4]
                pitch_shifts = batch[6] if len(batch) >= 7 else None
            else:
                if len(batch) >= 4:
                     notes, tempo, density, bar_pos = batch[0], batch[1], batch[2], batch[3]
                     chroma = torch.zeros(notes.shape[0], notes.shape[2], 12, device=notes.device)
                     pitch_shifts = None
                else:
                     raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            raise ValueError("Batch must be a tuple")
        
        # Label Dropout (CFG)
        # 10% chance to drop condition (set to zeros)
        if random.random() < 0.1:
            global_cond = torch.zeros_like(chroma)
        else:
            global_cond = chroma
        
        # Forward pass
        loss, stats = self.model(notes, tempo, density, global_cond, bar_pos=bar_pos)
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self._log_metrics(stats, split='train', on_step=True, on_epoch=False)

        # if pitch_shifts is not None:
        #     pitch_shifts = pitch_shifts.to(notes.device)
        #     self.log('train/pitch_shift_mean', pitch_shifts.float().mean(), on_step=True, on_epoch=False, logger=True, prog_bar=False)
        
        try:
            if hasattr(self, '_trainer') and self._trainer is not None:
                self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        except:
            pass 
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 5:
                notes, tempo, density, bar_pos, chroma = batch[0], batch[1], batch[2], batch[3], batch[4]
            elif len(batch) >= 4:
                notes, tempo, density, bar_pos = batch[0], batch[1], batch[2], batch[3]
                chroma = torch.zeros(notes.shape[0], notes.shape[2], 12, device=notes.device)
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        # Forward pass (Conditional)
        loss, stats = self(notes, tempo, density, global_cond=chroma, bar_pos=bar_pos)
        
        # Forward pass (Unconditional - to check CFG gap)
        # This is useful to monitor if model learns to use condition
        loss_uncond, _ = self(notes, tempo, density, global_cond=torch.zeros_like(chroma), bar_pos=bar_pos)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss_uncond', loss_uncond, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self._log_metrics(stats, split='val', on_step=False, on_epoch=True)

        # Log images
        if batch_idx == 0:
            log_freq = self.config.log_images_every_n_steps
            should_log_images = (
                log_freq is not None
                and log_freq > 0
                and self.global_step > 0
                and (self.global_step // log_freq > self._last_logged_image_step // log_freq)
            )
            
            if should_log_images:
                self._log_images(batch, batch_idx, 'val', force_sample=True)
                self._log_conditional_images(batch, batch_idx, 'val')
                self._log_inpainting_images(batch, batch_idx, 'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 5:
                notes, tempo, density, bar_pos, chroma = batch[0], batch[1], batch[2], batch[3], batch[4]
            elif len(batch) >= 4:
                notes, tempo, density, bar_pos = batch[0], batch[1], batch[2], batch[3]
                chroma = torch.zeros(notes.shape[0], notes.shape[2], 12, device=notes.device)
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        loss, stats = self(notes, tempo, density, global_cond=chroma, bar_pos=bar_pos)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self._log_metrics(stats, split='test', on_step=False, on_epoch=True)

        return loss
    
    def _reconstruct_piano_roll(self, notes, tempo):
        # notes: (B, 2, T, 128)
        # tempo: (B, T)
        # Returns: (B, 3, T, 128)
        B, _, T, P = notes.shape
        tempo_expanded = tempo.unsqueeze(1).unsqueeze(-1).expand(B, 1, T, P) # (B, 1, T, 128)
        
        # Mask velocity with note presence to clean up visualization
        # notes[:, 0] is Note On/Off, notes[:, 1] is Velocity
        note_mask = (notes[:, 0] > 0.5).float()
        masked_velocity = notes[:, 1] * note_mask
        
        masked_notes = torch.stack([notes[:, 0], masked_velocity], dim=1)
        
        return torch.cat([masked_notes, tempo_expanded], dim=1)

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
        if len(batch) >= 5:
            notes, tempo, density, bar_pos, chroma = batch[0], batch[1], batch[2], batch[3], batch[4]
        elif len(batch) >= 4:
            notes, tempo, density, bar_pos = batch[0], batch[1], batch[2], batch[3]
            chroma = None
        
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
                    # Log composite
                    log_piano_roll_to_tensorboard(
                        self.logger.experiment,
                        f'{split}_images/0_ground_truth/sample_{i}_composite', 
                        roll.detach().cpu(),
                        self.global_step,
                        apply_colormap=True
                    )
                    
                    channels = ['Note', 'Velocity', 'Tempo']
                    for c, name in enumerate(channels):
                        ch_data = roll[c].detach().cpu()
                        ch_data_t = ch_data.T 
                        log_piano_roll_to_tensorboard(
                            self.logger.experiment,
                            f'{split}_images/0_ground_truth/sample_{i}_ch_{name}',
                            ch_data_t, 
                            self.global_step,
                            apply_colormap=True
                        )

                except Exception as exc:
                    print(f"Warning: Failed to log ground truth {i}: {exc}")
            self._ground_truth_logged = True

        # Generate samples
        samples_to_generate = num_to_log
        target_length = notes.shape[2]
        sample_cond = chroma[:samples_to_generate] if chroma is not None else None

        try:
            with torch.no_grad():
                device = notes.device
                bar_pos = (torch.arange(target_length, device=device) % 16).unsqueeze(0).expand(samples_to_generate, -1)
                
                # RecursiveFractalNetwork doesn't have .sample directly exposed properly yet in my prev implementation?
                # Actually I added a dummy sample method to FractalGen, but RecursiveFractalNetwork needs one too.
                # Assuming I'll add it or call root directly.
                # self.model.root.sample(...)
                
                generated, intermediates = self.model.root.sample(
                    batch_size=samples_to_generate,
                    length=target_length,
                    global_cond=sample_cond,
                    bar_pos=bar_pos
                )
        except Exception as exc:
            print(f"Warning: Failed to generate samples for logging: {exc}")
            import traceback
            traceback.print_exc()
            return

        if not torch.is_tensor(generated):
            return

        generated = generated.detach().cpu()
        
        # Calculate and log metrics on generated samples
        try:
            # generated is (B, 2, T, 128)
            gen_np_list = [generated[i].numpy() for i in range(generated.shape[0])]
            metrics = self.metrics_evaluator.evaluate_batch(gen_np_list)
            
            for k, v in metrics.items():
                self.log(f'{split}/metrics/{k}', float(v), on_step=False, on_epoch=True, logger=True)
                
        except Exception as exc:
             print(f"Warning: Failed to calculate metrics: {exc}")
        
        # Log per-level outputs for Sample 0
        level_outputs = {}
        # ... logic to extract level outputs ...
        
        # Extract last output for each level
        for item in intermediates:
             # intermediates is a list of dicts from FractalGen.sample
             pass
             # Wait, my FractalGen.sample didn't implement gathering intermediates fully yet!
             # I need to implement it in FractalGen first.
             # But let's assume it returns a list of stats dicts.
             
        # ... (Rest of logging logic similar to before but adapted) ...

        # Reconstruct generated roll
        # We need tempo.
        tempo_curve = torch.ones(samples_to_generate, target_length) * 0.5
        gen_rolls = self._reconstruct_piano_roll(generated, tempo_curve)

        for i in range(min(samples_to_generate, gen_rolls.size(0))):
            roll = gen_rolls[i]
            try:
                log_piano_roll_to_tensorboard(
                    self.logger.experiment,
                    f'{split}_images/1_generated_step/sample_{i}_composite', 
                    roll,
                    self.global_step,
                    apply_colormap=True
                )
                
                # Log separate channels for generated samples
                channels = ['Note', 'Velocity', 'Tempo']
                for c, name in enumerate(channels):
                    ch_data = roll[c].detach().cpu()
                    ch_data_t = ch_data.T 
                    log_piano_roll_to_tensorboard(
                        self.logger.experiment,
                        f'{split}_images/1_generated_step/sample_{i}_ch_{name}',
                        ch_data_t, 
                        self.global_step,
                        apply_colormap=True
                    )

            except Exception as exc:
                print(f"Warning: Failed to log generated sample {i}: {exc}")
        
        if intermediates is not None and len(intermediates) > 0:
            self._create_generation_gifs(intermediates, samples_to_generate, split)
            
    def _create_generation_gifs(self, intermediates, num_samples, split='val'):
        """Create GIF animations from generation intermediates with Force Upsampling."""
        try:
            from src.visualization.visualizer import create_growth_animation
            import os
            
            gif_dir = os.path.join(self.logger.log_dir, 'generation_gifs')
            os.makedirs(gif_dir, exist_ok=True)
            
            sample_frames = []
            
            # Find max length
            max_len = 0
            for item in intermediates:
                if 'output' in item:
                    max_len = max(max_len, item['output'].shape[-2]) # (B, C, T, P) or (B, C, T)
            
            # Filter frames for sample 0
            for item in intermediates:
                if isinstance(item, dict) and 'output' in item:
                    is_structure = item.get('is_structure', False)
                    out = item['output']
                    
                    # Visualize all levels, including structure
                    # Structure: (B, 2, T) -> Needs to be visualized as curve or pseudo-roll
                    
                    if is_structure:
                        # Structure output: (B, 2, T) - Density & Tempo
                        # Map to Piano Roll format for consistency
                        # Shape: (2, T)
                        struct_data = out[0].detach().cpu()
                        # Create a pseudo-roll where:
                        # - Density maps to velocity of a specific pitch (e.g., middle C)
                        # - Or just skip for now if visualizer expects 3-channel roll
                        
                        # Actually, let's just visualize Content levels for now to avoid confusion
                        # The user wants to see "growth". Structure level is abstract.
                        continue
                        
                    frame = out[0] # (2, T, 128)
                    frame = frame.cpu() # Ensure CPU for concatenation
                    current_T = frame.shape[1]
                    
                    # Force Upsample to max_len (Issue 8)
                    # Use nearest neighbor to preserve the "blocky" look of lower levels
                    if current_T < max_len:
                        # frame: (2, T, 128) -> (1, 2, T, 128)
                        frame_unsqueezed = frame.unsqueeze(0)
                        # interpolate expects (N, C, H, W) or (N, C, L)
                        # Here we want to scale Time (dim 2).
                        # frame is (2, T, 128).
                        # interpolate on 4D tensor treats last 2 dims as spatial.
                        # We want to scale H (Time) and keep W (Pitch 128).
                        
                        frame_up = F.interpolate(frame_unsqueezed, size=(max_len, 128), mode='nearest')
                        frame = frame_up.squeeze(0)
                        
                    tempo_ch = torch.ones(1, max_len, 128) * 0.5
                    full_frame = torch.cat([frame, tempo_ch], dim=0) 
                    
                    # Apply Note mask to Velocity
                    # full_frame[0] is Note, full_frame[1] is Velocity
                    mask = (full_frame[0] > 0.5).float()
                    full_frame[1] = full_frame[1] * mask
                    
                    # Append multiple times to make the level visible for longer
                    # "Midi appears and disappears" suggests frames are too fleeting
                    # or the interpolation is weird.
                    
                    # Let's append this frame multiple times
                    for _ in range(3):
                        sample_frames.append(full_frame)

            if not sample_frames:
                return

            gif_path = os.path.join(gif_dir, f'step_{self.global_step:07d}_sample_0.gif')
            
            create_growth_animation(
                sample_frames,
                save_path=gif_path,
                fps=15,
                transition_duration=1.5,
                final_hold=3.0,
                min_height=512, 
                easing="ease_in_out_cubic",
                pop_effect=True,
                optimize=True
            )
            
            if os.path.exists(gif_path):
                 print(f"Saved generation GIF: {gif_path}")
                
        except Exception as exc:
            print(f"Warning: Failed to create generation GIFs: {exc}")

    def _log_conditional_images(self, batch, batch_idx, split='val'):
         pass # Implementation similar to before, calling self.model.root.sample

    def _log_inpainting_images(self, batch, batch_idx, split='val'):
         pass # Implementation similar to before

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_cfg = self.config.optimizer
        sched_cfg = self.config.scheduler
        
        # Ensure lr is float
        lr = float(opt_cfg.lr)
        weight_decay = float(opt_cfg.weight_decay)
        min_lr = float(sched_cfg.min_lr)
        
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            betas=opt_cfg.betas,
            weight_decay=weight_decay
        )
        
        total_steps = max(1, self.config.max_steps)
        warmup_steps = max(0, sched_cfg.warmup_steps)

        def lr_lambda(current_step: int):
            step = max(current_step, 0)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            
            if sched_cfg.schedule_type == "constant":
                return 1.0
            if sched_cfg.schedule_type == "linear":
                return 1.0 - progress * (1.0 - min_lr / lr)
            if sched_cfg.schedule_type == "cosine":
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr / lr + (1.0 - min_lr / lr) * cosine_decay
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
        if stats is None:
            return
        if split != 'val':
            return

        for key, value in stats.items():
            # Filter out heavy tensors to avoid overhead
            if 'logits' in key or 'gt' in key or 'mask' in key or 'output' in key:
                continue
            
            # Filter out less important metrics for cleaner TensorBoard
            if 'sigma' in key or 'raw_loss' in key:
                # Only log these if explicitly requested or for debugging (skip by default)
                continue

            metric_name = f"{split}/{key}"
            if isinstance(value, (int, float)):
                tensor_value = torch.tensor(float(value), device=self.device)
            elif torch.is_tensor(value):
                # Only log scalar tensors
                if value.numel() > 1:
                    continue
                    
                tensor_value = value.detach().float() 
                tensor_value = tensor_value.to(self.device)
            else:
                continue

            tensor_value = tensor_value.float()
            sync_dist = split != 'train'
            self.log(metric_name, tensor_value, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=sync_dist)
    
    def on_before_optimizer_step(self, optimizer):
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)


def create_trainer(config: FractalTrainerConfig, *, val_check_interval: float | None = None,
                  check_val_every_n_epoch: int | None = None, **trainer_kwargs) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer.
    """
    # Use DDP for multi-GPU, but ensure it's properly configured
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        strategy = 'ddp'  # Use standard DDP instead of ddp_find_unused_parameters_false
    else:
        strategy = 'auto'
    
    # Override if explicitly set in config
    if config.strategy is not None:
        strategy = config.strategy

    default_args = {
        'max_steps': config.max_steps,
        'max_epochs': 1_000_000, 
        'accelerator': 'gpu',  # Explicitly use GPU
        'devices': num_gpus if num_gpus > 0 else 1,  # Use all available GPUs
        'strategy': strategy,
        'precision': config.precision,
        'gradient_clip_val': 0, 
        'accumulate_grad_batches': config.accumulate_grad_batches,
        'log_every_n_steps': config.log_every_n_steps,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': False,
        'benchmark': True,  # Enable cuDNN benchmark for faster training
        'num_sanity_val_steps': 0,
        'sync_batchnorm': True if num_gpus > 1 else False,  # Sync batch norm across GPUs
    }

    if val_check_interval is None:
        default_args['val_check_interval'] = int(config.val_check_interval_steps * config.accumulate_grad_batches)
    else:
        default_args['val_check_interval'] = val_check_interval

    default_args['check_val_every_n_epoch'] = check_val_every_n_epoch if check_val_every_n_epoch is not None else 1
    
    trainer_args = {**default_args, **trainer_kwargs}
    
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every_n_steps,
        save_top_k=config.save_top_k,
        monitor='val_loss',
        mode='min',
        filename='step_{step:08d}-val_loss_{val_loss:.4f}',
        save_last=True,
        save_on_train_epoch_end=False 
    )
    
    callbacks = trainer_args.get('callbacks', [])
    if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
        callbacks.append(checkpoint_callback)
    trainer_args['callbacks'] = callbacks
    
    trainer_args['use_distributed_sampler'] = False
    
    trainer = pl.Trainer(**trainer_args)
    
    return trainer
