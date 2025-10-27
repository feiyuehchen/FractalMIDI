"""
PyTorch Lightning trainer for FractalGen Piano Roll model.
Simplified version that uses the complete hierarchical FractalGen architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

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
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class FractalTrainerConfig:
    """Main trainer configuration."""
    # Single model architecture: 128→16→4→1
    
    # Training hyperparameters
    max_epochs: int = 50
    grad_clip: float = 3.0
    
    # Visualization
    log_images_every_n_epochs: int = 1
    num_images_to_log: int = 4
    
    # Sub-configs
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.grad_clip >= 0, "grad_clip must be non-negative"


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
        
    def _build_model(self):
        """Build FractalGen model (128→16→4→1)."""
        model = fractalmar_piano(
            attn_dropout=0.0,
            proj_dropout=0.0,
            guiding_pixel=False,
            num_conds=5,
            v_weight=1.0
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
        loss = self.model(piano_rolls)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        piano_rolls, durations = batch
        
        # Forward pass (already computes loss)
        loss = self(piano_rolls, durations)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Log learning rate (only if trainer is attached)
        try:
            if hasattr(self, '_trainer') and self._trainer is not None:
                self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, prog_bar=True, logger=True)
        except:
            pass  # Skip if optimizer/trainer not available yet
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        piano_rolls, durations = batch
        
        # Forward pass
        loss = self(piano_rolls, durations)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # Log images periodically
        if batch_idx == 0 and self.current_epoch % self.config.log_images_every_n_epochs == 0:
            self._log_images(batch, batch_idx, 'val')
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        piano_rolls, durations = batch
        
        # Forward pass
        loss = self(piano_rolls, durations)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return loss
    
    def _log_images(self, batch, batch_idx, split='train'):
        """Log piano roll images to TensorBoard and save GIF."""
        if self.trainer is None or self.logger is None:
            return
        
        # Only log during validation
        if split != 'val':
            return
        
        # No need for persistent GIF frames list anymore
        # Each epoch will create its own generation process GIF
        
        piano_rolls, durations = batch
        num_to_log = min(self.config.num_images_to_log, piano_rolls.size(0))
        
        # Log ground truth only once at epoch 0 (not during sanity check)
        if not hasattr(self, '_logged_gt'):
            if self.current_epoch == 0 and not self.trainer.sanity_checking:
                print(f"\n📸 Logging ground truth images (epoch {self.current_epoch})...")
                for i in range(num_to_log):
                    piano_roll = piano_rolls[i]
                    if piano_roll.dim() == 3:
                        piano_roll = piano_roll.squeeze(0)
                    
                    try:
                        log_piano_roll_to_tensorboard(
                            self.logger.experiment,
                            f'val/ground_truth/sample_{i}',
                            piano_roll,
                            self.global_step,
                            apply_colormap=True
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log ground truth {i}: {e}")
                self._logged_gt = True
                print(f"✓ Ground truth logged!")
            else:
                return  # Skip if not epoch 0 and not logged yet
        
        # Generate samples every epoch (only if ground truth already logged)
        if not hasattr(self, '_logged_gt'):
            return  # Wait until ground truth is logged
        
        print(f"\n🎨 Generating samples with hierarchical visualization for epoch {self.current_epoch}...")
        with torch.no_grad():
            generated_samples = []
            
            for i in range(min(1, num_to_log)):
                try:
                    # Use different random seed for each epoch
                    torch.manual_seed(self.current_epoch * 1000 + i)
                    
                    # Vary temperature slightly based on epoch for diversity
                    temperature = 1.0 + (self.current_epoch % 5) * 0.1
                    
                    # Generate sample with intermediate outputs
                    result = self.model.sample(
                        batch_size=1,
                        cond_list=None,
                        num_iter_list=[8, 4, 2, 1],
                        cfg=1.0,
                        temperature=temperature,
                        return_intermediates=True
                    )
                    
                    if isinstance(result, tuple):
                        gen, intermediates = result
                        print(f"  ✓ Generated with {len(intermediates)} hierarchical levels")
                        
                        # Create GIF from hierarchical outputs
                        gif_frames = self._create_hierarchical_gif(intermediates, temperature)
                        if gif_frames:
                            gif_path = f"{self.trainer.log_dir}/generation_epoch_{self.current_epoch:03d}.gif"
                            
                            duration_per_frame = 10  
                            actual_total_sec = len(gif_frames) * duration_per_frame / 1000
                            
                            gif_frames[0].save(
                                gif_path,
                                save_all=True,
                                append_images=gif_frames[1:],
                                duration=duration_per_frame,
                                loop=0,
                                optimize=False 
                            )
                            print(f"  🎬 GIF saved: {gif_path} ({len(gif_frames)} frames, {duration_per_frame}ms/frame = {actual_total_sec:.1f}s total)")
                    else:
                        gen = result
                        print(f"  ✓ Generated sample {i} (shape: {gen.shape}, temp={temperature:.2f})")
                    
                    generated_samples.append(gen.squeeze(1) if gen.dim() == 4 else gen)
                    
                except Exception as e:
                    print(f"  ✗ Failed to generate sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Log generated images to TensorBoard
            if generated_samples:
                print(f"  📊 Logging to TensorBoard...")
                for i, gen_roll in enumerate(generated_samples):
                    if gen_roll.dim() == 3:
                        gen_roll = gen_roll.squeeze(0)
                    
                    try:
                        log_piano_roll_to_tensorboard(
                            self.logger.experiment,
                            f'val/generated/epoch_{self.current_epoch:03d}_sample_{i}',
                            gen_roll,
                            self.global_step,
                            apply_colormap=True
                        )
                        print(f"    ✓ Logged sample {i}")
                    except Exception as e:
                        print(f"    ✗ Failed to log generated {i}: {e}")
                
            else:
                print(f"  ⚠️  No samples generated")


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
        def lr_lambda(current_step: int):
            """Learning rate schedule function."""
            try:
                total_steps = self.trainer.estimated_stepping_batches
                current_epoch = current_step / max(1, total_steps) * self.config.max_epochs
            except (AttributeError, RuntimeError):
                # For testing without trainer
                current_epoch = current_step / 1000.0
            
            # Warmup
            if current_epoch < sched_cfg.warmup_epochs:
                return current_epoch / sched_cfg.warmup_epochs
            
            # After warmup
            progress = (current_epoch - sched_cfg.warmup_epochs) / (self.config.max_epochs - sched_cfg.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            
            if sched_cfg.schedule_type == "constant":
                return 1.0
            elif sched_cfg.schedule_type == "linear":
                return 1.0 - progress * (1.0 - sched_cfg.min_lr / opt_cfg.lr)
            elif sched_cfg.schedule_type == "cosine":
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return sched_cfg.min_lr / opt_cfg.lr + (1.0 - sched_cfg.min_lr / opt_cfg.lr) * cosine_decay
            else:
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
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
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

def create_trainer(config: FractalTrainerConfig, **trainer_kwargs) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer.
    
    Args:
        config: Trainer configuration
        **trainer_kwargs: Additional arguments for pl.Trainer (overrides config values)
    
    Returns:
        PyTorch Lightning Trainer
    """
    # Default trainer arguments
    default_args = {
        'max_epochs': config.max_epochs,
        'accelerator': 'auto',
        'devices': 'auto',
        'strategy': 'ddp' if torch.cuda.device_count() > 1 else 'auto',
        'precision': '32',  # Use FP32 for stability
        'gradient_clip_val': 0,  # Handled in on_before_optimizer_step
        'accumulate_grad_batches': 8,
        'log_every_n_steps': 50,
        'val_check_interval': 1.0,
        'check_val_every_n_epoch': 1,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': False,
        'benchmark': True,
    }
    
    # Merge with user-provided arguments
    trainer_args = {**default_args, **trainer_kwargs}
    
    # Disable distributed sampler (we use custom BucketBatchSampler)
    trainer_args['use_distributed_sampler'] = False
    
    # Create trainer
    trainer = pl.Trainer(**trainer_args)
    
    return trainer
