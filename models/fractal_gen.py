"""
Hierarchical FractalGen model for piano roll generation.
"""

import torch
import torch.nn as nn

from .mar_generator import PianoRollMAR
from .ar_generator import PianoRollAR
from .velocity_loss import PianoRollVelocityLoss


class PianoRollFractalGen(nn.Module):
    """
    Hierarchical Fractal Generative Model for Piano Rolls.
    Adapted from FractalGen with recursive structure.
    """
    def __init__(self,
                 img_size_list,
                 embed_dim_list,
                 num_blocks_list,
                 num_heads_list,
                 generator_type_list,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 guiding_pixel=False,
                 num_conds=5,
                 v_weight=1.0,
                 grad_checkpointing=False,
                 fractal_level=0,
                 scan_order='row_major',
                 mask_ratio_loc=1.0,
                 mask_ratio_scale=0.5):
        super().__init__()

        # Fractal specifics
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(img_size_list)

        # Dummy condition embedding (no class labels for piano rolls)
        if self.fractal_level == 0:
            self.dummy_cond = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
            torch.nn.init.normal_(self.dummy_cond, std=0.02)

        # Generator for current level
        if generator_type_list[fractal_level] == "mar":
            generator = PianoRollMAR
        elif generator_type_list[fractal_level] == "ar":
            generator = PianoRollAR
        else:
            raise NotImplementedError
        
        # Calculate seq_len based on patch arrangement
        # For piano roll: height is fixed at 128, width is variable
        h_patches = 128 // img_size_list[fractal_level+1]
        # Assume maximum width of 256 for seq_len calculation
        max_w_patches = 256 // img_size_list[fractal_level+1]
        expected_seq_len = h_patches * max_w_patches
        
        self.img_size = img_size_list[fractal_level]  # Store img_size for this level
        
        # Prepare generator kwargs
        generator_kwargs = dict(
            seq_len=(img_size_list[fractal_level] // img_size_list[fractal_level+1]) ** 2,  # Nominal seq_len
            patch_size=img_size_list[fractal_level+1],
            cond_embed_dim=embed_dim_list[fractal_level-1] if fractal_level > 0 else embed_dim_list[0],
            embed_dim=embed_dim_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=guiding_pixel if fractal_level > 0 else False,
            num_conds=num_conds,
            grad_checkpointing=grad_checkpointing,
            max_seq_len=expected_seq_len,  # Support longer sequences
            img_size=img_size_list[fractal_level],  # Pass img_size to generator
        )
        
        # Add scan_order for AR generators
        if generator_type_list[fractal_level] == "ar":
            generator_kwargs['scan_order'] = scan_order
        
        # Add mask_ratio parameters for MAR generators
        if generator_type_list[fractal_level] == "mar":
            generator_kwargs['mask_ratio_loc'] = mask_ratio_loc
            generator_kwargs['mask_ratio_scale'] = mask_ratio_scale
        
        self.generator = generator(**generator_kwargs)

        # Build next fractal level recursively
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = PianoRollFractalGen(
                img_size_list=img_size_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                attn_dropout=attn_dropout,
                proj_dropout=proj_dropout,
                guiding_pixel=guiding_pixel,
                num_conds=num_conds,
                v_weight=v_weight,
                grad_checkpointing=grad_checkpointing,
                fractal_level=fractal_level+1,
                scan_order=scan_order,
                mask_ratio_loc=mask_ratio_loc,
                mask_ratio_scale=mask_ratio_scale
            )
        else:
            # Final level uses velocity loss
            self.next_fractal = PianoRollVelocityLoss(
                c_channels=embed_dim_list[fractal_level],
                depth=num_blocks_list[fractal_level+1],
                width=embed_dim_list[fractal_level+1],
                num_heads=num_heads_list[fractal_level+1],
                v_weight=v_weight,
                level_index=fractal_level + 1,
            )

    def forward(self, imgs, cond_list=None):
        """Forward pass to compute loss recursively."""
        if self.fractal_level == 0:
            # Use dummy condition (no class labels)
            dummy_embedding = self.dummy_cond.expand(imgs.size(0), -1)
            cond_list = [dummy_embedding for _ in range(5)]

        # Get patches and conditions for next level
        imgs, cond_list, guiding_pixel_loss, level_stats = self.generator(imgs, cond_list)
        
        # Compute loss recursively and gather statistics from deeper levels
        child_loss, child_stats = self.next_fractal(imgs, cond_list)
        total_loss = child_loss + guiding_pixel_loss

        stats = {}
        if child_stats is not None:
            stats.update(child_stats)

        level_prefix = f'level_{self.fractal_level}'
        if level_stats is not None:
            for key, value in level_stats.items():
                stats[f'{level_prefix}/{key}'] = value

        stats[f'{level_prefix}/loss_contribution'] = child_loss.detach()
        stats[f'{level_prefix}/total_loss'] = total_loss.detach()

        return total_loss, stats

    def sample(self, batch_size, cond_list=None, num_iter_list=None, cfg=1.0, cfg_schedule="constant",
               temperature=1.0, filter_threshold=0, fractal_level=0, visualize=False, return_intermediates=False,
               _intermediates_list=None, _patch_pos=None):
        """Generate samples recursively, optionally returning intermediate outputs from all levels."""
        if cond_list is None:
            # Use dummy condition
            dummy_embedding = self.dummy_cond.expand(batch_size, -1)
            cond_list = [dummy_embedding for _ in range(5)]

        # Initialize intermediates with accumulation canvas at top level
        if return_intermediates and fractal_level == 0 and _intermediates_list is None:
            _intermediates_list = {
                'frames': [],  # List of intermediate frames
                # Initialize with -1 (white/silence) instead of 0
                'canvas': torch.full((batch_size, 1, 128, 256), -1.0)  # Accumulation canvas (128x256 for piano roll)
            }

        if fractal_level < self.num_fractal_levels - 2:
            # For next fractal level, wrap the sample call with position tracking
            def next_level_sample_function(cond_list, cfg, temperature, filter_threshold, patch_pos=None):
                result = self.next_fractal.sample(
                    batch_size=cond_list[0].size(0) if cfg == 1.0 else cond_list[0].size(0) // 2,
                    cond_list=cond_list,
                    num_iter_list=num_iter_list,
                    cfg=cfg,
                    cfg_schedule="constant",
                    temperature=temperature,
                    filter_threshold=filter_threshold,
                    fractal_level=fractal_level + 1,
                    return_intermediates=return_intermediates,
                    _intermediates_list=_intermediates_list,
                    _patch_pos=patch_pos  # Pass position down
                )
                return result
        else:
            # For velocity loss level (Level 3), wrap to pass intermediates and position
            def next_level_sample_function(cond_list, cfg, temperature, filter_threshold, patch_pos=None):
                result = self.next_fractal.sample(
                    cond_list=cond_list,
                    temperature=temperature,
                    cfg=cfg,
                    filter_threshold=filter_threshold,
                    _intermediates_list=_intermediates_list if return_intermediates else None,
                    _current_level=fractal_level + 1 if return_intermediates else None,
                    _patch_pos=patch_pos
                )
                # If recording and position is provided, update canvas with 1x1 result
                if _intermediates_list is not None and patch_pos is not None:
                    h_start, w_start, h_size, w_size = patch_pos
                    # Upsample 1x1 to patch size
                    vel_img = result.reshape(result.size(0), 1, 1, 1)
                    vel_upsampled = torch.nn.functional.interpolate(
                        vel_img, size=(h_size, w_size), mode='nearest'
                    )
                    # Update canvas
                    _intermediates_list['canvas'][:result.size(0), :, h_start:h_start+h_size, w_start:w_start+w_size] = vel_upsampled
                    # Don't record frame here - will be recorded by parent level
                return result

        # Recursively sample with intermediate recording
        result = self.generator.sample(
            cond_list, num_iter_list[fractal_level], cfg, cfg_schedule,
            temperature, filter_threshold, next_level_sample_function, visualize,
            _intermediates_list=_intermediates_list if return_intermediates else None,
            _current_level=fractal_level,  # Always pass level for correct shape calculation
            _patch_pos=_patch_pos
        )
        
        # Return with intermediates at top level
        if return_intermediates and fractal_level == 0:
            return result, _intermediates_list['frames']
        
        return result


# ==============================================================================
# Model Factory Functions
# ==============================================================================

def fractalmar_piano(**kwargs):
    """
    FractalGen model for piano roll generation (single configuration).
    Four-level hierarchy: 128 → 16 → 4 → 1.
    """
    generator_type_list = kwargs.pop("generator_type_list", ("mar", "mar", "mar", "mar"))
    if len(generator_type_list) != 4:
        raise ValueError("generator_type_list must contain exactly 4 entries")
    for idx, g in enumerate(generator_type_list):
        if g not in {"mar", "ar"}:
            raise ValueError(f"generator_type_list[{idx}] must be 'mar' or 'ar', got '{g}'")

    return PianoRollFractalGen(
        img_size_list=(128, 16, 4, 1),
        embed_dim_list=(512, 256, 128, 64),
        num_blocks_list=(12, 3, 2, 1),
        num_heads_list=(8, 4, 2, 2),
        generator_type_list=generator_type_list,
        fractal_level=0,
        **kwargs
    )

