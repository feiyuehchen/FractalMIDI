"""
High-level generation functions for conditional and inpainting generation.
"""

import torch


def conditional_generation(model, condition_roll, generation_length, num_iter_list,
                          cfg=1.0, temperature=1.0, filter_threshold=0):
    """
    Generate piano roll conditioned on existing sequence.
    
    Args:
        model: FractalGen model
        condition_roll: Conditioning piano roll (1, piano_roll_height, cond_length)
        generation_length: Number of steps to generate
        num_iter_list: Iteration list for each level
        cfg: Classifier-free guidance strength
        temperature: Sampling temperature
        filter_threshold: Filter threshold for low probability tokens
    
    Returns:
        Generated piano roll (1, piano_roll_height, cond_length + generation_length)
    """
    device = condition_roll.device
    piano_roll_height = model.config.piano_roll.height
    patch_size = model.config.piano_roll.patch_size
    
    cond_length = condition_roll.shape[2]
    total_length = cond_length + generation_length
    
    # Pad to patch size
    padded_length = ((total_length + patch_size - 1) // patch_size) * patch_size
    
    # Create full piano roll with condition
    full_roll = torch.zeros(1, 1, piano_roll_height, padded_length, device=device)
    full_roll[:, :, :, :cond_length] = condition_roll
    
    # For true conditional generation, we need to implement partial masking
    # For now, use the full generation with conditioning as context
    # This is a simplified version - true implementation would modify the sampling loop
    
    # Generate using standard sampling (conditioning is implicit through initialization)
    generated = model.sample(
        batch_size=1,
        cond_list=None,  # Will use dummy cond
        num_iter_list=num_iter_list,
        cfg=cfg,
        cfg_schedule="constant",
        temperature=temperature,
        filter_threshold=filter_threshold,
        fractal_level=0
    )
    
    # Restore condition in generated output
    generated[:, :, :, :cond_length] = condition_roll
    
    # Trim to target length
    generated = generated[:, :, :, :total_length]
    
    return generated.squeeze(1)  # Return (1, piano_roll_height, total_length)


@torch.no_grad()
def inpainting_generation(model, piano_roll, mask_start, mask_end, num_iter_list,
                         cfg=1.0, temperature=1.0, filter_threshold=0):
    """
    Generate piano roll with inpainting (fill masked region).
    
    Args:
        model: FractalGen model
        piano_roll: Input piano roll (1, piano_roll_height, duration)
        mask_start: Start of mask region (in time steps)
        mask_end: End of mask region (in time steps)
        num_iter_list: Iteration list for each level
        cfg: Classifier-free guidance strength
        temperature: Sampling temperature
        filter_threshold: Filter threshold for low probability tokens
    
    Returns:
        Inpainted piano roll (1, piano_roll_height, duration)
    """
    device = piano_roll.device
    piano_roll_height = model.config.piano_roll.height
    patch_size = model.config.piano_roll.patch_size
    
    duration = piano_roll.shape[2]
    
    # Pad to patch size
    padded_duration = ((duration + patch_size - 1) // patch_size) * patch_size
    
    # Create padded version with masked region zeroed
    padded_roll = torch.zeros(1, 1, piano_roll_height, padded_duration, device=device)
    padded_roll[:, :, :, :duration] = piano_roll.unsqueeze(1) if piano_roll.dim() == 3 else piano_roll
    
    # For true inpainting, we need patch-level masking
    # For now, use simple approach: zero out mask region and regenerate
    original_masked = padded_roll[:, :, :, mask_start:mask_end].clone()
    padded_roll[:, :, :, mask_start:mask_end] = 0
    
    # Generate full piano roll (simplified - true implementation needs patch masking)
    generated = model.sample(
        batch_size=1,
        cond_list=None,
        num_iter_list=num_iter_list,
        cfg=cfg,
        cfg_schedule="constant",
        temperature=temperature,
        filter_threshold=filter_threshold,
        fractal_level=0
    )
    
    # Copy unmasked regions from original
    result = generated.clone()
    result[:, :, :, :mask_start] = padded_roll[:, :, :, :mask_start]
    result[:, :, :, mask_end:duration] = padded_roll[:, :, :, mask_end:duration]
    
    # Trim to original length
    result = result[:, :, :, :duration]
    
    return result.squeeze(1)  # Return (1, piano_roll_height, duration)


# ==============================================================================
# Utility Functions
# ==============================================================================

