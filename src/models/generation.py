"""
High-level generation functions for conditional and inpainting generation.
"""

import torch


def conditional_generation(model, condition_roll, generation_length, num_iter_list,
                          cfg=1.0, temperature=1.0, filter_threshold=0,
                          return_intermediates=False, _intermediates_list=None):
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
        return_intermediates: Whether to return intermediate steps
        _intermediates_list: Optional container for streaming intermediates
    
    Returns:
        Generated piano roll (1, piano_roll_height, cond_length + generation_length)
        Or (generated, intermediates) if return_intermediates=True
    """
    device = condition_roll.device
    piano_roll_height = model.config.piano_roll.height
    patch_size = model.config.piano_roll.patch_size
    
    cond_length = condition_roll.shape[2]
    total_length = cond_length + generation_length
    
    # Pad to patch size
    padded_length = ((total_length + patch_size - 1) // patch_size) * patch_size
    
    # Create full piano roll with condition
    # Initialize with -1 (mask token) for regions to be generated
    full_roll = torch.full((1, 1, piano_roll_height, padded_length), fill_value=-1.0, device=device)
    full_roll[:, :, :, :cond_length] = condition_roll
    
    # Generate using standard sampling
    # Pass target_width to ensure correct size
    generated = model.sample(
        batch_size=1,
        cond_list=None,
        num_iter_list=num_iter_list,
        cfg=cfg,
        cfg_schedule="constant",
        temperature=temperature,
        filter_threshold=filter_threshold,
        fractal_level=0,
        target_width=padded_length,
        return_intermediates=return_intermediates,
        _intermediates_list=_intermediates_list
    )
    
    intermediates = None
    if isinstance(generated, tuple):
        generated, intermediates = generated
    
    # Restore condition in generated output (Naive conditioning)
    # This ensures condition is preserved exactly
    generated[:, :, :, :cond_length] = condition_roll
    
    # Trim to target length
    generated = generated[:, :, :, :total_length]
    
    result = generated.squeeze(1)
    
    if return_intermediates:
        return result, intermediates
    return result


@torch.no_grad()
def inpainting_generation(model, piano_roll, mask_start=None, mask_end=None, num_iter_list=None,
                         cfg=1.0, temperature=1.0, filter_threshold=0, 
                         mask=None, return_intermediates=False, _intermediates_list=None):
    """
    Generate piano roll with inpainting (fill masked region).
    Supports either (mask_start, mask_end) or arbitrary mask tensor.
    
    Args:
        model: FractalGen model
        piano_roll: Input piano roll (1, piano_roll_height, duration)
        mask_start: Start of mask region (in time steps)
        mask_end: End of mask region (in time steps)
        num_iter_list: Iteration list for each level
        cfg: Classifier-free guidance strength
        temperature: Sampling temperature
        filter_threshold: Filter threshold for low probability tokens
        mask: Boolean mask tensor (1, 1, H, W) where True = Generate, False = Keep
        return_intermediates: Whether to return intermediate steps
        _intermediates_list: Optional container for streaming intermediates
    
    Returns:
        Inpainted piano roll (1, piano_roll_height, duration)
    """
    device = piano_roll.device
    piano_roll_height = model.config.piano_roll.height
    patch_size = model.config.piano_roll.patch_size
    
    # Handle input shape
    if piano_roll.dim() == 3:
        piano_roll = piano_roll.unsqueeze(1) # (1, 1, H, W)
        
    duration = piano_roll.shape[3]
    
    # Pad to patch size
    padded_duration = ((duration + patch_size - 1) // patch_size) * patch_size
    
    # Create padded version of input
    padded_roll = torch.full((1, 1, piano_roll_height, padded_duration), fill_value=-1.0, device=device)
    padded_roll[:, :, :, :duration] = piano_roll
    
    # Prepare Boolean Mask (True = Generate/Masked, False = Keep)
    if mask is None:
        if mask_start is None or mask_end is None:
             # If no mask provided, treat as unconditional (generate all)
             mask_bool = torch.ones((1, 1, piano_roll_height, padded_duration), dtype=torch.bool, device=device)
             if mask_start is not None and mask_end is not None:
                 # Reset to zeros then set region
                 mask_bool.fill_(False)
                 mask_bool[:, :, :, mask_start:mask_end] = True
        else:
             mask_bool = torch.zeros((1, 1, piano_roll_height, padded_duration), dtype=torch.bool, device=device)
             mask_bool[:, :, :, mask_start:mask_end] = True
    else:
        # Handle provided mask
        if mask.dim() == 3: mask = mask.unsqueeze(1)
        mask_bool = mask.bool().to(device)
        # Pad mask if needed
        if mask_bool.shape[3] < padded_duration:
             pad_m = torch.zeros((1, 1, piano_roll_height, padded_duration), dtype=torch.bool, device=device)
             pad_m[:, :, :, :mask.shape[3]] = mask
             mask_bool = pad_m
        elif mask_bool.shape[3] > padded_duration:
             mask_bool = mask_bool[:, :, :, :padded_duration]

    # Construct Inpainting Inputs for Level 0
    # Patchify mask to Level 0 granularity (16x16)
    # If ANY pixel in a patch is masked (True), the whole patch is masked (True).
    l0_patch_size = model.generator.patch_size
    mask_pooled = torch.nn.functional.max_pool2d(
        mask_bool.float(), 
        kernel_size=l0_patch_size, 
        stride=l0_patch_size
    )
    inpainting_mask = mask_pooled.flatten(1) # (B, seq_len)
    
    # Patchify target (padded_roll) to Level 0 patches
    inpainting_target = model.generator.patchify(padded_roll)

    # Generate with inpainting constraints
    generated = model.sample(
        batch_size=1,
        cond_list=None,
        num_iter_list=num_iter_list,
        cfg=cfg,
        cfg_schedule="constant",
        temperature=temperature,
        filter_threshold=filter_threshold,
        fractal_level=0,
        target_width=padded_duration,
        return_intermediates=return_intermediates,
        inpainting_mask=inpainting_mask,
        inpainting_target=inpainting_target,
        _intermediates_list=_intermediates_list
    )
    
    intermediates = None
    if isinstance(generated, tuple):
        generated, intermediates = generated
    
    # Ensure generated shape matches padding
    if generated.shape[3] > padded_duration:
        generated = generated[:, :, :, :padded_duration]
    
    # Strict Mix: Enforce known pixels exactly
    # Use generated where mask is True (Unknown), else use original (padded_roll)
    result = torch.where(mask_bool, generated, padded_roll)
    
    # Trim to original duration
    result = result[:, :, :, :duration]
    
    if return_intermediates:
        return result.squeeze(1), intermediates
    return result.squeeze(1)
