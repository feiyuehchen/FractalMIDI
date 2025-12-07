"""
Utility functions for FractalMIDI models.
"""

import math
import torch


def mask_by_order(mask_len, order, bsz, seq_len):
    """
    Create mask based on ordering.
    
    Args:
        mask_len: Length of mask
        order: Ordering tensor
        bsz: Batch size
        seq_len: Sequence length
    
    Returns:
        Boolean mask tensor
    """
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(
        masking, dim=-1, index=order[:, :mask_len.long()],
        src=torch.ones(bsz, seq_len, device=order.device)
    ).bool()
    return masking


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """
    Scaled dot product attention with causal masking support.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor
    
    Returns:
        Attention output tensor
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

