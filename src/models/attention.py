"""
Attention modules for FractalMIDI models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from .utils import scaled_dot_product_attention


class Attention(nn.Module):
    """
    Standard bidirectional attention module with FlashAttention-2 support.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Optional scale factor for QK
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        # qkv: (B, N, 3*C) -> reshape -> (B, N, 3, H, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        
        # Try FlashAttention-2
        use_flash = FLASH_ATTN_AVAILABLE and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]
        
        if use_flash:
            # flash_attn_qkvpacked_func expects (B, N, 3, H, D)
            x = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal
            )
            # Output: (B, N, H, D)
            x = x.reshape(B, N, C)
        else:
            # Fallback to manual or SDPA
            # Permute for manual/SDPA: (3, B, H, N, D) or similar
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0) # (B, H, N, D)

            # Use PyTorch SDPA if available (it handles memory better than manual)
            if hasattr(F, 'scaled_dot_product_attention'):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    scale=self.scale,
                    is_causal=is_causal
                )
                x = x.transpose(1, 2).reshape(B, N, C)
            else:
                # Legacy manual implementation (uses custom scaled_dot_product_attention from utils)
                # Reshape for custom util: (B, H, N, D) -> (B*H, N, D) ?? No, util expects (..., L, D)
                # But util implementation: L, S = query.size(-2), key.size(-2)
                # It supports batch dimensions.
                
                # Use the utility function which supports causal/mask
                x = scaled_dot_product_attention(
                     q, k, v,
                     attn_mask=attn_mask,
                     dropout_p=self.attn_drop.p if self.training else 0.0,
                     is_causal=is_causal,
                     scale=self.scale
                )
                x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CausalAttention(nn.Module):
    """
    Causal (autoregressive) attention module with KV Cache support.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_norm: Whether to normalize Q and K
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
        norm_layer: Normalization layer class
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # KV Cache: will be set externally when needed
        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False, cache_position=None):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor (B, N, C)
            use_cache: If True, use and update KV cache
            cache_position: Position in cache to update (int), used when use_cache=True
        
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        if use_cache and self.k_cache is not None and self.v_cache is not None:
            # Update cache at specified position
            if cache_position is not None:
                # Single position update (for autoregressive generation)
                self.k_cache[:, :, cache_position:cache_position+N, :] = k
                self.v_cache[:, :, cache_position:cache_position+N, :] = v
                # Use full cache for attention
                k_full = self.k_cache[:, :, :cache_position+N, :]
                v_full = self.v_cache[:, :, :cache_position+N, :]
            else:
                # Full sequence update
                self.k_cache[:, :, :N, :] = k
                self.v_cache[:, :, :N, :] = v
                k_full = k
                v_full = v
        else:
            # No cache, use current k and v
            k_full = k
            v_full = v

        # Compute attention
        x = scaled_dot_product_attention(
            q, k_full, v_full,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

