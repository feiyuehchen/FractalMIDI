"""
Transformer blocks for FractalMIDI models.
"""

import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp

from .attention import Attention, CausalAttention


class Block(nn.Module):
    """
    Standard Transformer block with bidirectional attention.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Optional scale factor for QK
        proj_drop: Projection dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
        act_layer: Activation layer class
        norm_layer: Normalization layer class
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 proj_drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CausalBlock(nn.Module):
    """
    Transformer block with causal attention and KV Cache support.
    
    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        qkv_bias: Whether to use bias in QKV projection
        proj_drop: Projection dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
        act_layer: Activation layer class
        norm_layer: Normalization layer class
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, use_cache=False, cache_position=None):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor
            use_cache: If True, use KV cache in attention
            cache_position: Position in cache to update
        """
        x = x + self.drop_path(self.attn(self.norm1(x), use_cache=use_cache, cache_position=cache_position))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

