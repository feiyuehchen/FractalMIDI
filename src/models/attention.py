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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if position_ids is not None:
        cos = cos[position_ids].squeeze(1)
        sin = sin[position_ids].squeeze(1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        # cos, sin are (1, 1, seq_len, dim) or (1, seq_len, dim)
        # q, k are (B, H, N, D)
        # We need to broadcast. 
        # Usually cos/sin from RotaryEmbedding are (seq_len, dim).
        # We might need to unsqueeze to (1, 1, seq_len, dim).
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
        if sin.dim() == 2:
            sin = sin.unsqueeze(0).unsqueeze(0)
            
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

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
    Causal (autoregressive) attention module with KV Cache support and RoPE.
    
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
        
    def forward(self, x, use_cache=False, past_key_values=None, rotary_emb=None):
        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor (B, N, C)
            use_cache: If True, return updated key/values
            past_key_values: Tuple of (k_cache, v_cache) from previous step
            rotary_emb: Tuple of (cos, sin) for RoPE
        
        Returns:
            x: Output tensor (B, N, C)
            present_key_values: Tuple of (k, v) to be cached
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if provided
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k_past, v_past = past_key_values
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
            
        present_key_values = (k, v) if use_cache else None

        # Compute attention
        # Note: scaled_dot_product_attention handles causal mask internally if is_causal=True
        # But if we are using KV cache, we are processing step by step (N=1), and we attend to all previous (cached) keys.
        # In that case, we don't strictly need a causal mask matrix, but we need to ensure we attend to everything up to now.
        # If N > 1 (prefill), we need causal mask.
        
        # When using past_key_values (inference), N is usually 1, and we want to attend to all L = past + 1 tokens.
        # is_causal=True in SDPA creates a mask (L, L). If we pass q(1, H, 1, D) and k(1, H, L, D), 
        # SDPA expects causal mask to align.
        # Actually SDPA with is_causal=True handles this correctly even for N=1 if dimensions match logic?
        # No, typically for decoding we might manually handle it or rely on SDPA being smart.
        # Safest is to disable is_causal for N=1 inference (since we attend to everything available which is "past"),
        # and enable it for training (N=L).
        
        is_training_or_prefill = (past_key_values is None) and (N > 1)
        
        x = scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=is_training_or_prefill
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if use_cache:
            return x, present_key_values
        return x

class RelativeGlobalAttention(nn.Module):
    """
    Relative Global Attention (Music Transformer style).
    Includes 'skewing' mechanism for efficient relative position computation.
    """
    def __init__(self, dim, num_heads=8, max_rel_dist=1024, attn_drop=0., proj_drop=0., qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_rel_dist = max_rel_dist

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative Position Embeddings
        # We need embeddings for distances: -max_dist ... 0 ... +max_dist (if bidirectional)
        # Or just 0 ... -max_dist for causal? Music Transformer usually considers backward distances.
        # We'll use a vocabulary of size max_rel_dist.
        # E_r in paper.
        self.Er = nn.Parameter(torch.randn(max_rel_dist, self.head_dim))

    def _relative_skew(self, S_rel):
        # S_rel: (B, H, L, L)
        # Pad column of zeros on the left
        B, H, L, _ = S_rel.shape
        pad = torch.zeros(B, H, L, 1, device=S_rel.device, dtype=S_rel.dtype)
        S_rel = torch.cat([pad, S_rel], dim=-1) # (B, H, L, L+1)
        
        # Reshape to (B, H, L+1, L)
        S_rel = S_rel.view(B, H, L+1, L)
        
        # Slice to remove first row (which was the padded zeros in shifted view)
        # Actually we want top-left to be distance 0.
        # The skewing trick shifts indices so that (i, j) gets embedding for i-j.
        # We take (B, H, L, L) from the view.
        S_rel = S_rel[:, :, 1:, :] 
        return S_rel

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # (B, H, N, D)
        
        # Standard content-content attention
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        content_score = torch.matmul(q, k.transpose(-1, -2))
        
        # Content-Position attention
        # Q * Er^T
        # We need relative embeddings for relevant distances.
        # Music Transformer (Huang 2018): S_rel = skew(Q * E_r^T)
        # We assume self.Er covers the length N.
        # If N > max_rel_dist, we clamp or window? For now assume N <= max_rel_dist.
        
        # We need E_r of shape (N, D).
        # We take first N embeddings (representing dist 0 to -(N-1))
        # This assumes causal relative distances.
        er_seq = self.Er[:N, :] # (N, D)
        
        # Calculate Q * Er^T
        # q: (B, H, N, D)
        # er_seq: (N, D) -> (1, 1, N, D) for broadcasting?
        # We want (B, H, N, N).
        # We can compute Q * Er^T -> (B, H, N, N) where last N is the relative position index
        rel_score = torch.matmul(q, er_seq.transpose(0, 1)) # (B, H, N, N)
        
        # Skewing
        rel_score = self._relative_skew(rel_score)
        
        # Combine
        attn_score = (content_score + rel_score) * self.scale
        
        # Masking
        if is_causal:
            # Create causal mask
            mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
            attn_score.masked_fill_(mask, float('-inf'))
            
        if attn_mask is not None:
            # attn_mask: (B, N) or (B, N, N)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
            attn_score.masked_fill_(~attn_mask, float('-inf'))
            
        attn_probs = F.softmax(attn_score, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        
        x = torch.matmul(attn_probs, v) # (B, H, N, D)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
