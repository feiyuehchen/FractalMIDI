import torch
import torch.nn as nn
import math

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Conditioning vector modulates the scale and shift of LayerNorm.
    """
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(cond_dim, embed_dim * 2)
        
        # Initialize projection to be identity-like
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias[:embed_dim].fill_(1.0)
            self.proj.bias[embed_dim:].fill_(0.0)

    def forward(self, x, cond):
        # x: (B, T, E)
        # cond: (B, C) or (B, 1, C)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1) # (B, 1, C)
            
        scale_shift = self.proj(cond) # (B, 1, 2*E)
        scale, shift = scale_shift.chunk(2, dim=-1) # (B, 1, E)
        
        x = self.norm(x)
        x = x * scale + shift
        return x

class AdaLNModulator(nn.Module):
    """
    AdaLN Modulator only (Scale & Shift).
    Assumes input is already normalized.
    """
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, embed_dim * 2)
        # Zero-init to start as Identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias) 
        
    def forward(self, x, cond):
        # x: (B, T, E)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
            
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Modulation: x * (1 + scale) + shift
        x = x * (1 + scale) + shift
        return x

class TimePositionalEmbedding(nn.Module):
    """
    Sinusoidal Time Positional Embedding.
    """
    def __init__(self, embed_dim, max_len=16384):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, E)
        T = x.size(1)
        return self.pe[:T, :].unsqueeze(0) # (1, T, E)

class FractalInputProj(nn.Module):
    """
    Project Piano Roll (B, 2, T, 128) -> (B, T, E).
    Uses Conv2d over Pitch axis to capture harmonic relationships (intervals) invariantly.
    Includes Bar Position Embedding.
    """
    def __init__(self, in_channels, embed_dim, max_bar_len=16):
        super().__init__()
        # Input: (B, 2, T, 128)
        
        self.pitch_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(12, 1), stride=(1, 1), padding=(6, 0)),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(12, 1), stride=(1, 1), padding=(5, 0)), 
            nn.GELU(),
        )
        
        self.proj = nn.Conv1d(32 * 128, embed_dim, kernel_size=1)
        
        self.pos_emb = TimePositionalEmbedding(embed_dim)
        self.bar_emb = nn.Embedding(max_bar_len, embed_dim)

    def forward(self, x, bar_pos=None):
        x = x.permute(0, 1, 3, 2) 
        x = self.pitch_conv(x) # (B, 32, 128, T)
        B, C, P, T = x.shape
        x = x.reshape(B, C * P, T)
        x = self.proj(x) # (B, E, T)
        x = x.permute(0, 2, 1) # (B, T, E)
        
        x = x + self.pos_emb(x)
        
        if bar_pos is not None:
            x = x + self.bar_emb(bar_pos)
            
        return x

class FractalOutputProj(nn.Module):
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels * 128)

    def forward(self, x):
        B, T, E = x.shape
        x = self.proj(x) 
        x = x.view(B, T, 2, 128)
        x = x.permute(0, 2, 1, 3) 
        return x

class StructureInputProj(nn.Module):
    def __init__(self, in_channels, embed_dim, max_bar_len=16):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.pos_emb = TimePositionalEmbedding(embed_dim)
        self.bar_emb = nn.Embedding(max_bar_len, embed_dim)

    def forward(self, x, bar_pos=None):
        x = x.permute(0, 2, 1) # (B, T, 2)
        x = self.proj(x) # (B, T, E)
        x = x + self.pos_emb(x)
        
        if bar_pos is not None:
            x = x + self.bar_emb(bar_pos)
            
        return x

class StructureOutputProj(nn.Module):
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        x = self.proj(x) 
        x = x.permute(0, 2, 1) 
        return x

