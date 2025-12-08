import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import CausalAttention, RotaryEmbedding, apply_rotary_pos_emb, Attention

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
        if cond is None:
            return self.norm(x)
            
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
        # cond: (B, C) or (B, 1, C) or (B, T, C)
        if cond is None:
            return x
            
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        elif cond.dim() == 3 and cond.shape[1] != x.shape[1] and cond.shape[1] == 1:
            # If cond is (B, 1, C), it will broadcast correctly
            pass
        elif cond.dim() == 3 and cond.shape[1] != x.shape[1]:
            # If cond has time dimension but it doesn't match x, we might need to be careful.
            # However, usually AdaLN expects cond to be global or strictly matching.
            # If it's mismatching T, it will crash here or broadcast weirdly.
            # Assuming cond is properly prepared before calling this if it's temporal.
            pass
            
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Modulation: x * (1 + scale) + shift
        x = x * (1 + scale) + shift
        return x

class TimePositionalEmbedding(nn.Module):
    """
    Multi-Scale Absolute Temporal Embedding.
    Captures: Bar (16), Quarter Note (4), 8-Bar (128), and Global structure.
    """
    def __init__(self, embed_dim, max_len=16384):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-scale embeddings
        self.bar_emb = nn.Embedding(16, embed_dim)      # t % 16
        self.qn_emb = nn.Embedding(4, embed_dim)        # t % 4
        self.bar8_emb = nn.Embedding(128, embed_dim)    # t % 128
        self.global_emb = nn.Embedding(max_len, embed_dim) # t
        
        # Init weights (optional, but good practice)
        nn.init.normal_(self.bar_emb.weight, std=0.02)
        nn.init.normal_(self.qn_emb.weight, std=0.02)
        nn.init.normal_(self.bar8_emb.weight, std=0.02)
        nn.init.normal_(self.global_emb.weight, std=0.02)

    def forward(self, x):
        # x: (B, T, E)
        T = x.size(1)
        device = x.device
        
        t = torch.arange(T, device=device)
        
        # Calculate indices
        idx_bar = t % 16
        idx_qn = t % 4
        idx_8bar = t % 128
        
        # Clamp global to max_len
        max_g = self.global_emb.num_embeddings - 1
        idx_global = torch.clamp(t, 0, max_g)
        
        # Sum embeddings
        pe = (self.bar_emb(idx_bar) + 
              self.qn_emb(idx_qn) + 
              self.bar8_emb(idx_8bar) + 
              self.global_emb(idx_global)) # (T, E)
              
        return pe.unsqueeze(0) # (1, T, E)

class FractalInputProj(nn.Module):
    """
    Project Piano Roll (B, 2, T, 128) -> (B, T, E).
    Uses ConvNeXt V2 Block backbone for robust feature extraction.
    Then uses GPT-OSS style attention to mix features.
    Includes Multi-Scale Time Positional Embedding.
    """
    def __init__(self, in_channels, embed_dim, max_bar_len=16):
        super().__init__()
        
        # 1. Stem: Conv2d over (Pitch, Time)
        # Input: (B, 2, T, 128) -> Permute to (B, 2, 128, T)
        # Revised Stem: Kernel (4, 1), Stride (4, 1) -> Reduces Pitch 128->32, Keeps Time T.
        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=(4, 1), stride=(4, 1))
        
        # 2. ConvNeXt V2 Backbone (over Pitch-Time map)
        # Input: (B, E, 32, T)
        self.conv_stages = nn.ModuleList([
            ConvNeXtV2Block(embed_dim),
            ConvNeXtV2Block(embed_dim)
        ])
        
        # 3. Flatten Pitch & Project
        # (B, E, 32, T) -> (B, E*32, T) -> (B, E, T)
        self.proj_flatten = nn.Conv1d(embed_dim * 32, embed_dim, kernel_size=1)
        
        # 4. Time Embedding
        self.pos_emb = TimePositionalEmbedding(embed_dim)
        self.bar_emb = nn.Embedding(max_bar_len, embed_dim)
        
        # 5. Attention (GPT-OSS style)
        # Process sequence length T
        self.attn_block = AttentionBlock(embed_dim, num_heads=8) # Simplified
        
    def forward(self, x, bar_pos=None):
        # x: (B, 2, T, 128)
        x = x.permute(0, 1, 3, 2) # (B, 2, 128, T)
        
        # 1. Stem
        x = self.stem(x) # (B, E, 32, T)
        
        # 2. ConvNeXt
        for stage in self.conv_stages:
            x = stage(x)
            
        # 3. Flatten & Project
        B, C, P, T = x.shape
        x = x.reshape(B, C * P, T)
        x = self.proj_flatten(x) # (B, E, T)
        
        x = x.permute(0, 2, 1) # (B, T, E)
        
        # 4. Add Position Embeddings
        x = x + self.pos_emb(x)
        if bar_pos is not None:
            max_valid_idx = self.bar_emb.num_embeddings - 1
            bar_pos_clamped = torch.clamp(bar_pos, 0, max_valid_idx)
            x = x + self.bar_emb(bar_pos_clamped)
            
        # 5. Attention Block (Refining temporal context)
        x = self.attn_block(x)
        
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
            max_valid_idx = self.bar_emb.num_embeddings - 1
            bar_pos_clamped = torch.clamp(bar_pos, 0, max_valid_idx)
            x = x + self.bar_emb(bar_pos_clamped)
            
        return x

class StructureOutputProj(nn.Module):
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x, target=None, teacher_forcing=False):
        # target and teacher_forcing are ignored for structure level
        # but needed for compatibility with calling signature
        x = self.proj(x) 
        x = x.permute(0, 2, 1) 
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class GRN(nn.Module):
    """ Global Response Normalization layer """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXt V2 Block. """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

def swiglu(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x

class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = swiglu(x)
        x = self.c_proj(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.ln_1 = RMSNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads) # Using standard Attention
        self.ln_2 = RMSNorm(embed_dim)
        
        # Simple config object for MLP
        class Config: pass
        config = Config()
        config.hidden_size = embed_dim
        config.intermediate_size = 4 * embed_dim
        self.mlp = MLPBlock(config)

    def forward(self, x):
        # Attention part
        x = x + self.attn(self.ln_1(x))
        # MLP part
        x = x + self.mlp(self.ln_2(x))
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

class LSTMOutputProj(nn.Module):
    """
    LSTM-based Decoder. Kept for legacy compatibility but updated to output logits.
    """
    def __init__(self, embed_dim, out_channels=2, hidden_dim=1024, num_layers=2, num_vel_bins=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_vel_bins = num_vel_bins
        
        # Project Transformer embedding
        self.cond_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(out_channels, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output: 1 (Note) + num_vel_bins (Velocity bins)
        self.output_head = nn.Linear(hidden_dim, 1 + self.num_vel_bins)
        
        self.sos_token = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.normal_(self.sos_token, std=0.02)

    def forward(self, x, target=None, teacher_forcing=True):
        B, T, E = x.shape
        cond = self.cond_proj(x).reshape(B*T, 1, self.hidden_dim)
        
        if teacher_forcing and target is not None:
            target_seq = target.permute(0, 2, 3, 1).reshape(B*T, 128, self.out_channels)
            sos = self.sos_token.expand(B*T, 1, -1)
            lstm_input_seq = torch.cat([sos, target_seq[:, :-1, :]], dim=1)
            lstm_input_emb = self.input_proj(lstm_input_seq)
            
            cond_expanded = cond.expand(-1, 128, -1)
            full_input = torch.cat([lstm_input_emb, cond_expanded], dim=-1)
            
            output, _ = self.lstm(full_input)
            logits = self.output_head(output) # (B*T, 128, 1+num_vel_bins)
            
            logits = logits.view(B, T, 128, 1+self.num_vel_bins).permute(0, 3, 1, 2)
            return logits
            
        else:
            h_state = None 
            cond_step = cond
            curr_input = self.sos_token.expand(B*T, 1, -1)
            outputs = []
            
            for p in range(128):
                curr_emb = self.input_proj(curr_input)
                full_input = torch.cat([curr_emb, cond_step], dim=-1)
                out_step, h_state = self.lstm(full_input, h_state)
                logit_step = self.output_head(out_step)
                outputs.append(logit_step)
                
                # Sample
                pred_note = (torch.sigmoid(logit_step[:, :, 0]) > 0.5).float()
                vel_logits = logit_step[:, :, 1:]
                pred_vel_idx = torch.argmax(vel_logits, dim=-1)
                pred_vel = pred_vel_idx.float() / (self.num_vel_bins - 1.0) # Map to 0-1
                curr_input = torch.stack([pred_note, pred_vel], dim=-1)

            logits = torch.cat(outputs, dim=1)
            logits = logits.view(B, T, 128, 1 + self.num_vel_bins).permute(0, 3, 1, 2)
            return logits

    def sample(self, x, temperature=1.0):
        # ... (Similar implementation for sampling)
        return self.forward(x, teacher_forcing=False)

class ParallelOutputProj(nn.Module):
    """
    Parallel Output Projection (Like FractalGen's implicit assumption or simple VAE decoders).
    Predicts all 128 pitches simultaneously in O(1).
    
    Output shape: (B, 1 + num_vel_bins, T, 128)
    """
    def __init__(self, embed_dim, out_channels=2, num_vel_bins=32):
        super().__init__()
        self.num_vel_bins = num_vel_bins
        # Output: 1 Note Logit + num_vel_bins Velocity Bin Logits = (1 + num_vel_bins) channels per pitch
        self.out_dim_per_pitch = 1 + num_vel_bins
        
        # Directly project from Embedding to all Pitch logits
        # Input: (B, T, E) -> Output: (B, T, 128 * (1 + num_vel_bins))
        self.proj = nn.Linear(embed_dim, 128 * self.out_dim_per_pitch)

    def forward(self, x, target=None, teacher_forcing=False):
        # target and teacher_forcing are not needed for parallel generation
        B, T, E = x.shape
        
        # 1. Projection
        x = self.proj(x) # (B, T, 128 * out_dim_per_pitch)
        
        # 2. Reshape to (B, T, 128, out_dim_per_pitch)
        x = x.view(B, T, 128, self.out_dim_per_pitch)
        
        # 3. Permute to match expected format (B, Channels, T, Pitch)
        # Channels = 1 (Note) + num_vel_bins (Vel)
        x = x.permute(0, 3, 1, 2) # (B, out_dim_per_pitch, T, 128)
        
        return x

    def sample(self, x, temperature=1.0):
        # Parallel sampling, very fast
        logits = self.forward(x) # (B, 1+num_vel_bins, T, 128)
        
        # Separate Note and Velocity
        logits_note = logits[:, 0:1, :, :] # (B, 1, T, 128)
        logits_vel = logits[:, 1:, :, :]   # (B, num_vel_bins, T, 128)
        
        # Sample Note (Bernoulli)
        if temperature > 0:
            probs_note = torch.sigmoid(logits_note / temperature)
            sample_note = torch.bernoulli(probs_note)
        else:
            sample_note = (logits_note > 0).float()
            
        # Sample Velocity (Categorical)
        # Reshape: (B, num_vel_bins, T, 128) -> permute -> (B, T, 128, num_vel_bins)
        B, _, T, P = logits_vel.shape
        vel_permuted = logits_vel.permute(0, 2, 3, 1)
        
        if temperature > 0:
            probs_vel = F.softmax(vel_permuted / temperature, dim=-1) # (B, T, 128, num_vel_bins)
            # Flatten to sample
            probs_flat = probs_vel.reshape(-1, self.num_vel_bins)
            sample_vel_idx = torch.multinomial(probs_flat, 1).view(B, T, P)
        else:
            sample_vel_idx = torch.argmax(vel_permuted, dim=-1)
            
        sample_vel = sample_vel_idx.float() / (self.num_vel_bins - 1.0)
        sample_vel = sample_vel.unsqueeze(1) # (B, 1, T, 128)
        
        # Combine: (B, 2, T, 128)
        sample_out = torch.cat([sample_note, sample_vel], dim=1)
        
        return logits, sample_out

class PitchTransformerOutputProj(nn.Module):
    """
    Transformer-based Decoder for Pitch-wise Autoregression.
    Generates 128 pitches sequentially.
    Applies Rotary Positional Embedding (RoPE).
    Uses KV Cache for efficient inference.
    Outputs classification logits for velocity.
    """
    def __init__(self, embed_dim, out_channels=2, hidden_dim=128, num_layers=2, num_heads=1, num_vel_bins=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_vel_bins = num_vel_bins
        
        # Condition projection
        self.cond_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Input projection (Note, Vel) -> Hidden
        self.input_proj = nn.Linear(out_channels, hidden_dim)
        
        # SOS Token
        self.sos_token = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.normal_(self.sos_token, std=0.02)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'attn': CausalAttention(hidden_dim, num_heads=num_heads, qk_norm=True),
                'norm2': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
            })
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        # Output: 1 (Note Logit) + num_vel_bins (Velocity Logits)
        self.output_head = nn.Linear(hidden_dim, 1 + self.num_vel_bins)
        
        # RoPE
        head_dim = hidden_dim // num_heads
        self.rotary_emb = RotaryEmbedding(head_dim)

    def forward(self, x, target=None, teacher_forcing=True):
        B, T, E = x.shape
        cond = self.cond_proj(x).reshape(B*T, 1, self.hidden_dim)
        
        if teacher_forcing and target is not None:
            # Training
            target_seq = target.permute(0, 2, 3, 1).reshape(B*T, 128, self.out_channels)
            sos = self.sos_token.expand(B*T, 1, -1)
            inp_seq = torch.cat([sos, target_seq[:, :-1]], dim=1) # (B*T, 128, 2)
            
            x_emb = self.input_proj(inp_seq) + cond
            
            # RoPE
            rotary_pos_emb = self.rotary_emb(x_emb, seq_len=128)
            
            for block in self.blocks:
                residual = x_emb
                x_emb = block['norm1'](x_emb)
                x_emb = block['attn'](x_emb, rotary_emb=rotary_pos_emb)
                x_emb = residual + x_emb
                
                residual = x_emb
                x_emb = block['norm2'](x_emb)
                x_emb = block['mlp'](x_emb)
                x_emb = residual + x_emb
                
            x_emb = self.norm(x_emb)
            logits = self.output_head(x_emb) # (B*T, 128, 1+num_vel_bins)
            
            # (B, 1+num_vel_bins, T, 128)
            logits = logits.view(B, T, 128, 1 + self.num_vel_bins).permute(0, 3, 1, 2)
            return logits
            
        else:
            # Inference with KV Cache
            return self.sample(x, temperature=0.0) # Helper call if just forward with no target

    def sample(self, x, temperature=1.0):
        B, T, E = x.shape
        cond = self.cond_proj(x).reshape(B*T, 1, self.hidden_dim)
        
        curr_input = self.sos_token.expand(B*T, 1, -1)
        outputs = []
        samples = []
        
        # Init KV Cache
        past_key_values = [None] * self.num_layers
        
        # Precompute RoPE for max length 128
        rotary_pos_emb_full = self.rotary_emb(curr_input, seq_len=128) # Just to get cos/sin
        
        for p in range(128):
            curr_emb = self.input_proj(curr_input) + cond
            
            # RoPE for current position p
            # We need to slice rotary embeddings for position p
            # rotary_pos_emb returns (cos, sin) of shape (seq_len, dim)
            # We want position p.
            cos = rotary_pos_emb_full[0][p:p+1] # (1, D)
            sin = rotary_pos_emb_full[1][p:p+1] # (1, D)
            rotary_emb_step = (cos, sin)
            
            x_pass = curr_emb # (B*T, 1, H)
            
            for i, block in enumerate(self.blocks):
                residual = x_pass
                x_pass = block['norm1'](x_pass)
                
                # Attention with KV Cache
                x_pass, present_kv = block['attn'](
                    x_pass, 
                    use_cache=True, 
                    past_key_values=past_key_values[i], 
                    rotary_emb=rotary_emb_step
                )
                past_key_values[i] = present_kv
                
                x_pass = residual + x_pass
                
                residual = x_pass
                x_pass = block['norm2'](x_pass)
                x_pass = block['mlp'](x_pass)
                x_pass = residual + x_pass
            
            x_pass = self.norm(x_pass)
            logit_step = self.output_head(x_pass) # (B*T, 1, 1+num_vel_bins)
            outputs.append(logit_step)
            
            # Sample
            logits_note = logit_step[:, :, 0]
            logits_vel = logit_step[:, :, 1:]
            
            if temperature > 0:
                probs_note = torch.sigmoid(logits_note / temperature)
                pred_note = torch.bernoulli(probs_note)
                
                probs_vel = F.softmax(logits_vel / temperature, dim=-1)
                pred_vel_idx = torch.multinomial(probs_vel.squeeze(1), 1).float() # (B*T, 1)
            else:
                pred_note = (logits_note > 0).float()
                pred_vel_idx = torch.argmax(logits_vel, dim=-1).float()
                
            pred_vel = pred_vel_idx / (self.num_vel_bins - 1.0)
            
            # Prepare next input: (B*T, 1, 2)
            curr_sample = torch.stack([pred_note.squeeze(1), pred_vel.squeeze(-1) if pred_vel.dim() > 1 else pred_vel], dim=-1).unsqueeze(1)
            curr_input = curr_sample
            samples.append(curr_sample)
            
        logits = torch.cat(outputs, dim=1)
        logits = logits.view(B, T, 128, 1 + self.num_vel_bins).permute(0, 3, 1, 2)
        
        sampled_seq = torch.cat(samples, dim=1) # (B*T, 128, 2)
        sampled_seq = sampled_seq.view(B, T, 128, 2).permute(0, 3, 1, 2) # (B, 2, T, 128)
        
        return logits, sampled_seq
