import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.stats as stats
import numpy as np

from .attention import Attention, CausalAttention, RelativeGlobalAttention
from .components import (
    AdaLN, AdaLNModulator, 
    FractalInputProj, FractalOutputProj, 
    StructureInputProj, StructureOutputProj,
    PitchTransformerOutputProj, LSTMOutputProj, ParallelOutputProj
)

class FractalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=None, use_relative_attn=False):
        super().__init__()
        self.use_adaln = cond_dim is not None
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        if self.use_adaln:
            self.adaln1_mod = AdaLNModulator(embed_dim, cond_dim)
            self.adaln2_mod = AdaLNModulator(embed_dim, cond_dim)
        
        if use_relative_attn:
            # Issue 5: Music Transformer (Relative Attention)
            self.attn = RelativeGlobalAttention(embed_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        else:
            self.attn = Attention(embed_dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond=None, attn_mask=None, is_causal=False):
        # x: (B, T, E)
        residual = x
        x = self.norm1(x) 
        if self.use_adaln:
            x = self.adaln1_mod(x, cond) 
        x = self.attn(x, attn_mask=attn_mask, is_causal=is_causal)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        if self.use_adaln:
            x = self.adaln2_mod(x, cond)
        x = self.mlp(x)
        x = residual + x
            
        return x

class FractalLayer(nn.Module):
    """
    Single Level Generator (replaces TemporalGenerator).
    """
    def __init__(self, embed_dim, num_heads, num_blocks, cond_dim=None, input_channels=2, 
                 is_structure_level=False, max_bar_len=16, 
                 generator_type='mar', use_relative_attn=True, num_vel_bins=32,
                 pitch_generator_type='ar'):
        super().__init__()
        self.generator_type = generator_type
        self.is_structure_level = is_structure_level
        self.num_vel_bins = num_vel_bins
        
        if is_structure_level:
            self.input_proj = StructureInputProj(input_channels, embed_dim, max_bar_len=max_bar_len)
            self.output_proj = StructureOutputProj(embed_dim, input_channels)
        else:
            self.input_proj = FractalInputProj(input_channels, embed_dim, max_bar_len=max_bar_len)
            
            if pitch_generator_type == 'parallel':
                 self.output_proj = ParallelOutputProj(
                    embed_dim, input_channels, num_vel_bins=num_vel_bins
                )
            else:
                # Use updated PitchTransformerOutputProj (Issue 1 & 4) with configurable velocity bins
                self.output_proj = PitchTransformerOutputProj(
                    embed_dim, input_channels, hidden_dim=embed_dim, num_vel_bins=num_vel_bins
                )

        self.blocks = nn.ModuleList([
            FractalBlock(embed_dim, num_heads, cond_dim=cond_dim, use_relative_attn=use_relative_attn)
            for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(embed_dim) if cond_dim is None else AdaLN(embed_dim, cond_dim)
        
        if self.generator_type == 'mar':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None
            self.sos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.sos_token, std=0.02)

    def forward(self, x, cond=None, mask=None, prev_emb=None, bar_pos=None, is_causal=False, use_sos=False):
        x_emb = self.input_proj(x, bar_pos) # (B, T, E)
        
        if use_sos and self.generator_type == 'ar':
            B = x_emb.shape[0]
            sos = self.sos_token.expand(B, -1, -1)
            # Shift for AR: [SOS, x_0, ..., x_{T-2}] to predict [x_0, ..., x_{T-1}]
            # Assumes x matches target length T.
            x_emb = torch.cat([sos, x_emb[:, :-1]], dim=1)
        
        if self.generator_type == 'mar' and mask is not None:
            mask_expanded = mask.unsqueeze(-1) # (B, T, 1)
            x_emb = x_emb * (~mask_expanded) + self.mask_token * mask_expanded
            
        for block in self.blocks:
            x_emb = block(x_emb, cond, is_causal=is_causal)
            
        if isinstance(self.norm, AdaLN):
            x_emb = self.norm(x_emb, cond)
        else:
            x_emb = self.norm(x_emb)
            
        return x_emb

class FractalGen(nn.Module):
    """
    Recursive Fractal Generator Node.
    """
    def __init__(self, config, level_idx, parent=None, global_cond_dim=0):
        super().__init__()
        self.level_idx = level_idx
        self.config = config
        
        # Architecture params
        embed_dims = config.architecture.embed_dim_list
        num_heads = config.architecture.num_heads_list
        num_blocks = config.architecture.num_blocks_list
        generator_types = config.generator.generator_type_list
        
        self.embed_dim = embed_dims[level_idx]
        self.generator_type = generator_types[level_idx]
        self.num_vel_bins = config.piano_roll.num_velocity_bins
        
        # Read pitch generator type from config, default to 'ar'
        # Use getattr with explicit fallback for safety
        self.pitch_generator_type = getattr(config.generator, 'pitch_generator_type', 'ar')
        
        if level_idx == 0:
            print(f"FractalGen (Level {level_idx}): Pitch Generator Type = {self.pitch_generator_type}")
        
        # Downsample Logic
        num_levels = len(embed_dims)
        if num_levels == 3:
            factors = [16, 4, 1]
        elif num_levels == 4:
            factors = [16, 8, 4, 1]
        else:
            factors = [2**(num_levels-1-i) for i in range(num_levels)]
        self.downsample_factor = factors[level_idx]

        # Conditioning (Issue 7)
        self.cond_dim = 0
        self.parent_proj = None
        self.global_proj = None
        
        compressed_dim = config.architecture.compressed_dim
        
        if level_idx > 0:
            # Project parent embedding
            self.parent_proj = nn.Sequential(
                nn.Linear(embed_dims[level_idx-1], compressed_dim),
                nn.Conv1d(compressed_dim, compressed_dim, kernel_size=3, padding=1), # Windowed context (Issue 7)
                nn.GELU()
            )
            self.cond_dim += compressed_dim
            
        if global_cond_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(global_cond_dim, compressed_dim),
                nn.GELU()
            )
            self.cond_dim += compressed_dim
            
        if self.cond_dim == 0:
            self.cond_dim = None 
            
        # Layer
        self.layer = FractalLayer(
            embed_dim=self.embed_dim,
            num_heads=num_heads[level_idx],
            num_blocks=num_blocks[level_idx],
            cond_dim=self.cond_dim,
            input_channels=2,
            is_structure_level=(level_idx == 0),
            max_bar_len=config.architecture.max_bar_len,
            generator_type=self.generator_type,
            use_relative_attn=True, # Issue 5
            num_vel_bins=self.num_vel_bins,
            pitch_generator_type=self.pitch_generator_type
        )
        
        # Recursion
        if level_idx < num_levels - 1:
            self.next_level = FractalGen(config, level_idx + 1, parent=self, global_cond_dim=global_cond_dim)
        else:
            self.next_level = None

        # MAR mask ratio generator (truncated normal, biased towards high mask ratio)
        if self.generator_type == 'mar':
            self.mask_ratio_generator = stats.truncnorm(-4, 0, loc=1.0, scale=0.25)

    def _downsample(self, notes, density, tempo, global_cond):
        factor = self.downsample_factor
        if self.level_idx == 0:
             B, T = density.shape
             if T % factor != 0:
                 pad = factor - (T % factor)
                 density = F.pad(density, (0, pad))
                 tempo = F.pad(tempo, (0, pad), mode='replicate')
                 T = density.shape[1]
             d_down = density.view(B, T // factor, factor).mean(dim=2)
             t_down = tempo.view(B, T // factor, factor).mean(dim=2)
             return torch.stack([d_down, t_down], dim=1), None
        else:
             B, C, T, P = notes.shape
             if T % factor != 0:
                 pad = factor - (T % factor)
                 notes = F.pad(notes, (0, 0, 0, pad))
                 T = notes.shape[2]
             notes_reshaped = notes.view(B, C, T // factor, factor, P)
             c0 = notes_reshaped[:, 0].max(dim=2)[0]
             c1 = notes_reshaped[:, 1].max(dim=2)[0]
             return torch.stack([c0, c1], dim=1), None

    def forward(self, notes, tempo, density, global_cond=None, bar_pos=None, parent_emb=None, stats=None):
        if stats is None: stats = {}
        
        # 1. Prepare Inputs
        gt, _ = self._downsample(notes, density, tempo, global_cond)
        
        # Downsample bar_pos
        curr_bar_pos = None
        if bar_pos is not None:
            factor = self.downsample_factor
            T = bar_pos.shape[1]
            if T % factor != 0:
                pad = factor - (T % factor)
                bar_pos_padded = F.pad(bar_pos, (0, pad), value=0) 
                curr_bar_pos = bar_pos_padded.view(bar_pos.shape[0], -1, factor)[:, :, 0]
            else:
                curr_bar_pos = bar_pos.view(bar_pos.shape[0], -1, factor)[:, :, 0]

        # 2. Prepare Condition
        cond = None
        g_emb = None
        if self.global_proj is not None and global_cond is not None:
            factor = self.downsample_factor
            B, T, C = global_cond.shape
            if T % factor != 0:
                global_cond = F.pad(global_cond, (0, 0, 0, factor - (T % factor)))
            g_down = global_cond.view(B, -1, factor, C).mean(dim=2)
            g_emb = self.global_proj(g_down)
            
        p_emb = None
        # Ensure sequence length matches gt (Time is always dim 2: B, C, T or B, C, T, P)
        T_target = gt.shape[2]

        if self.parent_proj is not None and parent_emb is not None:
            curr_T = gt.shape[2]
            # parent_emb from previous level is (B, T_parent, E_parent)
            # Issue 7: We use Conv1d to aggregate context.
            # p_emb_up shape should be compatible with Conv1d input (B, E_parent, T_curr)
            
            # Transpose parent_emb to (B, E, T) for interpolation
            p_emb_t = parent_emb.transpose(1, 2) 
            p_emb_up = F.interpolate(p_emb_t, size=curr_T, mode='linear', align_corners=False)
            
            # Conv1d expects (B, Cin, T), parent_proj[0] is Linear, parent_proj[1] is Conv1d.
            # Wait, my definition was:
            # self.parent_proj = nn.Sequential(
            #    nn.Linear(embed_dims[level_idx-1], compressed_dim),
            #    nn.Conv1d(compressed_dim, compressed_dim, kernel_size=3, padding=1),
            #    nn.GELU()
            # )
            # Linear applies to last dim. Conv1d applies to last dimension as Time? No, Conv1d is (B, C, T).
            # So I need to be careful with transposes.
            
            # 1. Linear Project: (B, E_parent, T) -> (B, T, E_parent) -> Linear -> (B, T, Compressed)
            p_emb_up_t = p_emb_up.transpose(1, 2) # (B, T, E)
            p_emb_proj = self.parent_proj[0](p_emb_up_t) # (B, T, C)
            
            # 2. Conv1d: (B, T, C) -> (B, C, T) -> Conv -> (B, C, T)
            p_emb_conv_in = p_emb_proj.transpose(1, 2)
            p_emb_conv_out = self.parent_proj[1](p_emb_conv_in)
            
            # 3. Activation: (B, C, T)
            p_emb_act = self.parent_proj[2](p_emb_conv_out)
            
            # 4. Back to (B, T, C) for conditioning
            p_emb = p_emb_act.transpose(1, 2)
            
            if p_emb.shape[1] != T_target:
                 p_emb = F.interpolate(p_emb.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)
        
        if g_emb is not None and p_emb is not None:
             # Double check g_emb length too
             if g_emb.shape[1] != T_target:
                  g_emb = F.interpolate(g_emb.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)
             cond = torch.cat([p_emb, g_emb], dim=-1)
        elif g_emb is not None:
            if g_emb.shape[1] != T_target:
                  g_emb = F.interpolate(g_emb.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)
            cond = g_emb
        elif p_emb is not None:
            cond = p_emb
            
        # 3. Forward Generator
        mask = None
        if self.generator_type == 'mar':
            T = gt.shape[2]
            mask_rates = self.mask_ratio_generator.rvs(gt.shape[0])  # 從 truncnorm(-4, 0, loc=1.0, scale=0.25) 採樣
            num_masked_tokens = torch.Tensor(np.ceil(T * mask_rates)).cuda()
            
            # 統一語義：mask = True 表示「遮蔽」，mask = False 表示「保留」
            mask = torch.zeros(gt.shape[0], T, device=gt.device, dtype=torch.bool)
            for i in range(gt.shape[0]):
                indices = torch.randperm(T, device=gt.device)[:int(num_masked_tokens[i])]
                mask[i, indices] = True  # True = 遮蔽
             
        is_causal = (self.generator_type == 'ar')
        x_emb = self.layer(gt, cond, mask, bar_pos=curr_bar_pos, is_causal=is_causal, use_sos=is_causal)
        
        # Output Proj
        # IMPORTANT FIX: Even if is_causal=False (MAR), PitchTransformer needs target for teacher forcing
        # ParallelOutputProj ignores target, so it's safe to always pass it if available
        
        is_pitch_ar = isinstance(self.layer.output_proj, (PitchTransformerOutputProj, LSTMOutputProj))
        use_target = gt if (is_causal or is_pitch_ar) else None
        
        logits = self.layer.output_proj(x_emb, target=use_target, teacher_forcing=True)
        
        stats[f'level_{self.level_idx}/logits'] = logits
        stats[f'level_{self.level_idx}/gt'] = gt
        stats[f'level_{self.level_idx}/mask'] = mask
        
        # 4. Recursion
        if self.next_level:
            _, next_stats = self.next_level(notes, tempo, density, global_cond, bar_pos, parent_emb=x_emb, stats=stats)
            return None, next_stats 
            
        return None, stats
        
    @torch.no_grad()
    def sample(self, batch_size, length, global_cond=None, bar_pos=None, parent_emb=None, 
           stats=None, temperature=1.0, cfg=1.0, num_iter=1,
           callback=None, filter_threshold=0.0, 
           initial_content=None, inpaint_mask=None):
        """
        Sample from this level and recursively from child levels.
        
        Args:
            callback: Optional callback(step, total_steps, message)
            filter_threshold: Filter notes with probability < threshold
            initial_content: Initial content for inpainting (B, 2, T, 128) or (B, 2, T) for level 0
            inpaint_mask: Mask indicating which positions to inpaint (B, T) - True = inpaint
        """
        if stats is None: stats = {}
        device = next(self.parameters()).device
        
        # 1. Determine dimensions
        current_T = length // self.downsample_factor
        
        # 2. Prepare Condition
        cond = None
        g_emb = None
        if self.global_proj is not None and global_cond is not None:
            factor = self.downsample_factor
            B, T_g, C = global_cond.shape
            
            if T_g % factor != 0:
                 pad = factor - (T_g % factor)
                 global_cond_padded = F.pad(global_cond, (0, 0, 0, pad))
            else:
                 global_cond_padded = global_cond
                 
            g_down = global_cond_padded.view(B, -1, factor, C).mean(dim=2)
            
            if g_down.shape[1] > current_T:
                g_down = g_down[:, :current_T]
            elif g_down.shape[1] < current_T:
                g_down = F.pad(g_down, (0, 0, 0, current_T - g_down.shape[1]))
                
            g_emb = self.global_proj(g_down)

        p_emb = None
        if self.parent_proj is not None and parent_emb is not None:
            p_emb_t = parent_emb.transpose(1, 2) 
            p_emb_up = F.interpolate(p_emb_t, size=current_T, mode='linear', align_corners=False)
            
            p_emb_up_t = p_emb_up.transpose(1, 2)
            p_emb_proj = self.parent_proj[0](p_emb_up_t)
            
            p_emb_conv_in = p_emb_proj.transpose(1, 2)
            p_emb_conv_out = self.parent_proj[1](p_emb_conv_in)
            
            p_emb_act = self.parent_proj[2](p_emb_conv_out)
            
            p_emb = p_emb_act.transpose(1, 2)
            
        # Combine conditions - IMPORTANT: Always maintain expected cond_dim
        cond_conditional = None
        cond_unconditional = None
        
        # Determine if we need to pad with zeros to match expected cond_dim
        if self.cond_dim is not None:
            if g_emb is not None and p_emb is not None:
                # Both parent and global conditioning
                cond_conditional = torch.cat([p_emb, g_emb], dim=-1)
                cond_unconditional = torch.cat([p_emb, torch.zeros_like(g_emb)], dim=-1)
            elif g_emb is not None:
                # Only global conditioning (Level 0)
                cond_conditional = g_emb
                cond_unconditional = torch.zeros_like(g_emb)
            elif p_emb is not None:
                # Only parent conditioning - pad with zeros to match expected cond_dim
                # This happens when global_cond=None but we have parent
                expected_dim = self.cond_dim
                current_dim = p_emb.shape[-1]
                if current_dim < expected_dim:
                    # Pad with zeros for missing global conditioning
                    pad_size = expected_dim - current_dim
                    zero_pad = torch.zeros(p_emb.shape[0], p_emb.shape[1], pad_size, device=p_emb.device)
                    cond_conditional = torch.cat([p_emb, zero_pad], dim=-1)
                    cond_unconditional = cond_conditional  # Same (no CFG without global cond)
                else:
                    cond_conditional = p_emb
                    cond_unconditional = p_emb
            else:
                # No conditioning - this shouldn't happen if cond_dim is not None
                # But handle it gracefully
                cond_conditional = None
                cond_unconditional = None
            
        # Prepare bar_pos
        curr_bar_pos = None
        if bar_pos is not None:
            factor = self.downsample_factor
            T_bar = bar_pos.shape[1]
            if T_bar % factor != 0:
                pad = factor - (T_bar % factor)
                bar_pos_padded = F.pad(bar_pos, (0, pad), value=0) 
                curr_bar_pos = bar_pos_padded.view(bar_pos.shape[0], -1, factor)[:, :, 0]
            else:
                curr_bar_pos = bar_pos.view(bar_pos.shape[0], -1, factor)[:, :, 0]
            
            if curr_bar_pos.shape[1] > current_T:
                curr_bar_pos = curr_bar_pos[:, :current_T]

        # 3. Generation Loop
        if self.level_idx == 0:
            generated = torch.zeros(batch_size, 2, current_T, device=device)
        else:
            generated = torch.zeros(batch_size, 2, current_T, 128, device=device)

        # 如果有 initial_content，使用它作為起點
        if initial_content is not None:
            # Downsample initial_content to current level
            if self.level_idx == 0:
                # Structure level: downsample from full resolution
                factor = self.downsample_factor
                B, C, T = initial_content.shape[:3]
                if T % factor != 0:
                    pad = factor - (T % factor)
                    initial_content = F.pad(initial_content, (0, 0, 0, pad) if initial_content.dim() == 3 else (0, 0, 0, 0, 0, pad))
                
                if initial_content.dim() == 4:  # (B, 2, T, 128)
                    # Extract density and tempo
                    density = initial_content[:, 0].mean(dim=-1)  # (B, T)
                    tempo = initial_content[:, 1].mean(dim=-1)  # (B, T)
                    initial_struct = torch.stack([density, tempo], dim=1)  # (B, 2, T)
                    # Downsample
                    initial_struct = initial_struct.view(B, 2, -1, factor).mean(dim=-1)
                    generated = initial_struct
                else:  # (B, 2, T)
                    generated = initial_content.view(B, 2, -1, factor).mean(dim=-1)
            else:
                # Content level
                if initial_content.shape[2] == current_T:
                    generated = initial_content
                else:
                    # Downsample
                    factor = initial_content.shape[2] // current_T
                    generated = initial_content[:, :, ::factor, :]
        
        # Downsample inpaint_mask to current level
        curr_inpaint_mask = None
        if inpaint_mask is not None:
            factor = self.downsample_factor
            T_mask = inpaint_mask.shape[1]
            if T_mask % factor != 0:
                pad = factor - (T_mask % factor)
                inpaint_mask = F.pad(inpaint_mask, (0, pad), value=False)
            # Use max pooling to preserve True values
            curr_inpaint_mask = inpaint_mask.view(batch_size, -1, factor).any(dim=-1)
            if curr_inpaint_mask.shape[1] > current_T:
                curr_inpaint_mask = curr_inpaint_mask[:, :current_T]
        
        # MAR/AR generation logic
        if self.generator_type == 'mar':
            # MAR with inpainting support
            mask = torch.ones(batch_size, current_T, device=device, dtype=torch.bool)
            
            # If inpainting, only mask the inpaint regions
            if curr_inpaint_mask is not None:
                mask = curr_inpaint_mask
            
            orders = torch.argsort(torch.rand(batch_size, current_T, device=device), dim=1)
            
            for iteration in range(num_iter):
                if callback:
                    callback(iteration, num_iter, f"Level {self.level_idx}, iteration {iteration+1}/{num_iter}")
                
                # Conditional forward pass (一次處理所有位置)
                x_emb_cond = self.layer(generated, cond_conditional, mask=mask, bar_pos=curr_bar_pos, is_causal=False)
                
                # CFG
                if cfg > 1.0 and cond_unconditional is not None:
                    x_emb_uncond = self.layer(generated, cond_unconditional, mask=mask, bar_pos=curr_bar_pos, is_causal=False)
                    x_emb = x_emb_uncond + cfg * (x_emb_cond - x_emb_uncond)
                else:
                    x_emb = x_emb_cond
                
                # 生成預測
                if self.level_idx == 0:
                    out = self.layer.output_proj(x_emb)
                    # 只更新 masked 位置
                    generated = torch.where(mask.unsqueeze(1), out, generated)
                else:
                    if hasattr(self.layer.output_proj, 'sample'):
                        logits, sample_val = self.layer.output_proj.sample(x_emb, temperature=temperature)
                    else:
                        logits = self.layer.output_proj(x_emb)
                        
                        l_note = logits[:, 0:1, :, :]
                        l_vel = logits[:, 1:, :, :]
                        
                        note = (torch.sigmoid(l_note) > 0.5).float()
                        
                        vel_idx = torch.argmax(l_vel, dim=1, keepdim=True).float()
                        vel = vel_idx / (self.num_vel_bins - 1.0)
                        
                        vel = vel * note 
                        
                        sample_val = torch.cat([note, vel], dim=1) 
                        generated[:, :, t:t+1, :] = sample_val
            
                    # 只更新 masked 位置
                    mask_4d = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
                    generated = torch.where(mask_4d, sample_val, generated)
                
                # 計算下一輪的 mask（cosine schedule）
                if iteration < num_iter - 1:
                    mask_ratio = np.cos(math.pi / 2. * (iteration + 1) / num_iter)
                    num_masked = max(1, int(np.floor(current_T * mask_ratio)))
                    
                    # 根據固定順序選擇要保持 masked 的位置
                    new_mask = torch.zeros(batch_size, current_T, device=device, dtype=torch.bool)
                    for b in range(batch_size):
                        mask_indices = orders[b, :num_masked]
                        new_mask[b, mask_indices] = True
                    mask = new_mask

                # Apply filter_threshold if specified
                if filter_threshold > 0 and self.level_idx > 0:
                    # Filter low-probability notes
                    note_probs = torch.sigmoid(logits[:, 0, :, :])
                    low_conf_mask = note_probs < filter_threshold
                    generated[:, 0][low_conf_mask] = 0.0
                    generated[:, 1][low_conf_mask] = 0.0
            
        else:
            # AR: 逐步生成（保持原邏輯）
            for t in range(current_T):
                x_emb_cond = self.layer(generated, cond_conditional, mask=None, bar_pos=curr_bar_pos, is_causal=True)
                x_emb_t_cond = x_emb_cond[:, t:t+1]
                
                if cfg > 1.0 and cond_unconditional is not None:
                    x_emb_uncond = self.layer(generated, cond_unconditional, mask=None, bar_pos=curr_bar_pos, is_causal=True)
                    x_emb_t_uncond = x_emb_uncond[:, t:t+1]
                    x_emb_t = x_emb_t_uncond + cfg * (x_emb_t_cond - x_emb_t_uncond)
                else:
                    x_emb_t = x_emb_t_cond
                
                if self.level_idx == 0:
                    # Structure: Continuous output
                    out = self.layer.output_proj(x_emb_t)
                    generated[:, :, t:t+1] = out
                else:
                    # Content: Sample from output projection
                    if hasattr(self.layer.output_proj, 'sample'):
                        logits, sample_val = self.layer.output_proj.sample(x_emb_t, temperature=temperature)
                    else:
                        logits = self.layer.output_proj(x_emb_t)
                    
                        l_note = logits[:, 0:1, :, :]
                        l_vel = logits[:, 1:, :, :]
                        
                        note = (torch.sigmoid(l_note) > 0.5).float()
                        
                        vel_idx = torch.argmax(l_vel, dim=1, keepdim=True).float()
                        vel = vel_idx / (self.num_vel_bins - 1.0)
                        
                        vel = vel * note 
                        
                        sample_val = torch.cat([note, vel], dim=1) 
                        generated[:, :, t:t+1, :] = sample_val
        
        stats[f'level_{self.level_idx}/output'] = generated.clone()
        
        # 4. Recursion
        if self.next_level:
            x_emb_full = self.layer(generated, cond_conditional, mask=None, bar_pos=curr_bar_pos, is_causal=(self.generator_type == 'ar'))
            
            # Get num_iter for next level
            # Assuming num_iter_list is passed through stats or as parameter
            # For now, use same num_iter (can be refined later)
            return self.next_level.sample(batch_size, length, global_cond, bar_pos, parent_emb=x_emb_full, stats=stats, temperature=temperature, cfg=cfg, num_iter=num_iter, callback=callback, filter_threshold=filter_threshold, initial_content=initial_content, inpaint_mask=inpaint_mask)
            
        # Return final result
        intermediates = []
        for i in range(self.level_idx + 1):
            if f'level_{i}/output' in stats:
                item = {
                    'output': stats[f'level_{i}/output'],
                    'level': i,
                    'is_structure': (i == 0)
                }
                intermediates.append(item)
                
        return generated, intermediates

class RecursiveFractalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.root = FractalGen(config, 0, global_cond_dim=12) 
        
        num_levels = len(config.architecture.embed_dim_list)
        self.loss_sigmas = nn.Parameter(torch.zeros(num_levels)) 
        
    def forward(self, notes, tempo, density, global_cond, bar_pos=None):
        _, stats = self.root(notes, tempo, density, global_cond, bar_pos)
        
        total_loss = 0
        num_vel_bins = self.config.piano_roll.num_velocity_bins
        
        for i in range(len(self.loss_sigmas)):
            logits = stats.get(f'level_{i}/logits')
            if logits is None: continue
            
            gt = stats[f'level_{i}/gt']
            mask = stats[f'level_{i}/mask']
            
            if i == 0:
                l_dens = F.mse_loss(logits[:, 0], gt[:, 0])
                l_tempo = F.mse_loss(logits[:, 1], gt[:, 1])
                raw_loss = l_dens + l_tempo
            else:
                # Content (Issue 4: Classification for Velocity)
                # Logits: (B, 1+num_vel_bins, T, 128)
                note_logits = logits[:, 0, :, :] 
                note_gt = gt[:, 0, :, :]
                l_note = F.binary_cross_entropy_with_logits(note_logits, note_gt)
                
                vel_logits = logits[:, 1:, :, :] # (B, num_vel_bins, T, 128)
                vel_gt = gt[:, 1, :, :] 
                
                # Quantize GT to bins
                vel_gt_idx = (vel_gt * (num_vel_bins - 1)).long()
                vel_gt_idx = torch.clamp(vel_gt_idx, 0, num_vel_bins - 1)
                
                active_mask = (note_gt > 0.5)
                
                if active_mask.sum() > 0:
                    v_log_flat = vel_logits.permute(0, 2, 3, 1)[active_mask] # (N_active, num_vel_bins)
                    v_gt_flat = vel_gt_idx[active_mask] 
                    l_vel = F.cross_entropy(v_log_flat, v_gt_flat)
                else:
                    l_vel = 0.0
                    
                raw_loss = 10.0 * l_note + 1.0 * l_vel
                
            s = self.loss_sigmas[i]
            sigma_sq = torch.exp(s)
            weighted_loss = (1.0 / (2 * sigma_sq)) * raw_loss + s / 2.0
            
            total_loss += weighted_loss
            stats[f'level_{i}/raw_loss'] = raw_loss
            stats[f'level_{i}/sigma'] = torch.exp(s/2)
            
        return total_loss, stats
    
    @torch.no_grad()
    def sample(self, batch_size, length, global_cond=None, bar_pos=None, temperature=1.0, 
           cfg=1.0, num_iter_list=None, return_intermediates=False,
           callback=None, filter_threshold=0.0, 
           initial_content=None, inpaint_mask=None):
        """
        Generate samples using the hierarchical fractal network.
        
        Args:
            batch_size: Number of samples to generate
            length: Target sequence length (will be downsampled at each level)
            global_cond: Optional global conditioning (e.g., chroma features) (B, T, C)
            bar_pos: Optional bar position embeddings (B, T)
            temperature: Sampling temperature (higher = more random)
            cfg: Classifier-Free Guidance scale (1.0 = no guidance, >1.0 = stronger conditioning)
            num_iter_list: List of iteration counts per level for MAR refinement (e.g., [8, 4, 2])
            return_intermediates: Whether to return intermediate outputs (currently always returns)
            callback: Optional callback function(step, total_steps, message) for progress updates
            filter_threshold: Threshold for filtering low-probability notes (0.0 to disable)
            initial_content: Optional initial content for conditional/inpainting (B, 2, T, 128)
            inpaint_mask: Optional mask for inpainting (B, T) - True = inpaint, False = keep
            
        Returns:
            generated: Final generated piano roll (B, 2, T, 128)
            intermediates: List of intermediate outputs from each level
        """
        # Default num_iter if not provided
        if num_iter_list is None:
            num_levels = len(self.config.architecture.embed_dim_list)
            num_iter_list = [1] * num_levels  # Single pass for all levels
        
        # Use first level's num_iter (will be passed down recursively)
        num_iter = num_iter_list[0] if len(num_iter_list) > 0 else 1
        
        return self.root.sample(
            batch_size=batch_size,
            length=length, 
            global_cond=global_cond, 
            bar_pos=bar_pos, 
            temperature=temperature,
            cfg=cfg,
            num_iter=num_iter,
            callback=callback,
            filter_threshold=filter_threshold,
            initial_content=initial_content,
            inpaint_mask=inpaint_mask
        )
