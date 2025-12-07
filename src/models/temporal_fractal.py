import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

from .attention import Attention
from .components import (
    AdaLN, AdaLNModulator, TimePositionalEmbedding, 
    FractalInputProj, FractalOutputProj, 
    StructureInputProj, StructureOutputProj
)

class FractalBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=None):
        super().__init__()
        self.use_adaln = cond_dim is not None
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        if self.use_adaln:
            self.adaln1_mod = AdaLNModulator(embed_dim, cond_dim)
            self.adaln2_mod = AdaLNModulator(embed_dim, cond_dim)
            
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.cond_dim = cond_dim

    def forward(self, x, cond=None):
        # x: (B, T, E)
        # cond: (B, T, C) or (B, C)
        
        residual = x
        x = self.norm1(x) 
        if self.use_adaln:
            x = self.adaln1_mod(x, cond) 
        x = self.attn(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        if self.use_adaln:
            x = self.adaln2_mod(x, cond)
        x = self.mlp(x)
        x = residual + x
            
        return x

class TemporalGenerator(nn.Module):
    """
    MAR Generator for a specific temporal resolution.
    """
    def __init__(self, embed_dim, num_heads, num_blocks, cond_dim=None, input_channels=2, 
                 is_structure_level=False, max_bar_len=16):
        super().__init__()
        self.is_structure_level = is_structure_level
        self.cond_dim = cond_dim
        
        if is_structure_level:
            self.input_proj = StructureInputProj(input_channels, embed_dim, max_bar_len=max_bar_len)
            self.output_proj = StructureOutputProj(embed_dim, input_channels)
        else:
            self.input_proj = FractalInputProj(input_channels, embed_dim, max_bar_len=max_bar_len)
            self.output_proj = FractalOutputProj(embed_dim, input_channels)

        self.blocks = nn.ModuleList([
            FractalBlock(embed_dim, num_heads, cond_dim=cond_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim) if cond_dim is None else AdaLN(embed_dim, cond_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_features(self, x, cond=None, mask=None, prev_emb=None, bar_pos=None):
        x_emb = self.input_proj(x, bar_pos) # (B, T, E)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) # (B, T, 1)
            x_emb = x_emb * (~mask_expanded) + self.mask_token * mask_expanded
            
        effective_cond = cond
        if prev_emb is not None:
            effective_cond = prev_emb

        for block in self.blocks:
            x_emb = block(x_emb, effective_cond)
            
        if isinstance(self.norm, AdaLN):
            x_emb = self.norm(x_emb, effective_cond)
        else:
            x_emb = self.norm(x_emb)
            
        return x_emb

    def forward(self, x, cond=None, mask=None, prev_emb=None, bar_pos=None):
        x_emb = self.forward_features(x, cond, mask, prev_emb, bar_pos)
        logits = self.output_proj(x_emb) 
        return logits, x_emb

class FractalLoss(nn.Module):
    def forward(self, pred, target, mask, weights=None):
        tempo_weight = weights.get('tempo', 1.0) if weights else 1.0
        
        if pred.dim() == 3: # Structure Level (B, 2, T)
            loss_density = F.mse_loss(pred[:, 0], target[:, 0], reduction='none')
            loss_tempo = F.mse_loss(pred[:, 1], target[:, 1], reduction='none')
            
            loss_density = (loss_density * mask).sum() / (mask.sum() + 1e-6)
            loss_tempo = (loss_tempo * mask).sum() / (mask.sum() + 1e-6)
            
            return loss_density + tempo_weight * loss_tempo
            
        else: # Content Level (B, 2, T, 128)
            mask_bc = mask.unsqueeze(1).unsqueeze(-1)
            
            loss_note = F.binary_cross_entropy_with_logits(
                pred[:, 0], target[:, 0], reduction='none'
            )
            loss_note = (loss_note * mask_bc.squeeze(1)).sum() / (mask.sum() * 128 + 1e-6)
            
            active_notes = target[:, 0] > 0.5
            vel_mask = active_notes * mask_bc.squeeze(1)
            
            loss_vel = F.mse_loss(pred[:, 1], target[:, 1], reduction='none')
            loss_vel = (loss_vel * vel_mask).sum() / (vel_mask.sum() + 1e-6)
            
            return 10.0 * loss_note + 1.0 * loss_vel

class TemporalFractalNetwork(nn.Module):
    def __init__(self, 
                 input_channels=2,
                 embed_dims=(512, 256, 128),
                 num_heads=(8, 4, 2),
                 num_blocks=(6, 4, 2),
                 cond_dim=0,
                 max_bar_len=64,
                 compressed_dim=32,
                 compression_act='relu',
                 full_mask_prob=0.1):
        super().__init__()
        
        self.levels = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        self.downsample_factors = [16, 4, 1] 
        self.embed_dims = embed_dims
        
        self.compressed_dim = compressed_dim
        self.full_mask_prob = full_mask_prob
        self.max_bar_len = max_bar_len
        
        # Configurable activation
        act_layer = nn.ReLU()
        if compression_act == 'gelu':
            act_layer = nn.GELU()
        elif compression_act == 'identity':
            act_layer = nn.Identity()
        
        for i in range(3):
            if i > 0:
                self.cond_projs.append(nn.Sequential(
                    nn.Linear(embed_dims[i-1], compressed_dim),
                    act_layer
                ))
                level_cond_dim = compressed_dim
            else:
                self.cond_projs.append(nn.Identity())
                level_cond_dim = cond_dim if cond_dim > 0 else None
            
            is_structure = (i == 0)

            gen = TemporalGenerator(
                embed_dim=embed_dims[i],
                num_heads=num_heads[i],
                num_blocks=num_blocks[i],
                cond_dim=level_cond_dim,
                input_channels=input_channels, 
                is_structure_level=is_structure,
                max_bar_len=max_bar_len
            )
            self.levels.append(gen)
            
        self.loss_fn = FractalLoss()

    def _downsample_structure(self, density, tempo, factor):
        if factor == 1:
            return torch.stack([density, tempo], dim=1) 
        
        B, T = density.shape
        if T % factor != 0:
            pad_len = factor - (T % factor)
            density = F.pad(density, (0, pad_len))
            tempo = F.pad(tempo, (0, pad_len), mode='replicate')
            T = density.shape[1]
            
        dens_reshaped = density.view(B, T // factor, factor)
        tempo_reshaped = tempo.view(B, T // factor, factor)
        
        d_down = dens_reshaped.mean(dim=2)
        t_down = tempo_reshaped.mean(dim=2)
        
        return torch.stack([d_down, t_down], dim=1) 

    def _downsample_content(self, notes, factor):
        if factor == 1:
            return notes
        
        B, C, T, P = notes.shape
        if T % factor != 0:
            pad_len = factor - (T % factor)
            notes = F.pad(notes, (0, 0, 0, pad_len))
            T = notes.shape[2]
            
        notes_reshaped = notes.view(B, C, T // factor, factor, P)
        c0 = notes_reshaped[:, 0].max(dim=2)[0]
        c1 = notes_reshaped[:, 1].max(dim=2)[0]
        
        res = torch.stack([c0, c1], dim=1)
        return res

    def _upsample_embedding(self, emb, target_len):
        B, T_low, E = emb.shape
        emb_perm = emb.permute(0, 2, 1) 
        emb_up = F.interpolate(emb_perm, size=target_len, mode='linear', align_corners=False)
        return emb_up.permute(0, 2, 1)

    def _downsample_bar_pos(self, bar_pos, factor):
        if factor == 1:
            return bar_pos
            
        B, T = bar_pos.shape
        if T % factor != 0:
            pad_len = factor - (T % factor)
            last_pos = bar_pos[:, -1]
            pad_seq = torch.arange(1, pad_len+1, device=bar_pos.device).unsqueeze(0).expand(B, -1)
            # Assuming max_bar_len wrapping roughly, but actually bar_pos values are just integers
            # We just extend pattern or reuse last
            pad_vals = (last_pos.unsqueeze(1) + pad_seq) % self.max_bar_len # Correctly use configured max_bar_len
            bar_pos = torch.cat([bar_pos, pad_vals], dim=1)
            T = bar_pos.shape[1]
            
        return bar_pos.view(B, T // factor, factor)[:, :, 0]

    def forward(self, notes, tempo, density, global_cond=None, loss_weights=None, bar_pos=None):
        stats = {}
        total_loss = 0
        prev_level_emb = None
        
        for i, level_mod in enumerate(self.levels):
            factor = self.downsample_factors[i]
            
            if i == 0: 
                gt = self._downsample_structure(density, tempo, factor) 
                level_cond = global_cond
            else: 
                gt = self._downsample_content(notes, factor) 
                level_cond = None 
            
            if bar_pos is not None:
                level_bar_pos = self._downsample_bar_pos(bar_pos, factor)
            else:
                level_bar_pos = None

            B = gt.shape[0]
            T = gt.shape[2] 
            
            # Mix of random masking and full masking (to learn generation from scratch)
            if torch.rand(1).item() < self.full_mask_prob:
                mask_ratio = 1.0
            else:
                mask_ratio = torch.rand(1, device=notes.device).item() * 0.5 + 0.5

            mask = torch.rand(B, T, device=notes.device) < mask_ratio
            
            cond_emb = None
            if i > 0 and prev_level_emb is not None:
                upsampled = self._upsample_embedding(prev_level_emb, T) 
                cond_emb = self.cond_projs[i](upsampled) 
                level_cond = cond_emb 
            elif i == 0:
                level_cond = global_cond

            logits, x_emb = level_mod(gt, level_cond, mask, None, bar_pos=level_bar_pos)
            prev_level_emb = x_emb
            
            loss = self.loss_fn(logits, gt, mask, weights=loss_weights)
            total_loss += loss
            
            stats[f'level_{i}/loss'] = loss.detach()
            stats[f'level_{i}/mask_ratio'] = mask_ratio

        return total_loss, stats

    @torch.no_grad()
    def sample(self, batch_size, length, global_cond=None, 
               cfg=1.0, temperature=1.0, num_iter_list=[8, 4, 2],
               initial_content=None, inpaint_mask=None,
               return_intermediates=False, callback=None,
               bar_pos=None, refinement_steps=3,
               deep_sample_mask_ratio=0.3): # Configurable mask ratio
        
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        if bar_pos is None:
            t_seq = torch.arange(length, device=device)
            bar_pos = (t_seq % 16).unsqueeze(0).expand(batch_size, -1)
        
        intermediates = []
        prev_level_emb = None
        final_output = None
        
        # Tracking condition influence
        condition_influence = [] # List of (level, condition_map)
        
        for i, level_mod in enumerate(self.levels):
            factor = self.downsample_factors[i]
            current_len = length // factor
            
            level_bar_pos = self._downsample_bar_pos(bar_pos, factor)
            # Fix: Trim bar_pos if padding in _downsample made it longer than current_len
            if level_bar_pos.shape[1] > current_len:
                level_bar_pos = level_bar_pos[:, :current_len]
            
            if i == 0: 
                current_canvas = torch.zeros(batch_size, 2, current_len, device=device, dtype=dtype)
            else: 
                current_canvas = torch.zeros(batch_size, 2, current_len, 128, device=device, dtype=dtype)
                
            level_known = None
            level_inpaint_mask = None
            
            if initial_content is not None:
                if i == 0:
                    notes_full = initial_content
                    dens_full = notes_full[:, 0].sum(dim=-1) / 12.0
                    dens_full = torch.clamp(dens_full, 0, 1)
                    tempo_full = torch.ones_like(dens_full) * 0.5 
                    level_known = self._downsample_structure(dens_full, tempo_full, factor) 
                    if inpaint_mask is not None:
                        m_reshaped = inpaint_mask.view(batch_size, current_len, factor)
                        level_inpaint_mask = m_reshaped.max(dim=2)[0] 
                else:
                    level_known = self._downsample_content(initial_content, factor)
                    if inpaint_mask is not None:
                        m_reshaped = inpaint_mask.view(batch_size, current_len, factor)
                        level_inpaint_mask = m_reshaped.max(dim=2)[0] 
            
            num_iter = num_iter_list[i] if i < len(num_iter_list) else 1
            
            for step in range(num_iter):
                mask_ratio = 1.0 - (step / num_iter)
                
                if level_known is not None and level_inpaint_mask is not None:
                    keep_mask = (level_inpaint_mask < 0.5)
                    if i == 0:
                        current_canvas = torch.where(keep_mask.unsqueeze(1), level_known, current_canvas)
                    else:
                        current_canvas = torch.where(keep_mask.unsqueeze(1).unsqueeze(-1), level_known, current_canvas)

                if level_inpaint_mask is not None:
                    mask = (level_inpaint_mask > 0.5)
                else:
                    # Respect mask_ratio for iterative refinement
                    mask = torch.rand(batch_size, current_len, device=device) < mask_ratio

                cond_emb = None
                level_cond = None
                if i > 0 and prev_level_emb is not None:
                    upsampled = self._upsample_embedding(prev_level_emb, current_len)
                    cond_emb = self.cond_projs[i](upsampled)
                    level_cond = cond_emb
                    # Capture influence: This cond_emb is derived from Level i-1 structure
                    if step == 0: # Only capture once per level
                        infl_data = {
                            'level': i,
                            'cond_map': cond_emb.detach().cpu() # (B, T, C)
                        }
                        condition_influence.append(infl_data)
                        if callback:
                            callback({'type': 'condition_influence', 'data': infl_data})
                elif i == 0:
                    level_cond = global_cond
                
                logits, x_emb = level_mod(current_canvas, level_cond, mask, None, bar_pos=level_bar_pos)
                
                if i == 0:
                    pred = torch.clamp(logits, 0, 1)
                else:
                    pred_note = torch.sigmoid(logits[:, 0])
                    pred_vel = torch.clamp(logits[:, 1], 0, 1)
                    pred = torch.stack([pred_note, pred_vel], dim=1)
                
                current_canvas = pred
                
                step_data = {'level': i, 'step': step, 'output': current_canvas.detach().cpu(), 'is_structure': (i == 0)}
                if return_intermediates: intermediates.append(step_data)
                if callback: callback(step_data)

            # Deep Sampling Refinement Loop
            if i > 0 and refinement_steps > 0:
                for ref_step in range(refinement_steps):
                    if level_known is not None and level_inpaint_mask is not None:
                        keep_mask = (level_inpaint_mask < 0.5)
                        current_canvas = torch.where(keep_mask.unsqueeze(1).unsqueeze(-1), level_known, current_canvas)

                    # Use configurable mask ratio
                    refine_mask = torch.rand(batch_size, current_len, device=device) < deep_sample_mask_ratio
                    if level_inpaint_mask is not None:
                        refine_mask = refine_mask & (level_inpaint_mask > 0.5)
                    
                    cond_emb = None
                    if prev_level_emb is not None:
                        upsampled = self._upsample_embedding(prev_level_emb, current_len)
                        cond_emb = self.cond_projs[i](upsampled)
                        level_cond = cond_emb
                    
                    logits, _ = level_mod(current_canvas, level_cond, refine_mask, None, bar_pos=level_bar_pos)
                    
                    pred_note = torch.sigmoid(logits[:, 0])
                    pred_vel = torch.clamp(logits[:, 1], 0, 1)
                    pred = torch.stack([pred_note, pred_vel], dim=1)
                    
                    mask_bc = refine_mask.unsqueeze(1).unsqueeze(-1)
                    current_canvas = torch.where(mask_bc, pred, current_canvas)
                    
                    step_data = {'level': i, 'step': num_iter + ref_step, 'output': current_canvas.detach().cpu(), 'is_structure': False, 'is_refinement': True}
                    if return_intermediates: intermediates.append(step_data)

            # End of Level: Get final embedding
            if level_known is not None and level_inpaint_mask is not None:
                keep_mask = (level_inpaint_mask < 0.5)
                if i == 0:
                    current_canvas = torch.where(keep_mask.unsqueeze(1), level_known, current_canvas)
                else:
                    current_canvas = torch.where(keep_mask.unsqueeze(1).unsqueeze(-1), level_known, current_canvas)
            
            mask = torch.zeros(batch_size, current_len, dtype=torch.bool, device=device)
            
            cond_emb = None
            level_cond = None
            if i > 0 and prev_level_emb is not None:
                upsampled = self._upsample_embedding(prev_level_emb, current_len)
                cond_emb = self.cond_projs[i](upsampled)
                level_cond = cond_emb
            elif i == 0:
                level_cond = global_cond
            
            _, x_emb = level_mod(current_canvas, level_cond, mask, None, bar_pos=level_bar_pos)
            prev_level_emb = x_emb
            
            if i > 0:
                final_output = current_canvas

        # Add condition influence to results if needed, or just return in intermediates
        if return_intermediates:
            intermediates.append({'type': 'condition_influence', 'data': condition_influence})

        return final_output, intermediates
