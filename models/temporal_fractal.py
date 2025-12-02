import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict

from .attention import Attention

# ==============================================================================
# Components
# ==============================================================================

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
    Uses an intermediate Pitch Embedding Layer to capture harmonic relationships.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # Input: (B, 2, T, 128). Flattened to (B, T, 256) usually.
        # Improved: Project 128 Pitch dim first?
        # Critique Suggestion: Pitch Embedding Layer.
        # "Don't just throw 128 pitches into Conv1d."
        # "Use a shared weight Linear or small Conv1d on Pitch axis."
        
        # Approach:
        # 1. Treat (2, 128) as features per time step.
        # 2. Flatten to (B, T, 256).
        # 3. Dense projection to higher dim (e.g. 512) to allow "wiggle room".
        # 4. Then mix.
        
        # Let's implement the "Pitch Embedding" idea more explicitly.
        # We process the (2, 128) block at each timestep.
        # Let's use a Linear layer that maps (2*128) -> embed_dim.
        # But add a hidden layer to capture interactions?
        # Critique says: "at least add a Dense layer... to project to higher dim"
        
        self.pitch_mlp = nn.Sequential(
            nn.Linear(in_channels * 128, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.pos_emb = TimePositionalEmbedding(embed_dim)

    def forward(self, x):
        # x: (B, 2, T, 128)
        B, C, T, P = x.shape
        # Permute to (B, T, C*P) -> (B, T, 256)
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * P) 
        
        x = self.pitch_mlp(x) # (B, T, E)
        x = x + self.pos_emb(x)
        return x

class FractalOutputProj(nn.Module):
    """
    Project (B, T, E) -> (B, 2, T, 128).
    """
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels * 128)

    def forward(self, x):
        # x: (B, T, E)
        B, T, E = x.shape
        x = self.proj(x) # (B, T, 2 * 128)
        x = x.view(B, T, 2, 128)
        x = x.permute(0, 2, 1, 3) # (B, 2, T, 128)
        return x

class StructureInputProj(nn.Module):
    """
    Project Structure (B, 2, T) -> (B, T, E).
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        self.pos_emb = TimePositionalEmbedding(embed_dim)

    def forward(self, x):
        # x: (B, 2, T)
        x = x.permute(0, 2, 1) # (B, T, 2)
        x = self.proj(x) # (B, T, E)
        x = x + self.pos_emb(x)
        return x

class StructureOutputProj(nn.Module):
    """
    Project (B, T, E) -> (B, 2, T).
    """
    def __init__(self, embed_dim, out_channels=2):
        super().__init__()
        self.proj = nn.Linear(embed_dim, out_channels)

    def forward(self, x):
        # x: (B, T, E)
        x = self.proj(x) # (B, T, 2)
        x = x.permute(0, 2, 1) # (B, 2, T)
        return x

class FractalBlock(nn.Module):
    """
    Transformer Block with Time Attention and AdaLN.
    Modified to use AdaLN for Tempo/Structure conditioning.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0, cond_dim=None):
        super().__init__()
        # Always use AdaLN if cond_dim is provided
        self.use_adaln = cond_dim is not None
        
        if self.use_adaln:
            self.norm1 = AdaLN(embed_dim, cond_dim)
            self.norm2 = AdaLN(embed_dim, cond_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            
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
        # cond: (B, T, C) or (B, C) - Conditioning signal (Tempo/Structure)
        
        residual = x
        
        if self.use_adaln:
            x = self.norm1(x, cond)
        else:
            x = self.norm1(x)
            
        x = self.attn(x)
        x = residual + x
        
        residual = x
        
        if self.use_adaln:
            x = self.norm2(x, cond)
        else:
            x = self.norm2(x)
            
        x = self.mlp(x)
        x = residual + x
            
        return x

class TemporalGenerator(nn.Module):
    """
    MAR Generator for a specific temporal resolution.
    Handles both Structure (Level 0) and Content (Level 1+) generation.
    """
    def __init__(self, embed_dim, num_heads, num_blocks, cond_dim=None, input_channels=2, is_structure_level=False):
        super().__init__()
        self.is_structure_level = is_structure_level
        
        if is_structure_level:
            # Input: (B, 2, T) -> Density, Tempo
            self.input_proj = StructureInputProj(input_channels, embed_dim)
            self.output_proj = StructureOutputProj(embed_dim, input_channels)
        else:
            # Input: (B, 2, T, 128) -> Note, Vel
            self.input_proj = FractalInputProj(input_channels, embed_dim)
            self.output_proj = FractalOutputProj(embed_dim, input_channels)

        self.blocks = nn.ModuleList([
            FractalBlock(embed_dim, num_heads, cond_dim=cond_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim) if cond_dim is None else AdaLN(embed_dim, cond_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward_features(self, x, cond=None, mask=None, prev_emb=None):
        """
        Forward pass returning embeddings.
        x: Input tensor
           - Structure: (B, 2, T)
           - Content:   (B, 2, T, 128)
        cond: Conditioning embedding (Global)
        mask: (B, T) - Boolean mask (True = masked)
        prev_emb: Conditioning from previous level (Structure/Content upsampled)
        """
        x_emb = self.input_proj(x) # (B, T, E)
        
        # Apply mask token
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) # (B, T, 1)
            x_emb = x_emb * (~mask_expanded) + self.mask_token * mask_expanded
            
        # Determine effective conditioning signal
        # If prev_emb is provided (Level 1+), use it as the primary condition for AdaLN.
        # If Level 0, use global cond.
        # Note: If we want both Global AND Structure, we should fuse them before.
        # Here we prioritize Structure for L1+.
        effective_cond = cond
        if prev_emb is not None:
            effective_cond = prev_emb

        # Transformer blocks
        for block in self.blocks:
            x_emb = block(x_emb, effective_cond)
            
        # Final Norm
        if isinstance(self.norm, AdaLN):
            x_emb = self.norm(x_emb, effective_cond)
        else:
            x_emb = self.norm(x_emb)
            
        return x_emb

    def forward(self, x, cond=None, mask=None, prev_emb=None):
        x_emb = self.forward_features(x, cond, mask, prev_emb)
        logits = self.output_proj(x_emb) 
        return logits, x_emb

class FractalLoss(nn.Module):
    def forward(self, pred, target, mask):
        # mask: [Batch, Time]
        
        if pred.dim() == 3: # Structure Level (B, 2, T)
            # Ch0: Density (MSE)
            # Ch1: Tempo (MSE)
            loss_density = F.mse_loss(pred[:, 0], target[:, 0], reduction='none')
            loss_tempo = F.mse_loss(pred[:, 1], target[:, 1], reduction='none')
            
            # Apply mask
            loss_density = (loss_density * mask).sum() / (mask.sum() + 1e-6)
            loss_tempo = (loss_tempo * mask).sum() / (mask.sum() + 1e-6)
            
            return loss_density + loss_tempo
            
        else: # Content Level (B, 2, T, 128)
            # Ch0: Note (BCEWithLogits)
            # Ch1: Velocity (MSE)
            
            # Broadcast mask to (B, 1, T, 1)
            mask_bc = mask.unsqueeze(1).unsqueeze(-1)
            
            loss_note = F.binary_cross_entropy_with_logits(
                pred[:, 0], target[:, 0], reduction='none'
            )
            loss_note = (loss_note * mask_bc.squeeze(1).squeeze(-1)).sum() / (mask.sum() * 128 + 1e-6)
            
            # Velocity Loss - only on active notes in TARGET
            active_notes = target[:, 0] > 0.5
            vel_mask = active_notes * mask_bc.squeeze(1).squeeze(-1)
            
            loss_vel = F.mse_loss(pred[:, 1], target[:, 1], reduction='none')
            loss_vel = (loss_vel * vel_mask).sum() / (vel_mask.sum() + 1e-6)
            
            # Weighting: Note loss is harder (sparse), so weight it up
            return 10.0 * loss_note + 1.0 * loss_vel

class TemporalFractalNetwork(nn.Module):
    """
    Hierarchical Fractal Network with Temporal Resolution Scaling.
    Level 0: Structure (Density, Tempo)
    Level 1+: Content (Note, Velocity)
    """
    def __init__(self, 
                 input_channels=2,
                 embed_dims=(512, 256, 128),
                 num_heads=(8, 4, 2),
                 num_blocks=(6, 4, 2),
                 cond_dim=0):
        super().__init__()
        
        self.levels = nn.ModuleList()
        self.cond_projs = nn.ModuleList()
        # Downsample factors relative to INPUT resolution.
        # Level 0 is coarsest. Level 2 is finest (factor 1).
        # Assuming 3 levels. 
        # Level 0: Structure. Resolution T/16? Or T/4?
        # Critique says "Level 0 -> Chord/Density".
        # Let's say:
        # Level 0 (Structure): 1/16 resolution.
        # Level 1 (Content): 1/4 resolution.
        # Level 2 (Content): 1/1 resolution.
        self.downsample_factors = [16, 4, 1] 
        self.embed_dims = embed_dims
        
        for i in range(3):
            # Projection for previous level embedding
            # Level 0: None
            # Level 1: Cond on L0 (Structure)
            # Level 2: Cond on L1 (Content)
            if i > 0:
                self.cond_projs.append(nn.Linear(embed_dims[i-1], embed_dims[i]))
            else:
                self.cond_projs.append(nn.Identity())
            
            is_structure = (i == 0)
            
            # Determine condition dimension for this level
            # Level 0: Global Condition only
            # Level 1+: Structure/Previous Level Condition (projected to current embed_dim)
            if i == 0:
                level_cond_dim = cond_dim if cond_dim > 0 else None
            else:
                # We project prev_emb to embed_dims[i] before passing as cond
                level_cond_dim = embed_dims[i]

            gen = TemporalGenerator(
                embed_dim=embed_dims[i],
                num_heads=num_heads[i],
                num_blocks=num_blocks[i],
                cond_dim=level_cond_dim,
                input_channels=input_channels, 
                is_structure_level=is_structure
            )
            self.levels.append(gen)
            
        self.loss_fn = FractalLoss()

    def _downsample_structure(self, density, tempo, factor):
        # density, tempo: (B, T)
        if factor == 1:
            return torch.stack([density, tempo], dim=1) # (B, 2, T)
        
        B, T = density.shape
        if T % factor != 0:
            pad_len = factor - (T % factor)
            density = F.pad(density, (0, pad_len))
            tempo = F.pad(tempo, (0, pad_len), mode='replicate')
            T = density.shape[1]
            
        # Reshape (B, T/F, F)
        dens_reshaped = density.view(B, T // factor, factor)
        tempo_reshaped = tempo.view(B, T // factor, factor)
        
        # Density: Mean (average density over block)
        d_down = dens_reshaped.mean(dim=2)
        # Tempo: Mean (average tempo over block)
        t_down = tempo_reshaped.mean(dim=2)
        
        return torch.stack([d_down, t_down], dim=1) # (B, 2, T_new)

    def _downsample_content(self, notes, factor):
        # notes: (B, 2, T, 128) [Note, Vel]
        if factor == 1:
            return notes
        
        B, C, T, P = notes.shape
        if T % factor != 0:
            pad_len = factor - (T % factor)
            notes = F.pad(notes, (0, 0, 0, pad_len))
            T = notes.shape[2]
            
        notes_reshaped = notes.view(B, C, T // factor, factor, P)
        
        # Note (Ch0): Max pool (presence)
        c0 = notes_reshaped[:, 0].max(dim=3)[0]
        # Vel (Ch1): Max pool
        c1 = notes_reshaped[:, 1].max(dim=3)[0]
        
        return torch.stack([c0, c1], dim=1)

    def _upsample_embedding(self, emb, target_len):
        # emb: (B, T_low, E)
        B, T_low, E = emb.shape
        emb_perm = emb.permute(0, 2, 1) 
        emb_up = F.interpolate(emb_perm, size=target_len, mode='linear', align_corners=False)
        return emb_up.permute(0, 2, 1)

    def forward(self, notes, tempo, density, global_cond=None):
        # notes: (B, 2, T, 128)
        # tempo: (B, T)
        # density: (B, T)
        
        stats = {}
        total_loss = 0
        
        prev_level_emb = None
        
        for i, level_mod in enumerate(self.levels):
            factor = self.downsample_factors[i]
            
            # Prepare GT
            if i == 0: # Structure Level
                gt = self._downsample_structure(density, tempo, factor) # (B, 2, T_low)
                # For L0, we condition on nothing (or global_cond)
                level_cond = global_cond
                
            else: # Content Level
                gt = self._downsample_content(notes, factor) # (B, 2, T_low, 128)
                
                # CRITICAL FIX:
                # Pass Tempo/Structure as side-condition via AdaLN, NOT as input channel.
                # We need the Structure info for this level.
                # We have 'prev_level_emb' which is the embedding of L0 (Structure).
                # This contains the Structure info!
                # In previous iteration we used `cond_projs` to project `prev_level_emb` to `cond_emb`.
                # That `cond_emb` IS the Structure condition.
                # So we just pass that as `cond`.
                level_cond = None # Will be set via cond_emb logic below
            
            B = gt.shape[0]
            T = gt.shape[2] 
            
            # Random mask ratio [0.5, 1.0]
            mask_ratio = torch.rand(1, device=notes.device).item() * 0.5 + 0.5
            mask = torch.rand(B, T, device=notes.device) < mask_ratio
            
            # Conditioning
            cond_emb = None
            if i > 0 and prev_level_emb is not None:
                upsampled = self._upsample_embedding(prev_level_emb, T) 
                cond_emb = self.cond_projs[i](upsampled) # (B, T, E_curr)
                level_cond = cond_emb # Use structure embedding as condition
            elif i == 0:
                # Level 0 has no previous level, use global_cond if available
                level_cond = global_cond

            # Forward
            # Note: we pass 'level_cond' as the 'cond' argument to TemporalGenerator
            logits, x_emb = level_mod(gt, level_cond, mask, None)
            
            # Store for next level
            prev_level_emb = x_emb
            
            # Loss
            loss = self.loss_fn(logits, gt, mask)
            total_loss += loss
            
            stats[f'level_{i}/loss'] = loss.detach()
            stats[f'level_{i}/mask_ratio'] = mask_ratio

        return total_loss, stats

    @torch.no_grad()
    def sample(self, batch_size, length, global_cond=None, 
               cfg=1.0, temperature=1.0, num_iter_list=[8, 4, 2],
               initial_content=None, inpaint_mask=None,
               return_intermediates=False):
        """
        Generate samples using hierarchical MAR.
        Supports inpainting if initial_content and inpaint_mask are provided.
        
        initial_content: (B, 2, T, 128) - Optional known content
        inpaint_mask: (B, T) - 1.0 where generation is needed (mask), 0.0 where content is known (keep).
                      Note: This is "generation mask" logic (1=unknown).
        """
        device = next(self.parameters()).device
        
        intermediates = []
        prev_level_emb = None
        
        final_output = None
        
        for i, level_mod in enumerate(self.levels):
            factor = self.downsample_factors[i]
            current_len = length // factor
            
            # Initialize canvas
            if i == 0: # Structure
                current_canvas = torch.zeros(batch_size, 2, current_len, device=device)
            else: # Content
                current_canvas = torch.zeros(batch_size, 2, current_len, 128, device=device)
                
            # Inpainting setup for this level
            level_known = None
            level_inpaint_mask = None
            
            if initial_content is not None:
                # Downsample initial content/mask to current level
                if i == 0:
                    # Structure level inpainting? 
                    # If we provide Note content, we can derive Structure content (Density/Tempo) from it?
                    # Or we just generate structure from scratch (unconstrained) unless user provides Structure hints?
                    # For now, let's assume inpainting is mostly for Content.
                    # But strictly, if user freezes a region, we should respect it at L0 too?
                    # To respect it at L0, we'd need to compute Density/Tempo from initial_content.
                    # Let's implement that: derive L0 GT from initial_content.
                    
                    # Extract tempo/density from initial_content
                    # initial_content is (B, 2, T, 128).
                    # Note layer is Ch0.
                    notes_full = initial_content
                    # Density = sum(notes_full[:, 0], dim=2) / 12.0 clipped
                    dens_full = notes_full[:, 0].sum(dim=-1) / 12.0
                    dens_full = torch.clamp(dens_full, 0, 1)
                    # Tempo is hard to extract from roll without explicit tempo track.
                    # Let's assume Tempo is uniform or just use 0.5 (120 BPM) if unknown.
                    # Or ignore L0 constraints for now if too complex.
                    # Let's ignore L0 constraints for simplicity in this pass, 
                    # UNLESS inpaint_mask is ALL 0 (reconstruction).
                    # Actually, consistent inpainting requires L0 to match L1 constraints.
                    # Let's compute approximate L0 GT.
                    tempo_full = torch.ones_like(dens_full) * 0.5 # Default placeholder
                    
                    level_known = self._downsample_structure(dens_full, tempo_full, factor) # (B, 2, T_low)
                    
                    if inpaint_mask is not None:
                        # Downsample mask (Max pool - if any part of block is masked, mask the block?)
                        # Or Min pool? If we need to keep ANY part, we should perhaps keep constraints?
                        # Let's use Max pool (conservative masking - if in doubt, generate)
                        m_reshaped = inpaint_mask.view(batch_size, current_len, factor)
                        level_inpaint_mask = m_reshaped.max(dim=2)[0] # (B, T_low)
                    
                else:
                    # Content level
                    level_known = self._downsample_content(initial_content, factor)
                    if inpaint_mask is not None:
                        m_reshaped = inpaint_mask.view(batch_size, current_len, factor)
                        level_inpaint_mask = m_reshaped.max(dim=2)[0] # (B, T_low)
            
            # Iterative generation
            num_iter = num_iter_list[i] if i < len(num_iter_list) else 1
            
            for step in range(num_iter):
                # Masking schedule
                mask_ratio = 1.0 - (step / num_iter)
                
                # Create random mask for refinement
                # If inpainting, we MUST mask the inpaint region (level_inpaint_mask=1).
                # We also randomly mask the rest for MAR refinement? 
                # Or just mask the inpaint region?
                # Standard MAR: mask random subset.
                # Inpainting: Fixed mask = level_inpaint_mask.
                # Combined: mask = level_inpaint_mask OR (random_mask AND level_inpaint_mask?)
                # Actually, for inpainting, we want to generate the masked region.
                # The known region should stay fixed.
                # So we enforce known region at input.
                
                # Current simplified approach:
                # 1. Current Canvas has estimates.
                # 2. Enforce known parts on Canvas.
                # 3. Mask unknown parts (probabilistically or fully).
                
                if level_known is not None and level_inpaint_mask is not None:
                    # Enforce known values where mask is 0
                    # level_inpaint_mask is 1 for "generate this", 0 for "keep this"
                    keep_mask = (level_inpaint_mask < 0.5)
                    if i == 0:
                         # (B, 2, T)
                        current_canvas = torch.where(keep_mask.unsqueeze(1), level_known, current_canvas)
                    else:
                         # (B, 2, T, 128)
                        current_canvas = torch.where(keep_mask.unsqueeze(1).unsqueeze(-1), level_known, current_canvas)

                # Generate mask for this step
                # We want to refine the 'inpaint' region.
                # mask should be 1 where we want model to predict.
                # Usually we mask everything initially, then less and less.
                # Here we just mask the inpaint region?
                # Let's stick to "Refine All" logic but enforce knowns.
                # So we ask model to predict EVERYTHING (Mask=1), but we inject knowns into input.
                
                # Wait, if we pass Mask=1, model ignores input at those positions.
                # So we should only Mask the parts we want to change.
                # If step 0 (high noise), mask everything inside inpaint_region.
                # If step N (low noise), mask subset?
                # For simplicity: Mask = 1.0 (Full Mask) on Inpaint Region.
                # Outside Inpaint Region: Mask = 0 (No Mask), so model sees known context.
                
                if level_inpaint_mask is not None:
                    # Mask only the inpaint region
                    # (B, T) boolean
                    mask = (level_inpaint_mask > 0.5)
                    # Also apply random masking? No, simple inpainting first.
                else:
                    # Unconditional: Full mask initially?
                    # MAR usually: Mask 100% -> Predict -> Sample -> Mask 90%...
                    # Here we implement simplified "Predict All" at every step but feed back previous prediction.
                    mask = torch.ones(batch_size, current_len, dtype=torch.bool, device=device)

                # Conditioning
                cond_emb = None
                level_cond = None
                if i > 0 and prev_level_emb is not None:
                    upsampled = self._upsample_embedding(prev_level_emb, current_len)
                    cond_emb = self.cond_projs[i](upsampled)
                    level_cond = cond_emb
                elif i == 0:
                    level_cond = global_cond
                
                # Forward
                logits, x_emb = level_mod(current_canvas, level_cond, mask, None)
                
                # Sample
                if i == 0:
                    # Structure (Regression)
                    # No sigmoid for regression targets? Or assumed normalized?
                    # Density/Tempo are 0-1 normalized.
                    pred = torch.clamp(logits, 0, 1)
                else:
                    # Content
                    # Ch0: Note (Sigmoid)
                    # Ch1: Vel (Linear/Clamp)
                    pred_note = torch.sigmoid(logits[:, 0])
                    pred_vel = torch.clamp(logits[:, 1], 0, 1)
                    pred = torch.stack([pred_note, pred_vel], dim=1)
                
                # Update canvas
                current_canvas = pred
                
                if return_intermediates:
                    intermediates.append({
                        'level': i,
                        'step': step,
                        'output': current_canvas.detach().cpu(),
                        'is_structure': (i == 0)
                    })

            # End of Level: Get final embedding (Enforce knowns one last time)
            if level_known is not None and level_inpaint_mask is not None:
                keep_mask = (level_inpaint_mask < 0.5)
                if i == 0:
                    current_canvas = torch.where(keep_mask.unsqueeze(1), level_known, current_canvas)
                else:
                    current_canvas = torch.where(keep_mask.unsqueeze(1).unsqueeze(-1), level_known, current_canvas)
            
            # Get embedding with NO mask
            mask = torch.zeros(batch_size, current_len, dtype=torch.bool, device=device)
            cond_emb = None
            level_cond = None
            if i > 0 and prev_level_emb is not None:
                upsampled = self._upsample_embedding(prev_level_emb, current_len)
                cond_emb = self.cond_projs[i](upsampled)
                level_cond = cond_emb
            elif i == 0:
                level_cond = global_cond
            
            _, x_emb = level_mod(current_canvas, level_cond, mask, None)
            prev_level_emb = x_emb
            
            if i > 0: # Only return content as final output
                final_output = current_canvas

        return final_output, intermediates

