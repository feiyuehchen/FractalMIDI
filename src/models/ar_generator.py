"""
Autoregressive (AR) generator for piano roll patches.
"""

from functools import partial
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch.nn.functional as F
from .blocks import CausalBlock


class PianoRollAR(nn.Module):
    """Autoregressive generator for piano roll patches."""

    def __init__(self, seq_len, patch_size, cond_embed_dim, embed_dim, num_blocks, num_heads,
                 attn_dropout=0.1, proj_dropout=0.1, num_conds=5, guiding_pixel=False,
                 grad_checkpointing=False, max_seq_len=None, img_size=128, scan_order='row_major',
                 piano_roll_height=128, velocity_vocab_size=256, ar_prefix_mask_ratio=1.0):
        super().__init__()

        self.img_size = img_size
        self.piano_roll_height = piano_roll_height  # Piano roll height (MIDI pitch range)
        self.velocity_vocab_size = velocity_vocab_size  # Velocity vocabulary size
        # ar_prefix_mask_ratio is deprecated/unused in improved AR
        
        # Use max_seq_len for param init, but limit runtime seq_len for lower levels to avoid OOM
        if self.img_size >= self.piano_roll_height:
            # Top level: use full seq_len for temporal dimension
            self.seq_len = max_seq_len or seq_len
        else:
            # Lower levels: limit to nominal seq_len to avoid explosion
            self.seq_len = seq_len
        self.max_seq_len = max_seq_len or seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_conds = num_conds
        self.guiding_pixel = guiding_pixel
        self.grad_checkpointing = grad_checkpointing
        self.scan_order = scan_order  # 'row_major' or 'column_major'

        # Embedding layers with careful initialization for AR stability
        self.patch_emb = nn.Linear(1 * patch_size ** 2, embed_dim, bias=True)
        self.patch_emb_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.cond_emb = nn.Linear(cond_embed_dim, embed_dim, bias=True)
        
        # Position embedding with +1 for condition token
        self.pos_embed = nn.Parameter(torch.zeros(1, (self.max_seq_len + 1), embed_dim))

        self.blocks = nn.ModuleList([
            CausalBlock(embed_dim, num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=proj_dropout, attn_drop=attn_dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.initialize_weights()

    def _get_scan_order(self, h_patches, w_patches):
        """
        Get the scanning order for patches.
        
        Args:
            h_patches: Number of patches in height dimension
            w_patches: Number of patches in width dimension
        
        Returns:
            order: List of (h, w) tuples indicating the order to visit patches
        """
        if self.scan_order == 'row_major':
            # Row-major: scan left to right, top to bottom
            # 0  1  2  3
            # 4  5  6  7
            # 8  9  10 11
            order = [(h, w) for h in range(h_patches) for w in range(w_patches)]
        elif self.scan_order == 'column_major':
            # Column-major: scan top to bottom, left to right
            # 0  3  6  9
            # 1  4  7  10
            # 2  5  8  11
            order = [(h, w) for w in range(w_patches) for h in range(h_patches)]
        else:
            raise ValueError(f"Unknown scan_order: {self.scan_order}")
        return order
    
    def _spatial_to_linear(self, h, w, h_patches, w_patches):
        """Convert spatial coordinates (h, w) to linear index."""
        if self.scan_order == 'row_major':
            return h * w_patches + w
        elif self.scan_order == 'column_major':
            return w * h_patches + h
        else:
            raise ValueError(f"Unknown scan_order: {self.scan_order}")
    
    def _linear_to_spatial(self, idx, h_patches, w_patches):
        """Convert linear index to spatial coordinates (h, w)."""
        if self.scan_order == 'row_major':
            h = idx // w_patches
            w = idx % w_patches
        elif self.scan_order == 'column_major':
            h = idx % h_patches
            w = idx // h_patches
        else:
            raise ValueError(f"Unknown scan_order: {self.scan_order}")
        return h, w

    def initialize_weights(self):
        # Improved position embedding initialization
        # Use smaller std for better stability, especially for AR
        torch.nn.init.normal_(self.pos_embed, std=.01)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use xavier_uniform for better gradient flow
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """
        Convert piano roll to patches, respecting scan_order.
        """
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        
        if self.scan_order == 'column_major':
            # Transpose grid (h, w) -> (w, h) so that flattening works in column-major order
            x = x.permute(0, 2, 1, 3, 4, 5)
            
        x = x.reshape(bsz, -1, c * p ** 2)
        return x

    def unpatchify(self, x, h_patches, w_patches):
        """
        Convert patches back to piano roll, respecting scan_order.
        """
        bsz = x.shape[0]
        p = self.patch_size

        if self.scan_order == 'column_major':
            # Reshape to transposed grid (w, h)
            x = x.reshape(bsz, w_patches, h_patches, 1, p, p)
            # Permute back to (h, w)
            x = x.permute(0, 2, 1, 3, 4, 5)
        else:
            x = x.reshape(bsz, h_patches, w_patches, 1, p, p)

        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, 1, h_patches * p, w_patches * p)
        return x

    def _build_neighbor_conditions(self, middle_cond):
        bsz, seq_len, c = middle_cond.size()
        
        # Special case: single token (when using incremental generation)
        if seq_len == 1:
            # For a single token, we can't compute spatial neighbors easily from just the tensor
            # In standard AR, we usually just pass the embedding itself as the "context" for next level
            # Or we'd need spatial info. For simplicity, we replicate.
            single_cond = middle_cond  # (bsz, 1, c)
            return [single_cond for _ in range(self.num_conds)]
        
        # Calculate grid dimensions based on img_size and patch_size
        h = self.img_size // self.patch_size
        
        # For variable width support
        if h * h == seq_len:
            w = h
        else:
            w = seq_len // h
            if h * w != seq_len:
                # Fallback
                for possible_h in range(1, seq_len + 1):
                    if seq_len % possible_h == 0:
                        h = possible_h
                        w = seq_len // h
                        break

        # Reshape to grid, respecting scan_order
        if self.scan_order == 'column_major':
            # Sequence is column-major: reshape(w, h) fills h first (column)
            middle_grid = middle_cond.reshape(bsz, w, h, c)
            middle_grid = middle_grid.permute(0, 2, 1, 3) # -> (bsz, h, w, c)
        else:
            middle_grid = middle_cond.reshape(bsz, h, w, c)

        top_cond_grid = torch.cat([torch.zeros(bsz, 1, w, c, device=middle_cond.device), middle_grid[:, :-1]], dim=1)
        right_cond_grid = torch.cat([middle_grid[:, :, 1:], torch.zeros(bsz, h, 1, c, device=middle_cond.device)], dim=2)
        bottom_cond_grid = torch.cat([middle_grid[:, 1:], torch.zeros(bsz, 1, w, c, device=middle_cond.device)], dim=1)
        left_cond_grid = torch.cat([torch.zeros(bsz, h, 1, c, device=middle_cond.device), middle_grid[:, :, :-1]], dim=2)

        neighbor_grids = [middle_grid, top_cond_grid, right_cond_grid, bottom_cond_grid, left_cond_grid]
        final_neighbors = []

        for g in neighbor_grids:
            if self.scan_order == 'column_major':
                # Permute to (bsz, w, h, c) then reshape to (bsz, seq_len, c)
                # This flattens in column-major order
                g = g.permute(0, 2, 1, 3)
                g = g.reshape(bsz, seq_len, c)
            else:
                g = g.reshape(bsz, seq_len, c)
            final_neighbors.append(g)

        return final_neighbors[:self.num_conds]

    def predict(self, patches, cond_list, input_pos=None):
        """
        Predict next tokens with correct causal indexing.
        
        Args:
            patches: Input patches (B, seq_len, patch_dim) - the history so far
            cond_list: List of conditions
            input_pos: Integer or None. If provided, returns prediction for token at this position.
                       
        Causal AR Logic (Generation):
            - At step k, history contains [P0, P1, ..., P_{k-1}] (k tokens)
            - After prepending cond: [Cond, P0, P1, ..., P_{k-1}] (k+1 tokens)
            - After transformer: [H_cond, H_0, H_1, ..., H_{k-1}] (k+1 hidden states)
            - H_cond predicts P0, H_0 predicts P1, ..., H_{k-1} predicts P_k
            - To predict P_k (token at position k), we use H_{k-1} at index k
            - So: x[:, input_pos] gives us the predictor for token at position input_pos
            
        Causal AR Logic (Training):
            - Input patches: [P0, P1, ..., P_N] (N+1 tokens)
            - After prepending cond: [Cond, P0, P1, ..., P_N] (N+2 tokens)
            - After transformer: [H_cond, H_0, H_1, ..., H_N] (N+2 hidden states)
            - We want: H_cond→P0, H_0→P1, ..., H_{N-1}→P_N
            - So we return [H_cond, H_0, ..., H_{N-1}] = x[:, :-1]
        """
        x = self.patch_emb(patches)
        cond_source = cond_list[0]
        if cond_source.dim() > 2:
            leading = cond_source.shape[0]
            cond_source = cond_source.reshape(leading, -1, cond_source.shape[-1]).mean(dim=1)
        cond_token = self.cond_emb(cond_source).unsqueeze(1)
        
        # Prepend condition token to create full input sequence
        x = torch.cat([cond_token, x], dim=1)

        seq_len = x.size(1)
        pos_embed = self.pos_embed
        
        # Positional encoding
        if seq_len <= pos_embed.size(1):
            pos = pos_embed[:, :seq_len, :]
        else:
            pos = F.interpolate(
                pos_embed.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        x = x + pos
        x = self.patch_emb_ln(x)

        # Apply transformer blocks with causal attention
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for block in self.blocks:
                x = checkpoint(block, x)
        else:
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)

        # Output handling with correct causal indexing
        if input_pos is not None:
            # Generation mode: return predictor for token at position input_pos
            # x has shape (B, len(history)+1, D) where history has input_pos tokens
            # We want x[:, input_pos] which is the hidden state that predicts token input_pos
            middle_cond = x[:, input_pos : input_pos + 1]
        else:
            # Training mode: return all predictors except the last one
            # Input has N patches, x has N+1 hidden states (including cond)
            # We return first N hidden states to predict N patches
            middle_cond = x[:, :-1]

        return self._build_neighbor_conditions(middle_cond)

    def forward(self, piano_rolls, cond_list):
        """
        Training forward pass for AR generator.
        
        Process:
        1. Patchify input piano rolls
        2. Pass through AR transformer to get conditions for next level
        3. Return all patches and their conditions for next level loss computation
        """
        width_before = piano_rolls.shape[-1]
        if width_before % self.patch_size != 0:
            pad_required = self.patch_size - (width_before % self.patch_size)
            piano_rolls = F.pad(piano_rolls, (0, pad_required))
        else:
            pad_required = 0

        patches = self.patchify(piano_rolls)  # Ground truth patches (B, seq_len, patch_dim)
        bsz = patches.size(0)
        seq_len_actual = patches.size(1)
        
        # AR Training: Feed full sequence, causal attention ensures autoregressive property
        # predict() with input_pos=None returns all predictors [H_cond, H_0, ..., H_{N-1}]
        # which will be used to compute loss against targets [P0, P1, ..., P_N]
        cond_list_next = self.predict(
            patches,
            cond_list,
            input_pos=None
        )
        
        # Flatten conditions for next level (each patch gets its own condition)
        for idx in range(len(cond_list_next)):
            cond_list_next[idx] = cond_list_next[idx].reshape(
                cond_list_next[idx].size(0) * cond_list_next[idx].size(1), -1
            )
        
        # Reshape patches for next level (flatten batch and sequence dimensions)
        patches_flat = patches.reshape(-1, 1, self.patch_size, self.patch_size)
        
        # Diagnostic statistics
        stats = {
            'sequence_length': torch.tensor(float(seq_len_actual), device=piano_rolls.device),
            'pad_added': torch.tensor(float(pad_required), device=piano_rolls.device),
            'num_patches': torch.tensor(float(patches_flat.size(0)), device=piano_rolls.device)
        }

        # No guiding pixel loss for AR
        return patches_flat, cond_list_next, torch.tensor(0.0, device=piano_rolls.device), stats

    def sample(self, cond_list, num_iter, cfg, cfg_schedule, temperature, filter_threshold,
               next_level_sample_function, visualize=False, _intermediates_list=None, _current_level=None,
               _patch_pos=None, target_width=None, inpainting_mask=None, inpainting_target=None):
        """
        Autoregressive sampling.
        """
        if cfg == 1.0:
            bsz = cond_list[0].size(0)
        else:
            bsz = cond_list[0].size(0) // 2

        if self.guiding_pixel:
            guiding_pixels = torch.zeros(bsz, 1, device=cond_list[0].device, dtype=cond_list[0].dtype)
            if not cfg == 1.0:
                guiding_pixels = torch.cat([guiding_pixels, guiding_pixels], dim=0)
            cond_list.append(guiding_pixels)

        # Calculate dimensions
        h_patches = self.img_size // self.patch_size
        if target_width is not None and self.img_size == 128:
            w_patches = target_width // self.patch_size
        else:
            w_patches = h_patches
        actual_seq_len = h_patches * w_patches

        # Initialize canvas
        # If inpainting, we can initialize with target (though we overwrite in loop)
        canvas = torch.full(
            (bsz, actual_seq_len, self.patch_size**2),
            fill_value=-1.0,
            device=cond_list[0].device,
            dtype=cond_list[0].dtype
        )
        if inpainting_target is not None:
             # Ensure target matches canvas shape
             if inpainting_target.shape[1] == actual_seq_len:
                 canvas = inpainting_target.clone().to(canvas.device).to(canvas.dtype)
        
        # Record every step for smoother animation
        record_interval = 1 if _intermediates_list is not None else actual_seq_len + 1
        
        # Scan order for visualization/placement
        scan_order = self._get_scan_order(h_patches, w_patches)
        
        # O(N^2) generation loop (re-process history at each step)
        # This is slower but more robust than caching complexity
        for step_idx in range(actual_seq_len):
            # Check if this step is known (for inpainting)
            # Assuming batch size 1 for simplicity of logic optimization, or consistent mask across batch
            is_known = False
            if inpainting_mask is not None:
                # If all batch items are known at this step
                if torch.all(inpainting_mask[:, step_idx] == 0):
                    is_known = True
            
            if is_known:
                # Skip prediction and generation, just keep value from canvas (initialized from target)
                # Canvas was already initialized with target, so we are good.
                # Just record if needed
                pass
            else:
                # Prepare history
                if step_idx == 0:
                    history = torch.zeros(bsz, 0, self.patch_size**2, device=canvas.device, dtype=canvas.dtype)
                else:
                    history = canvas[:, :step_idx, :]
                
                # Predict next token
                # We pass the full history so far.
                # input_pos=step_idx tells predict to return the output for the NEXT token (at step_idx)
                conds = self.predict(history, cond_list, input_pos=step_idx)
                
                # Extract condition (predict returns list of neighbors)
                # conds is list of (B, 1, D)
                cond_for_patch = [c[:, 0] for c in conds]
                
                # Handle CFG
                if cfg != 1.0:
                    cond_for_patch = [torch.cat([c, c], dim=0) for c in cond_for_patch]
                
                # Generate patch content
                patch_content = next_level_sample_function(
                    cond_list=cond_for_patch,
                    cfg=cfg,
                    temperature=temperature,
                    filter_threshold=filter_threshold
                )
                
                if cfg != 1.0:
                    patch_content = patch_content[:bsz]
                
                patch_flat = patch_content.reshape(bsz, -1)
                
                # Update canvas
                # If partial inpainting (mixed known/unknown in batch), we should mask the update
                if inpainting_mask is not None:
                    # Only update where mask is 1 (unknown)
                    # mask shape (B, seq_len). Slice for this step: (B,)
                    step_mask = inpainting_mask[:, step_idx].bool() # True = Unknown/Update
                    # Expand mask for patch dim
                    step_mask_exp = step_mask.unsqueeze(-1).expand_as(patch_flat)
                    
                    # Where mask is True (unknown), use generated. Where False (known), keep canvas (target).
                    canvas[:, step_idx] = torch.where(step_mask_exp, patch_flat, canvas[:, step_idx])
                else:
                    canvas[:, step_idx] = patch_flat
            
            # Record intermediate
            if _intermediates_list is not None and _current_level == 0 and (step_idx % record_interval == 0 or step_idx == actual_seq_len - 1):
                if self.patch_size == 1:
                    img_temp = canvas.reshape(bsz, 1, self.img_size, w_patches)
                else:
                    img_temp = self.unpatchify(canvas, h_patches, w_patches)
                
                _intermediates_list['canvas'][:bsz] = img_temp
                _intermediates_list['frames'].append({
                    'level': _current_level,
                    'iteration': step_idx + 1,
                    'level_name': f'AR L{_current_level} patch {step_idx+1}/{actual_seq_len}',
                    'img_size': self.img_size,
                    'output': _intermediates_list['canvas'][0:1].clone(),
                    'step_id': step_idx
                })

        if self.patch_size == 1:
            return canvas.reshape(bsz, 1, self.img_size, w_patches)
        return self.unpatchify(canvas, h_patches, w_patches)
