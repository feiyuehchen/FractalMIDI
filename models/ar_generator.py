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
                 piano_roll_height=128, velocity_vocab_size=256):
        super().__init__()

        self.img_size = img_size
        self.piano_roll_height = piano_roll_height  # Piano roll height (MIDI pitch range)
        self.velocity_vocab_size = velocity_vocab_size  # Velocity vocabulary size
        
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

        self.patch_emb = nn.Linear(1 * patch_size ** 2, embed_dim, bias=True)
        self.patch_emb_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.cond_emb = nn.Linear(cond_embed_dim, embed_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, (self.max_seq_len + 1), embed_dim))

        self.blocks = nn.ModuleList([
            CausalBlock(embed_dim, num_heads, mlp_ratio=4.0,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        proj_drop=proj_dropout, attn_drop=attn_dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # KV Cache state
        self.cache_enabled = False

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

    def setup_caches(self, batch_size, max_seq_len):
        """
        Setup KV caches for all attention blocks.
        
        Args:
            batch_size: Batch size for generation
            max_seq_len: Maximum sequence length to cache
        """
        device = next(self.parameters()).device
        self.cache_enabled = True
        
        for block in self.blocks:
            # Allocate cache buffers for each attention layer
            # Cache shape: (batch_size, num_heads, max_seq_len, head_dim)
            cache_shape = (batch_size, block.attn.num_heads, max_seq_len + 1, block.attn.head_dim)  # +1 for condition token
            block.attn.k_cache = torch.zeros(cache_shape, device=device, dtype=torch.float32)
            block.attn.v_cache = torch.zeros(cache_shape, device=device, dtype=torch.float32)
    
    def clear_caches(self):
        """Clear all KV caches."""
        self.cache_enabled = False
        for block in self.blocks:
            block.attn.k_cache = None
            block.attn.v_cache = None

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x, h_patches, w_patches):
        bsz = x.shape[0]
        p = self.patch_size

        x = x.reshape(bsz, h_patches, w_patches, 1, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, 1, h_patches * p, w_patches * p)
        return x

    def _build_neighbor_conditions(self, middle_cond):
        bsz, seq_len, c = middle_cond.size()
        h = int(np.sqrt(seq_len))
        w = seq_len // h
        if h * w != seq_len:
            h = self.img_size // self.patch_size
            w = seq_len // h

        middle_grid = middle_cond.reshape(bsz, h, w, c)

        top_cond = torch.cat([torch.zeros(bsz, 1, w, c, device=middle_cond.device), middle_grid[:, :-1]], dim=1)
        right_cond = torch.cat([middle_grid[:, :, 1:], torch.zeros(bsz, h, 1, c, device=middle_cond.device)], dim=2)
        bottom_cond = torch.cat([middle_grid[:, 1:], torch.zeros(bsz, 1, w, c, device=middle_cond.device)], dim=1)
        left_cond = torch.cat([torch.zeros(bsz, h, 1, c, device=middle_cond.device), middle_grid[:, :, :-1]], dim=2)

        neighbors = [
            middle_cond,
            top_cond.reshape(bsz, seq_len, c),
            right_cond.reshape(bsz, seq_len, c),
            bottom_cond.reshape(bsz, seq_len, c),
            left_cond.reshape(bsz, seq_len, c)
        ]
        return neighbors[:self.num_conds]

    def predict(self, patches, cond_list, use_cache=False, cache_position=None):
        """
        Predict next tokens.
        
        Args:
            patches: Input patches (B, seq_len, patch_dim)
            cond_list: List of conditions
            use_cache: If True, use KV cache (for AR sampling)
            cache_position: Position in cache to update (for incremental generation)
        """
        x = self.patch_emb(patches)
        cond_source = cond_list[0]
        if cond_source.dim() > 2:
            leading = cond_source.shape[0]
            cond_source = cond_source.reshape(leading, -1, cond_source.shape[-1]).mean(dim=1)
        cond_token = self.cond_emb(cond_source).unsqueeze(1)
        x = torch.cat([cond_token, x], dim=1)

        seq_len = x.size(1)
        pos_embed = self.pos_embed
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

        # Apply transformer blocks with optional caching
        if self.grad_checkpointing and not torch.jit.is_scripting() and self.training:
            for block in self.blocks:
                x = checkpoint(block, x, use_cache, cache_position)
        else:
            for block in self.blocks:
                x = block(x, use_cache=use_cache, cache_position=cache_position)
        x = self.norm(x)

        middle_cond = x[:, 1:]
        return self._build_neighbor_conditions(middle_cond)

    def forward(self, piano_rolls, cond_list):
        """
        Training forward pass.
        For AR, we use teacher forcing: given full sequence, predict each position.
        """
        width_before = piano_rolls.shape[-1]
        if width_before % self.patch_size != 0:
            pad_required = self.patch_size - (width_before % self.patch_size)
            piano_rolls = F.pad(piano_rolls, (0, pad_required))
        else:
            pad_required = 0

        patches = self.patchify(piano_rolls)
        seq_len_actual = patches.size(1)
        
        # Use full sequence for training (teacher forcing)
        # No cache during training - process full sequence at once
        cond_list_next = self.predict(patches, cond_list, use_cache=False, cache_position=None)

        for idx in range(len(cond_list_next)):
            cond_list_next[idx] = cond_list_next[idx].reshape(
                cond_list_next[idx].size(0) * cond_list_next[idx].size(1), -1
            )

        patches = patches.reshape(patches.size(0) * patches.size(1), 1, self.patch_size, self.patch_size)

        stats = {
            'masked_fraction': torch.zeros(1, device=piano_rolls.device, dtype=torch.float32),
            'masked_tokens_per_sample': torch.tensor(float(seq_len_actual), device=piano_rolls.device),
            'sequence_length': torch.tensor(float(seq_len_actual), device=piano_rolls.device),
            'pad_added': torch.tensor(float(pad_required), device=piano_rolls.device)
        }

        return patches, cond_list_next, torch.tensor(0.0, device=piano_rolls.device), stats

    def sample(self, cond_list, num_iter, cfg, cfg_schedule, temperature, filter_threshold,
               next_level_sample_function, visualize=False, _intermediates_list=None, _current_level=None, _patch_pos=None, target_width=None):
        """
        Autoregressive sampling: generate each patch sequentially.
        
        Unlike MAR which masks and fills, AR generates patches one by one,
        using previously generated patches as context for the next patch.
        
        Args:
            target_width: Target width for generation at Level 0 only (128, 256, 512, etc.)
                         For other levels, this should be None and will use square patches.
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

        # Calculate actual sequence length
        # For Level 0 (img_size=128): use target_width to support variable widths
        # For other levels: generate square patches (width = height = img_size)
        h_patches = self.img_size // self.patch_size
        if target_width is not None and self.img_size == 128:
            # Level 0: variable width
            w_patches = target_width // self.patch_size
        else:
            # Other levels: square patches
            w_patches = h_patches
        actual_seq_len = h_patches * w_patches

        # Initialize canvas with 0 (silence) for unconditional generation
        canvas = torch.zeros(
            (bsz, actual_seq_len, self.patch_size**2),
            device=cond_list[0].device,
            dtype=cond_list[0].dtype
        )
        
        # Setup KV Cache for efficient AR generation
        cache_bsz = bsz if cfg == 1.0 else bsz * 2
        self.setup_caches(batch_size=cache_bsz, max_seq_len=actual_seq_len)
        
        # Generate patches autoregressively with KV Cache
        # For intermediate recording: only record every N patches to avoid too many frames
        record_interval = max(1, actual_seq_len // 8) if _intermediates_list is not None else actual_seq_len + 1
        
        # Get scanning order
        scan_order = self._get_scan_order(h_patches, w_patches)
        
        try:
            for step_idx in range(actual_seq_len):
                # Get spatial position for this step
                h, w = scan_order[step_idx]
                
                # Convert spatial position to linear index in canvas
                patch_idx = self._spatial_to_linear(h, w, h_patches, w_patches)
                
                # For KV Cache: we only need to process the current patch position
                # All previous patches are already cached
                if step_idx == 0:
                    # First step: process full canvas (all zeros) to initialize cache
                    current_canvas = canvas
                else:
                    # Subsequent steps: only process up to current position
                    # This allows cache to accumulate incrementally
                    current_canvas = canvas[:, :patch_idx+1, :]
                
                # Get conditions using KV Cache
                with torch.no_grad():
                    conds = self.predict(
                        current_canvas, 
                        cond_list, 
                        use_cache=True,
                        cache_position=step_idx if step_idx > 0 else None
                    )
                
                # Extract condition for current patch: (bsz, seq_len, dim) -> (bsz, dim)
                # For cached mode, we only get the last position's condition
                if step_idx == 0:
                    cond_for_patch = [c[:, patch_idx] for c in conds]
                else:
                    # When using cache, output is only for new positions
                    cond_for_patch = [c[:, -1] for c in conds]
                
                # Handle CFG: duplicate conditions if needed
                if cfg != 1.0:
                    cond_for_patch = [torch.cat([c, c], dim=0) for c in cond_for_patch]
                
                # Call next level to generate this single patch
                patch_content = next_level_sample_function(
                    cond_list=cond_for_patch,
                    cfg=cfg,
                    temperature=temperature,
                    filter_threshold=filter_threshold
                )
                
                # Handle CFG output: take only conditional branch
                if cfg != 1.0:
                    patch_content = patch_content[:bsz]
                
                # Flatten patch: (bsz, 1, H, W) -> (bsz, H*W)
                patch_flat = patch_content.reshape(bsz, -1)
                
                # Update canvas at current position
                canvas[:, patch_idx] = patch_flat
                
                # Record intermediate state periodically
                if _intermediates_list is not None and _current_level == 0 and (patch_idx % record_interval == 0 or patch_idx == actual_seq_len - 1):
                    if self.patch_size == 1:
                        img_temp = canvas.reshape(bsz, 1, self.img_size, w_patches)
                    else:
                        img_temp = self.unpatchify(canvas, h_patches, w_patches)
                    
                    _intermediates_list['canvas'][:bsz] = img_temp
                    _intermediates_list['frames'].append({
                        'level': _current_level,
                        'iteration': patch_idx + 1,
                        'level_name': f'AR L{_current_level} patch {patch_idx+1}/{actual_seq_len}',
                        'img_size': self.img_size,
                        'output': _intermediates_list['canvas'][0:1].clone(),
                        'step_id': patch_idx
                    })
        finally:
            # Always clean up cache
            self.clear_caches()

        if self.patch_size == 1:
            return canvas.reshape(bsz, 1, self.img_size, w_patches)
        return self.unpatchify(canvas, h_patches, w_patches)


