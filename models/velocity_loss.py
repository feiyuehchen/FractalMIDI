"""
Velocity loss layer for piano roll generation.
"""

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from .blocks import CausalBlock



class MlmLayer(nn.Module):
    """Masked language modeling layer."""
    def __init__(self, vocab_size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, x, word_embeddings):
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(x, word_embeddings)
        logits = logits + self.bias
        return logits



class PianoRollVelocityLoss(nn.Module):
    """
    Velocity prediction loss for piano roll patches.
    Adapted from PixelLoss for single-channel velocity values.
    """
    def __init__(self, c_channels, width, depth, num_heads, v_weight=1.0, level_index: int = 3, velocity_vocab_size=256):
        super().__init__()

        self.level_index = level_index
        self.velocity_vocab_size = velocity_vocab_size
        self.cond_proj = nn.Linear(c_channels, width)
        self.v_codebook = nn.Embedding(velocity_vocab_size, width)  # Velocity codebook [0-255]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))  # Learnable mask token for sampling

        self.ln = nn.LayerNorm(width, eps=1e-6)
        self.blocks = nn.ModuleList([
            CausalBlock(width, num_heads=num_heads, mlp_ratio=4.0,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                       proj_drop=0, attn_drop=0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-6)

        self.v_weight = v_weight
        self.v_mlm = MlmLayer(velocity_vocab_size)

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.v_codebook.weight, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
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

    def predict(self, target, cond_list):
        """Predict velocity values."""
        target = target.reshape(target.size(0), target.size(1))
        
        # Check if target contains mask token (-1)
        is_masked = (target == -1.0).all(dim=-1, keepdim=True)  # (B, 1)
        
        # Convert normalized values to [0, 255] for non-masked tokens
        # For masked tokens, we'll use the mask_token embedding
        target_converted = (target * 255 + 1e-2 * torch.randn_like(target)).round().long()
        target_converted = torch.clamp(target_converted, 0, 255)

        # Take middle condition
        cond = cond_list[0]
        if cond.dim() > 2:
            leading = cond.shape[0]
            cond = cond.reshape(leading, -1, cond.shape[-1]).mean(dim=1)
        
        # Use mask token for masked positions, v_codebook for real values
        bsz = target.size(0)
        target_emb = self.v_codebook(target_converted[:, 0:1])  # (B, 1, width)
        mask_emb = self.mask_token.expand(bsz, -1, -1)  # (B, 1, width)
        
        # Select appropriate embedding based on mask
        target_emb = torch.where(is_masked.unsqueeze(-1), mask_emb, target_emb)
        
        x = torch.cat([self.cond_proj(cond).unsqueeze(1), target_emb], dim=1)
        x = self.ln(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            v_logits = self.v_mlm(x[:, 0], self.v_codebook.weight)

        return v_logits, target_converted

    def forward(self, target, cond_list):
        """Training forward pass."""
        logits, target = self.predict(target, cond_list)
        loss_v = self.criterion(logits, target[:, 0])

        if self.training:
            loss = self.v_weight * loss_v
        else:
            # For NLL computation
            loss = loss_v

        loss_mean = loss.mean()
        
        # Compute entropy more memory-efficiently
        # For velocity_loss, logits is (batch, 256), sample subset of batch to save memory
        with torch.no_grad():
            batch_size = logits.size(0)
            subset_size = min(16, batch_size)  # Only compute on subset of batch
            indices = torch.randperm(batch_size, device=logits.device)[:subset_size]
            logits_subset = logits[indices].detach()
            entropy = torch.distributions.Categorical(logits=logits_subset).entropy().mean()
        
        stats = {
            f'level_{self.level_index}/velocity_loss': loss_mean.detach(),
            f'level_{self.level_index}/velocity_entropy': entropy
        }

        return loss_mean, stats

    def sample(self, cond_list, temperature, cfg, filter_threshold=0, _intermediates_list=None, _current_level=None, _patch_pos=None):
        """Sample velocity values (with optional intermediate recording for Level 3)."""
        if cfg == 1.0:
            bsz = cond_list[0].size(0)
        else:
            bsz = cond_list[0].size(0) // 2
        
        # Initialize with -1 (mask token)
        # If cfg != 1.0, we need 2*bsz
        total_bsz = cond_list[0].size(0)
        velocity_values = torch.full((total_bsz, 1), fill_value=-1.0, device=cond_list[0].device)

        logits, _ = self.predict(velocity_values, cond_list)
        
        # Apply temperature
        logits = logits / max(temperature, 1e-8)

        if not cfg == 1.0:
            # CFG Logic: Keep both conditional and unconditional logits for next step?
            # Wait, Velocity Loss level is the FINAL level. It doesn't need to maintain state for next MAR step.
            # But if the caller (MAR generator) expects 2*B samples to update its 2*B canvas, 
            # then we should return 2*B samples?
            
            # Standard CFG usually mixes logits and returns ONE sample (the guided one).
            # logits = uncond + cfg * (cond - uncond) -> This results in 1 set of logits.
            
            # However, MAR generator maintains [Pos, Neg] patches. 
            # If we return 1 mixed sample, we can only update Pos. 
            # Neg (unconditional) part of the canvas will become stale or be updated with mixed result (wrong).
            
            # To maintain "Negative Trajectory" for autoregressive/iterative processes,
            # usually we need to sample independently?
            # But MAR is not exactly a trajectory like Diffusion. 
            # In each step of MAR, we predict MASKED tokens.
            
            # If we mix logits here, we get the "Best" token. 
            # If we return this "Best" token, we can put it in Pos.
            # What do we put in Neg?
            # Neg should be sampled from Unconditional logits.
            
            cond_logits = logits[:bsz]
            uncond_logits = logits[bsz:]
            
            # Mixed Logits (for Positive sample)
            # Filter low probability conditional logits
            cond_probs = torch.softmax(cond_logits, dim=-1)
            mask = cond_probs < filter_threshold
            # Apply filter logic...
            
            mixed_logits = uncond_logits + cfg * (cond_logits - uncond_logits)
            
            # We need to return 2*B samples: [Sample(Mixed), Sample(Uncond)]
            # This ensures the upper level MAR updates its Pos canvas with Mixed result,
            # and Neg canvas with Uncond result (to keep it "purely unconditional" for next steps).
            
            # Sample Positive (Mixed)
            mixed_probs = torch.softmax(mixed_logits, dim=-1) + 1e-10
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
            sampled_pos = torch.multinomial(mixed_probs, num_samples=1).reshape(-1)
            
            # Sample Negative (Unconditional)
            uncond_probs = torch.softmax(uncond_logits, dim=-1) + 1e-10
            uncond_probs = uncond_probs / uncond_probs.sum(dim=-1, keepdim=True)
            sampled_neg = torch.multinomial(uncond_probs, num_samples=1).reshape(-1)
            
            # Combine
            sampled_ids = torch.cat([sampled_pos, sampled_neg], dim=0)
            
            # Convert to [0, 1]
            velocity_values = sampled_ids.float() / 255.0
            velocity_values = velocity_values.view(total_bsz, 1, 1, 1) # (2*B, 1, 1, 1)
            
            return velocity_values

        # CFG == 1.0 case
        # Add numerical stability
        logits = torch.clamp(logits, min=-20, max=20)
        
        # Sample token
        probs = torch.softmax(logits, dim=-1)
        
        # Add small epsilon to avoid numerical issues
        probs = probs + 1e-10
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        sampled_ids = torch.multinomial(probs, num_samples=1).reshape(-1)
        
        # Convert to [0, 1] range
        # 0 -> 0 (silence), 255 -> 1 (max velocity)
        velocity_values[:, 0] = sampled_ids.float() / 255.0

        # Reshape to (bsz, 1, 1, 1) which represents a 1x1 patch
        velocity_values = velocity_values.view(bsz, 1, 1, 1)

        return velocity_values


# ==============================================================================
# Hierarchical FractalGen Model
# ==============================================================================

