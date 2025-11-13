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
        # Convert normalized [0, 1] to [0, 255]
        # Add small noise to avoid rounding issues
        target = (target * 255 + 1e-2 * torch.randn_like(target)).round().long()
        target = torch.clamp(target, 0, 255)

        # Take middle condition
        cond = cond_list[0]
        if cond.dim() > 2:
            leading = cond.shape[0]
            cond = cond.reshape(leading, -1, cond.shape[-1]).mean(dim=1)
        x = torch.cat([self.cond_proj(cond).unsqueeze(1), self.v_codebook(target[:, 0:1])], dim=1)
        x = self.ln(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            v_logits = self.v_mlm(x[:, 0], self.v_codebook.weight)

        return v_logits, target

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
        
        # Initialize with 0 (silence) for unconditional generation
        # For training, -1 is used as mask token, but for sampling we start from silence
        velocity_values = torch.zeros((bsz, 1), device=cond_list[0].device)

        if cfg == 1.0:
            logits, _ = self.predict(velocity_values, cond_list)
        else:
            logits, _ = self.predict(
                torch.cat([velocity_values, velocity_values], dim=0),
                cond_list
            )
        
        # Apply temperature
        logits = logits / max(temperature, 1e-8)

        if not cfg == 1.0:
            cond_logits = logits[:bsz]
            uncond_logits = logits[bsz:]

            # Filter low probability conditional logits
            cond_probs = torch.softmax(cond_logits, dim=-1)
            mask = cond_probs < filter_threshold
            uncond_logits[mask] = torch.max(
                uncond_logits,
                cond_logits - torch.max(cond_logits, dim=-1, keepdim=True)[0] + 
                torch.max(uncond_logits, dim=-1, keepdim=True)[0]
            )[mask]

            logits = uncond_logits + cfg * (cond_logits - uncond_logits)

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

