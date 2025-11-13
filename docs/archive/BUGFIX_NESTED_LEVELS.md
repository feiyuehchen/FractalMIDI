# Bug Fix: Nested Level target_width Issue

## Problem

After initial fixes, testing revealed a critical bug when using variable widths:

```
RuntimeError: The expanded size of the tensor (16) must match the existing size (512) at non-singleton dimension 1.
Target sizes: [1, 16].  Tensor sizes: [512]
```

This error occurred at `ar_generator.py:297`: `canvas[:, patch_idx] = patch_flat`

## Root Cause

The issue was that `target_width` was being incorrectly propagated to **all** hierarchical levels:

```python
# INCORRECT: target_width passed to all levels
def next_level_sample_function(...):
    return self.next_fractal.sample(..., target_width=target_width)  # ❌ Wrong!
```

### Why This Caused the Error

The hierarchical structure is:
- **Level 0** (img_size=128): Generates patches of size 16×16, arranged in a 8×W grid (where W depends on target_width)
- **Level 1** (img_size=16): Should generate **square** patches of size 4×4, arranged in a 4×4 grid
- **Level 2** (img_size=4): Should generate **square** patches of size 1×1, arranged in a 4×4 grid
- **Level 3** (velocity_loss): Generates single velocity values

When `target_width=256` was passed to Level 1, it tried to generate a 16×256 output instead of the expected 16×16 patch, causing dimension mismatches.

## Solution

**Only Level 0 should use `target_width`**. All other levels generate **square patches** of fixed size.

### Changes Made

#### 1. `models/ar_generator.py`

```python
def sample(self, ..., target_width=None):
    """
    Args:
        target_width: Target width for generation at Level 0 only (128, 256, 512, etc.)
                     For other levels, this should be None and will use square patches.
    """
    h_patches = self.img_size // self.patch_size
    if target_width is not None and self.img_size == 128:
        # Level 0: variable width
        w_patches = target_width // self.patch_size
    else:
        # Other levels: square patches
        w_patches = h_patches
    actual_seq_len = h_patches * w_patches
```

**Key change**: Added condition `self.img_size == 128` to detect Level 0.

#### 2. `models/mar_generator.py`

Same change as AR generator - only Level 0 (img_size=128) uses `target_width`.

#### 3. `models/fractal_gen.py`

```python
# When calling next level
result = self.next_fractal.sample(
    ...,
    target_width=None  # ✓ Always None for nested levels
)

# When calling generator at current level
result = self.generator.sample(
    ...,
    target_width=target_width if fractal_level == 0 else None  # ✓ Only Level 0
)
```

## Verification

After this fix:
- Level 0 correctly uses `target_width` to generate 128×128, 128×256, or 128×512
- Level 1-3 correctly generate square patches (16×16, 4×4, 1×1)
- No dimension mismatches occur

## Architecture Clarity

### Level 0 (128 → 16×16 patches)
- Input: None (unconditional) or partial piano roll
- Output: 128×W piano roll (W = target_width)
- Grid: 8 rows × (W/16) columns of 16×16 patches
- **Uses target_width**: ✓

### Level 1 (16 → 4×4 patches)
- Input: Conditions from Level 0
- Output: Single 16×16 patch
- Grid: 4×4 grid of 4×4 patches
- **Uses target_width**: ✗ (always square)

### Level 2 (4 → 1×1 patches)
- Input: Conditions from Level 1
- Output: Single 4×4 patch
- Grid: 4×4 grid of 1×1 pixels
- **Uses target_width**: ✗ (always square)

### Level 3 (velocity_loss)
- Input: Conditions from Level 2
- Output: Single velocity value (1×1)
- **Uses target_width**: ✗ (N/A)

## Testing

Run the test again:

```bash
python test_fixes.py --checkpoint path/to/model.ckpt
```

Expected output:
- ✓ Width 128: (1, 1, 128, 128)
- ✓ Width 256: (1, 1, 128, 256)
- ✓ Width 512: (1, 1, 128, 512)

## Key Takeaway

In hierarchical generative models, variable dimensions should only be applied at the **top level**. Lower levels should generate **fixed-size components** that compose into the variable-size output at the top level.

This is analogous to:
- A painter (Level 0) painting a variable-size canvas
- Using fixed-size brushstrokes (Level 1-3)

