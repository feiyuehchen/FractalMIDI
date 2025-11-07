# Model Structure Documentation

## Overview

The FractalMIDI model code has been refactored into a modular structure for better readability and maintainability. The original `model.py` (1500+ lines) has been split into focused modules in the `models/` package.

## Directory Structure

```
FractalMIDI/
├── model.py                    # Backward compatibility interface (imports from models/)
├── model_old.py                # Original monolithic model file (backup)
└── models/                     # Modular model components
    ├── __init__.py             # Package exports
    ├── utils.py                # Utility functions
    ├── attention.py            # Attention modules
    ├── blocks.py               # Transformer blocks
    ├── mar_generator.py        # Masked Autoregressive generator
    ├── ar_generator.py         # Autoregressive generator
    ├── velocity_loss.py        # Velocity loss layer
    ├── fractal_gen.py          # Main FractalGen model
    └── generation.py           # High-level generation functions
```

## Module Descriptions

### `models/utils.py` (~80 lines)
**Purpose**: Common utility functions

**Contents**:
- `mask_by_order()`: Create mask based on ordering
- `scaled_dot_product_attention()`: Scaled dot product attention with causal masking
- `count_parameters()`: Count trainable parameters

### `models/attention.py` (~100 lines)
**Purpose**: Attention mechanisms

**Contents**:
- `Attention`: Standard bidirectional attention
- `CausalAttention`: Causal (autoregressive) attention

### `models/blocks.py` (~80 lines)
**Purpose**: Transformer building blocks

**Contents**:
- `Block`: Standard Transformer block with bidirectional attention
- `CausalBlock`: Transformer block with causal attention

### `models/mar_generator.py` (~550 lines)
**Purpose**: Masked Autoregressive generator

**Contents**:
- `PianoRollMAR`: Main MAR generator class
  - Parallel generation with masking
  - Variable masking ratios
  - Patchify/unpatchify operations
  - Fast sampling with CFG support

**Key Features**:
- Supports variable-length sequences
- Dynamic padding for non-multiple patch sizes
- Efficient batch processing

### `models/ar_generator.py` (~300 lines)
**Purpose**: Autoregressive generator

**Contents**:
- `PianoRollAR`: Main AR generator class
  - Sequential patch generation
  - Configurable scan order (row-major/column-major)
  - Causal attention masking
  - True autoregressive sampling

**Key Features**:
- Two scanning orders: row-major (temporal) and column-major (harmonic)
- Efficient memory management for hierarchical levels
- Intermediate frame recording for GIF generation

### `models/velocity_loss.py` (~200 lines)
**Purpose**: Final layer for velocity prediction

**Contents**:
- `MlmLayer`: Masked language modeling layer
- `PianoRollVelocityLoss`: Velocity prediction layer
  - 256-level quantization (0-255)
  - Temperature-based sampling
  - Numerical stability improvements

**Key Features**:
- Converts continuous velocity to discrete tokens
- Supports classifier-free guidance
- Handles white background initialization (-1.0)

### `models/fractal_gen.py` (~240 lines)
**Purpose**: Hierarchical FractalGen model

**Contents**:
- `PianoRollFractalGen`: Main hierarchical model
  - Recursive structure (4 levels: 128→16→4→1)
  - Dynamic generator selection (MAR/AR)
  - Dummy condition for unconditional generation
  - Intermediate recording for visualization

- `fractalmar_piano()`: Factory function
  - Default configuration for piano roll generation
  - Configurable generator types and scan order

**Key Features**:
- Hierarchical coarse-to-fine generation
- Flexible architecture (mix MAR and AR at different levels)
- Support for conditional and unconditional generation

### `models/generation.py` (~120 lines)
**Purpose**: High-level generation functions

**Contents**:
- `conditional_generation()`: Generate continuation from prefix
- `inpainting_generation()`: Fill in masked regions

**Key Features**:
- Simplified API for common generation tasks
- Automatic padding and preprocessing
- Support for variable-length inputs

## Usage Examples

### Import from modular structure (recommended for new code)
```python
from models import (
    PianoRollFractalGen,
    fractalmar_piano,
    PianoRollMAR,
    PianoRollAR,
)

# Create model with AR generators and column-major scan
model = fractalmar_piano(
    generator_type_list=("ar", "ar", "ar", "ar"),
    scan_order='column_major'
)
```

### Import from backward compatibility interface (existing code)
```python
from model import fractalmar_piano, PianoRollFractalGen

# Existing code continues to work without changes
model = fractalmar_piano()
```

### Import specific components
```python
# Import only what you need
from models.ar_generator import PianoRollAR
from models.mar_generator import PianoRollMAR
from models.utils import count_parameters

# Use components directly
ar_gen = PianoRollAR(
    seq_len=256,
    patch_size=4,
    cond_embed_dim=512,
    embed_dim=256,
    num_blocks=12,
    num_heads=8,
    scan_order='row_major'
)

print(f"Parameters: {count_parameters(ar_gen):,}")
```

## Benefits of Modular Structure

1. **Improved Readability**: Each file focuses on a single component (~80-550 lines vs 1500+ lines)
2. **Easier Maintenance**: Changes to one component don't affect others
3. **Better Testing**: Individual modules can be tested in isolation
4. **Clearer Dependencies**: Import statements show component relationships
5. **Reusability**: Components can be reused in other projects
6. **Backward Compatibility**: Existing code continues to work via `model.py` interface

## Migration Guide

### For existing code
No changes needed! The `model.py` interface provides full backward compatibility:
```python
# This still works
from model import fractalmar_piano
model = fractalmar_piano()
```

### For new code
Prefer importing from the `models` package:
```python
# Better for new code
from models import fractalmar_piano
model = fractalmar_piano()
```

### For contributors
When modifying model code:
1. Edit the appropriate file in `models/`
2. Test imports: `python -c "from models import fractalmar_piano"`
3. Test backward compatibility: `python -c "from model import fractalmar_piano"`
4. Run existing tests to ensure no regressions

## File Size Comparison

| File | Lines | Purpose |
|------|-------|---------|
| `model_old.py` | 1567 | Original monolithic file |
| `models/utils.py` | 80 | Utilities |
| `models/attention.py` | 100 | Attention modules |
| `models/blocks.py` | 80 | Transformer blocks |
| `models/mar_generator.py` | 550 | MAR generator |
| `models/ar_generator.py` | 300 | AR generator |
| `models/velocity_loss.py` | 200 | Velocity loss |
| `models/fractal_gen.py` | 240 | Main model |
| `models/generation.py` | 120 | Generation functions |
| **Total (modular)** | **1670** | **8 focused files** |

The modular structure has slightly more lines due to:
- Separate docstrings for each module
- Clearer import statements
- Better code organization

But each file is now much easier to read and understand!

## Related Documentation

- [AR vs MAR](AR_vs_MAR.md): Comparison of generator types
- [Scan Order](SCAN_ORDER.md): Explanation of AR scanning orders
- [GIF Generation](GIF_Generation.md): Intermediate frame recording
- [Unconditional Generation](UNCONDITIONAL_GENERATION.md): How generation starts

