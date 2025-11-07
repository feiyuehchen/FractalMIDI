# Recent Changes Summary

## 1. Scan Order Feature (2024-11-07)

### What's New
Added support for two different scanning orders in Autoregressive (AR) generation:
- **Row-major** (default): Left-to-right, then top-to-bottom
- **Column-major**: Top-to-bottom, then left-to-right

### Usage
```bash
# In run_training.sh, set:
SCAN_ORDER="row_major"    # or "column_major"

# Or via command line:
python main.py --scan_order row_major ...
```

### Benefits
- **Row-major**: Better for melodic music with temporal continuity
- **Column-major**: Better for harmonic music with chord progressions

### Documentation
See [docs/SCAN_ORDER.md](docs/SCAN_ORDER.md) for detailed explanation and examples.

---

## 2. Model Code Refactoring (2024-11-07)

### What Changed
The monolithic `model.py` (1567 lines) has been refactored into a modular structure:

```
models/
├── __init__.py              # Package exports
├── utils.py                 # Utility functions (~80 lines)
├── attention.py             # Attention modules (~100 lines)
├── blocks.py                # Transformer blocks (~80 lines)
├── mar_generator.py         # MAR generator (~550 lines)
├── ar_generator.py          # AR generator (~300 lines)
├── velocity_loss.py         # Velocity loss (~200 lines)
├── fractal_gen.py           # Main model (~240 lines)
└── generation.py            # Generation functions (~120 lines)
```

### Benefits
1. **Improved Readability**: Each file focuses on a single component
2. **Easier Maintenance**: Changes to one component don't affect others
3. **Better Testing**: Individual modules can be tested in isolation
4. **Clearer Dependencies**: Import statements show relationships
5. **Backward Compatibility**: Existing code continues to work

### Migration
**No changes needed for existing code!** The `model.py` interface provides full backward compatibility:

```python
# This still works
from model import fractalmar_piano
model = fractalmar_piano()
```

For new code, prefer importing from `models` package:
```python
# Better for new code
from models import fractalmar_piano
model = fractalmar_piano()
```

### Documentation
See [docs/MODEL_STRUCTURE.md](docs/MODEL_STRUCTURE.md) for detailed module descriptions.

---

## 3. Simplified Shell Scripts

### What Changed
Consolidated training and inference scripts:
- **Kept**: `run_training.sh` and `run_inference.sh`
- **Removed**: Extra variant scripts (row_major, column_major specific versions)

### Configuration
All options are now configurable via variables at the top of each script:

```bash
# In run_training.sh
GENERATOR_TYPES="ar,ar,ar,ar"  # or "mar,mar,mar,mar"
SCAN_ORDER="row_major"          # or "column_major"
```

---

## Files Added
- `models/__init__.py` - Package exports
- `models/utils.py` - Utility functions
- `models/attention.py` - Attention modules
- `models/blocks.py` - Transformer blocks
- `models/mar_generator.py` - MAR generator
- `models/ar_generator.py` - AR generator
- `models/velocity_loss.py` - Velocity loss
- `models/fractal_gen.py` - Main FractalGen model
- `models/generation.py` - Generation functions
- `docs/SCAN_ORDER.md` - Scan order documentation
- `docs/MODEL_STRUCTURE.md` - Model structure documentation
- `CHANGES_SUMMARY.md` - This file

## Files Modified
- `model.py` - Now a backward compatibility interface
- `main.py` - Added `--scan_order` argument
- `trainer.py` - Added `scan_order` to ModelConfig
- `run_training.sh` - Added SCAN_ORDER configuration
- `README.md` - Updated file structure and usage

## Files Renamed/Backed Up
- `model.py` → `model_old.py` (backup of original monolithic file)

## Testing
All changes have been tested for:
- ✓ Import compatibility
- ✓ Backward compatibility with existing code
- ✓ Trainer functionality
- ✓ Scan order implementation

---

## Quick Start with New Features

### Train with Row-Major Scan (Default)
```bash
# Edit run_training.sh
SCAN_ORDER="row_major"

bash run_training.sh
```

### Train with Column-Major Scan
```bash
# Edit run_training.sh
SCAN_ORDER="column_major"

bash run_training.sh
```

### Use Modular Imports in Your Code
```python
# Import specific components
from models.ar_generator import PianoRollAR
from models.mar_generator import PianoRollMAR
from models import fractalmar_piano

# Create model
model = fractalmar_piano(
    generator_type_list=("ar", "ar", "ar", "ar"),
    scan_order='row_major'
)
```

---

## Questions?
- Scan order: See [docs/SCAN_ORDER.md](docs/SCAN_ORDER.md)
- Model structure: See [docs/MODEL_STRUCTURE.md](docs/MODEL_STRUCTURE.md)
- AR vs MAR: See [docs/AR_vs_MAR.md](docs/AR_vs_MAR.md)

