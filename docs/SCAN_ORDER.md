# Scan Order Configuration

## Overview

The FractalMIDI model now supports two different scanning orders for Autoregressive (AR) generation:
1. **Row-major** (default): Left-to-right, then top-to-bottom
2. **Column-major**: Top-to-bottom, then left-to-right

This feature allows you to control how the AR model generates patches sequentially, which can affect the musical structure and coherence of the generated output.

## Scanning Order Visualization

For a 4x3 patch grid (4 rows, 3 columns):

### Row-Major Order (Default)
```
0  1  2
3  4  5
6  7  8
9  10 11
```
The model generates patches in this order:
- Row 0: patches 0, 1, 2 (left to right)
- Row 1: patches 3, 4, 5 (left to right)
- Row 2: patches 6, 7, 8 (left to right)
- Row 3: patches 9, 10, 11 (left to right)

**Musical Interpretation**: This order emphasizes **temporal continuity** within each pitch range. The model sees complete horizontal "slices" of time before moving to the next pitch range.

### Column-Major Order
```
0  4  8
1  5  9
2  6  10
3  7  11
```
The model generates patches in this order:
- Column 0: patches 0, 1, 2, 3 (top to bottom)
- Column 1: patches 4, 5, 6, 7 (top to bottom)
- Column 2: patches 8, 9, 10, 11 (top to bottom)

**Musical Interpretation**: This order emphasizes **harmonic structure** at each time step. The model sees complete vertical "slices" of pitches before moving to the next time step.

## Usage

### Training

In `run_training.sh`:
```bash
SCAN_ORDER="row_major"  # or "column_major"
python main.py \
    --generator_types "ar,ar,ar,ar" \
    --scan_order "$SCAN_ORDER" \
    ...
```

### Command Line
```bash
# Row-major (default)
python main.py --scan_order row_major ...

# Column-major
python main.py --scan_order column_major ...
```

### Python API
```python
from model import fractalmar_piano

# Row-major
model = fractalmar_piano(
    generator_type_list=("ar", "ar", "ar", "ar"),
    scan_order='row_major'
)

# Column-major
model = fractalmar_piano(
    generator_type_list=("ar", "ar", "ar", "ar"),
    scan_order='column_major'
)
```

## When to Use Each Order

### Row-Major (Default)
**Best for:**
- Melodic music with clear temporal progression
- Music where horizontal continuity is important
- When you want the model to "think" in terms of time evolution

**Characteristics:**
- Generates complete time segments before moving to next pitch range
- Better at maintaining temporal coherence
- May produce more "flowing" melodies

### Column-Major
**Best for:**
- Harmonic music with chord progressions
- Music where vertical structure (harmony) is important
- When you want the model to "think" in terms of harmonic relationships

**Characteristics:**
- Generates complete harmonic snapshots before moving forward in time
- Better at maintaining harmonic coherence
- May produce more structured chord progressions

## Implementation Details

### Internal Representation
The piano roll is divided into patches, and each patch is assigned a spatial coordinate `(h, w)`:
- `h`: Height index (pitch dimension)
- `w`: Width index (time dimension)

The scanning order determines the mapping between:
- **Step index** (0, 1, 2, ...): The order in which patches are generated
- **Spatial coordinates** (h, w): The position in the piano roll

### Conversion Functions
The `PianoRollAR` class includes three helper methods:

1. `_get_scan_order(h_patches, w_patches)`: Returns a list of `(h, w)` tuples in generation order
2. `_spatial_to_linear(h, w, h_patches, w_patches)`: Converts spatial coordinates to linear index
3. `_linear_to_spatial(idx, h_patches, w_patches)`: Converts linear index to spatial coordinates

### Compatibility
- **MAR (Masked Autoregressive)**: Scan order does NOT apply to MAR generators, as they generate all patches simultaneously with masking.
- **Checkpoints**: Models trained with different scan orders are NOT compatible. You must retrain if you change the scan order.
- **Inference**: The scan order used during inference must match the scan order used during training.

## Examples

### Training Two Models
```bash
# Train row-major model
SCAN_ORDER="row_major"
OUTPUT_DIR="outputs/fractalgen_ar_row_major"
python main.py --scan_order row_major --output_dir "$OUTPUT_DIR" ...

# Train column-major model
SCAN_ORDER="column_major"
OUTPUT_DIR="outputs/fractalgen_ar_column_major"
python main.py --scan_order column_major --output_dir "$OUTPUT_DIR" ...
```

### Comparing Outputs
After training both models, you can compare their outputs:
```bash
# Generate with row-major model
python inference.py \
    --checkpoint outputs/fractalgen_ar_row_major/checkpoints/last.ckpt \
    --output_dir outputs/inference_row_major

# Generate with column-major model
python inference.py \
    --checkpoint outputs/fractalgen_ar_column_major/checkpoints/last.ckpt \
    --output_dir outputs/inference_column_major
```

## Technical Notes

1. **Canvas Update**: During generation, the canvas is updated at position `patch_idx`, which is computed from the spatial coordinates `(h, w)` using `_spatial_to_linear()`.

2. **Condition Extraction**: When predicting the next patch, the condition is extracted from the same `patch_idx` position in the condition tensor.

3. **Intermediate Recording**: For GIF generation, the intermediate frames are recorded based on the step index, not the spatial position.

4. **Position Embedding**: The position embedding in the transformer is based on the linear index, so different scan orders will result in different position embeddings for the same spatial location.

## Future Extensions

Possible future enhancements:
- **Spiral order**: Start from center and spiral outward
- **Random order**: Randomize the order during training for more robust generation
- **Learned order**: Let the model learn the optimal scanning order
- **Hierarchical order**: Different orders at different fractal levels

## References

- Original FractalGen paper uses row-major order implicitly
- Column-major order is inspired by PixelCNN variants that use vertical scanning
- The choice of scanning order is an active research topic in autoregressive image generation

