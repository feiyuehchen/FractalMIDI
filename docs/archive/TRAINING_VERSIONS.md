# FractalMIDI Training Versions

This document explains the different training configurations for the FractalMIDI model.

## Overview

The FractalMIDI model supports flexible piano roll dimensions, with two primary training versions:

1. **128×128 Version**: Standard square piano roll
2. **256×256 Version**: Extended temporal dimension (actually 128×512 resized)

## Version Details

### 128×128 Version

**Configuration:**
- Input shape: `(128 pitches, 128 time steps)`
- Total tokens: 16,384
- Temporal coverage: 128 × 1/16 notes = 8 measures (at 4/4 time)

**Use case:**
- Short musical phrases
- Motifs and patterns
- Quick generation
- Lower memory footprint

**Training settings:**
```python
crop_length = 128
target_width = 128
```

### 256×256 Version

**Configuration:**
- Visual shape: `(256, 256)` - appears as a square
- Actual representation: Two stacked 128×256 segments
  - Upper half: 128 pitches × first 256 time steps
  - Lower half: 128 pitches × next 256 time steps (continuation)
- Total temporal length: 512 time steps
- Temporal coverage: 512 × 1/16 notes = 32 measures (at 4/4 time)

**Rationale:**
- The 256×256 format allows the model to leverage spatial hierarchies designed for square images
- Internally, this represents a 128×512 piano roll that has been vertically "folded" in half
- The lower 128 rows continue the musical sequence from the upper 128 rows
- This design maximizes temporal modeling while maintaining architectural efficiency

**Use case:**
- Longer musical pieces
- Full phrases with development
- Better capture of musical structure
- More context for generation

**Training settings:**
```python
crop_length = 512
target_width = 512  # Generates 128×512, displayed as 256×256
```

## Data Value Semantics

The model uses a specific value range with semantic meaning:

- **`-1.0`**: Mask token (white) - indicates positions to be predicted during training
- **`0.0`**: Silence (black) - no note at this position
- **`(0, 1]`**: Velocity - normalized MIDI velocity values
  - `0.1` = quiet note (pianissimo)
  - `0.5` = medium velocity (mezzo-forte)
  - `1.0` = loud note (fortissimo)

## Inference Configuration

### For 128×128 Model

```bash
python inference.py \
    --checkpoint path/to/128x128_model.ckpt \
    --target_width 128 \
    --generation_length 128
```

### For 256×256 Model (128×512 actual)

```bash
python inference.py \
    --checkpoint path/to/256x256_model.ckpt \
    --target_width 512 \
    --generation_length 512
```

### Custom Widths

The model supports arbitrary widths (must be divisible by 4):

```bash
# Generate 128×256 (16 measures)
--target_width 256 --generation_length 256

# Generate 128×1024 (64 measures)
--target_width 1024 --generation_length 1024
```

## Architecture Hierarchy

Both versions use the same fractal hierarchy:

- **Level 0**: 128 → patches
- **Level 1**: 16 → patches
- **Level 2**: 4 → patches
- **Level 3**: 1 × 1 pixels (velocity values)

The number of patches at each level scales with `target_width`:
- For width=128: Level 0 has 128÷16 × 128÷16 = 8×8 = 64 patches
- For width=256: Level 0 has 128÷16 × 256÷16 = 8×16 = 128 patches
- For width=512: Level 0 has 128÷16 × 512÷16 = 8×32 = 256 patches

## Generator Types

Each level can use either:
- **MAR** (Masked Autoregressive): Parallel iterative refinement
- **AR** (Autoregressive): Sequential patch-by-patch generation

Training configurations specify generator types per level:
```python
generator_type_list = ("mar", "mar", "mar", "mar")  # All MAR
generator_type_list = ("ar", "ar", "ar", "ar")      # All AR
generator_type_list = ("mar", "mar", "ar", "ar")    # Mixed
```

## Scan Orders for AR

For AR generators, two scanning orders are supported:

### Row Major (Pitch Priority)
```
Scan order: (0,0) → (0,1) → (0,2) → ... → (1,0) → (1,1) → ...
Effect: Generates one pitch at a time across all time steps
```

### Column Major (Time Priority)
```
Scan order: (0,0) → (1,0) → (2,0) → ... → (0,1) → (1,1) → ...
Effect: Generates one time step at a time across all pitches
```

**Note:** Time-major (column_major) is generally better for music as it preserves chords and vertical harmony.

## Training Script Examples

### Training 128×128 Version

```bash
python main.py \
    --train_data dataset/train.txt \
    --train_batch_size 8 \
    --max_steps 200000 \
    --generator_types "mar,mar,mar,mar" \
    --output_dir outputs/fractalgen_128x128
```

The dataset configuration should use:
```python
crop_length = 128
random_crop = True
```

### Training 256×256 Version (128×512)

```bash
python main.py \
    --train_data dataset/train_long.txt \
    --train_batch_size 4 \
    --max_steps 200000 \
    --generator_types "mar,mar,mar,mar" \
    --output_dir outputs/fractalgen_256x256
```

The dataset configuration should use:
```python
crop_length = 512
random_crop = True
```

## Memory Considerations

- **128×128**: ~4GB GPU memory per batch
- **256×256** (512 width): ~8-12GB GPU memory per batch
- **Custom widths**: Memory scales linearly with width

## Best Practices

1. **Match training and inference widths**: Generate at the same width the model was trained on
2. **Pad to multiples of 4**: Ensure `target_width` is divisible by 4 (patch size)
3. **Start with silence**: Model initializes with 0 (silence) for unconditional generation
4. **Use appropriate iterations**: Longer sequences may benefit from more iterations
5. **Temperature tuning**: Lower temperature (0.7-0.9) often produces better structured music

## Future Extensions

The architecture can be extended to support:
- Variable heights (e.g., 88 keys for standard piano)
- Multi-track generation (multiple instruments)
- Conditional generation on tempo, key, style
- Real-time interactive generation

