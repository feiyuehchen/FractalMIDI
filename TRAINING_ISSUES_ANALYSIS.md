# FractalMIDI Training Issues Analysis

## Problem Summary
Generated MIDI files have severe quality issues despite very low validation loss (0.0015).

## Quantitative Analysis

### Generation Quality Issues
| Metric | Training Data | Generated (v2 step 40k) | Issue |
|--------|---------------|------------------------|-------|
| Note density | 4.3 notes/sec | 114-123 notes/sec | **26-28x too dense** |
| Pitch std | 9.21 | 37.0 | **4x too dispersed** |
| Pitch range | 35-78 (43 notes) | 0-127 (128 notes) | **Uses all pitches** |
| Velocity mean | 82.1 | 72-84 | Similar (OK) |

### Training Configuration Issues

#### Issue 1: Dropout Settings Not Applied ✅ FIXED
- **Problem**: Config file set `attn_dropout: 0.1, proj_dropout: 0.1`, but model trained with `0.0`
- **Cause**: `train_utils.py` didn't read dropout from config YAML
- **Impact**: Model severely overfitted (validation loss 0.0015 is suspiciously low)
- **Status**: **FIXED** in commits above

#### Issue 2: Using Full AR Instead of MAR
- **Current**: `generator_types: [ar, ar, ar, ar]`
- **Problem**: Autoregressive (AR) generation is harder to train and more prone to error accumulation
- **Recommendation**: Consider using MAR (Masked Autoregressive) for at least the coarse levels
  - Example: `generator_types: [mar, mar, ar, ar]` or `generator_types: [mar, mar, mar, mar]`

#### Issue 3: Model Architecture Too Small?
- Current: `embed_dim_list: [256, 256, 128, 64]`
- Current: `num_blocks_list: [8, 4, 2, 1]`
- This is 1/2 the size of default config
- **May not have enough capacity** to learn complex musical structures

#### Issue 4: Training Data May Need Review
- Only 1 sample checked: POP909/488/488.mid
- Should verify:
  - Data quality and diversity
  - Proper preprocessing
  - No corrupted files

## Root Cause Analysis

The extremely low validation loss (0.0015) combined with terrible generation quality indicates **severe overfitting**:

1. **Zero dropout** → Model memorized training data pixel-by-pixel
2. **Model can reconstruct training examples perfectly** (low val loss)
3. **But learned no generalizable musical structure** (terrible generation)

This is a classic overfitting pattern: the model became a "compression algorithm" for the training set rather than learning music composition rules.

## Solutions

### Immediate Solution: Retrain with Fixed Config ✅ READY
The config bug has been fixed. Retrain with proper dropout:

```bash
cd /home/feiyueh/FractalMIDI
# Clean old checkpoints to avoid confusion
rm -rf outputs/fractalgen_small/checkpoints/version_2
bash run_training.sh config/train_small.yaml
```

Expected improvements:
- Dropout will prevent overfitting
- Validation loss will be higher (~0.05-0.15 is normal)
- Generation quality should improve significantly

### Alternative Solution: Switch to MAR

Edit `config/train_small.yaml`:
```yaml
model:
  generator_types: [mar, mar, mar, mar]  # Change from [ar, ar, ar, ar]
  mask_ratio_loc: 1.0
  mask_ratio_scale: 0.5
```

MAR advantages:
- More stable training
- Better for learning long-range dependencies
- Used in original FractalGen paper

### Advanced Solution: Increase Model Capacity

If quality is still poor after dropout fix, try larger model:
```yaml
model:
  embed_dim_list: [384, 384, 192, 96]  # 1.5x current size
  num_blocks_list: [10, 6, 3, 2]       # More blocks
```

## Improved Inference Settings ✅ APPLIED

Updated `config/inference_small.yaml` with:
- `temperature: 0.95` (down from 1.2) → More coherent output
- `sparsity_bias: 0.15` (new) → Reduce note density
- `velocity_threshold: 0.25` (up from 0.10) → Filter low-velocity notes

These help but **cannot fix fundamental training issues**.

## Recommended Next Steps

1. ✅ Config bug fixed
2. ✅ Inference config improved
3. ⏳ **RETRAIN** with fixed dropout configuration
4. Monitor training:
   - Validation loss should stabilize around 0.05-0.15 (NOT 0.0015)
   - Check generated samples during training (every 2000 steps)
5. If still poor, try MAR instead of AR
6. If still poor, increase model capacity

## How to Monitor Training

```bash
# View training logs
tensorboard --logdir outputs/fractalgen_small/logs --port 6006

# Check training progress
tail -f outputs/fractalgen_small/logs/version_X/events.out.tfevents.*
```

Look for:
- Validation loss: Should be 0.05-0.15 (not 0.001)
- Training loss: Should decrease gradually
- Generated samples: Check piano rolls in TensorBoard

## Expected Timeline

With fixed config:
- Training: ~11-12 hours for 50k steps (on 2 GPUs)
- Should see reasonable results by 20k steps (~5 hours)
- Can do early inference test at 10k steps

---

Generated: $(date)
Status: Config fixes applied, ready for retraining

