#!/bin/bash
# Tuned inference script to compensate for checkpoint-code mismatch
# Use this with old checkpoints trained before the fixes

set -e

echo "========================================================================"
echo "FractalGen MIDI Inference - Tuned Parameters"
echo "========================================================================"
echo ""

export CUDA_VISIBLE_DEVICES=2

# Model checkpoint
CHECKPOINT="/home/feiyueh/FractalMIDI/outputs/fractalgen_ar_ar_ar_ar_row_major/checkpoints/step_00100000-val_loss_0.0010.ckpt"

# Generation mode
MODE="unconditional"

# Visualization
SAVE_GIF="--save_gif"

# Generation parameters
NUM_SAMPLES=10
GENERATION_LENGTH=256
TARGET_WIDTH=256

# Sampling parameters - TUNED for old checkpoint
NUM_ITER_LIST="20 12 8 2"  # More iterations for better quality
CFG=1.0
TEMPERATURE=0.7  # Lower temperature for more deterministic output
SPARSITY_BIAS=0.5  # Add sparsity bias to reduce note density
VELOCITY_THRESHOLD=0.2  # Higher threshold to filter weak notes

# Output
OUTPUT_DIR="outputs/inference_tuned"
SAVE_IMAGES="--save_images"

echo "Configuration (TUNED):"
echo "  Checkpoint: $CHECKPOINT"
echo "  Mode: $MODE"
echo "  Num samples: $NUM_SAMPLES"
echo "  Generation length: $GENERATION_LENGTH"
echo "  Target width: $TARGET_WIDTH"
echo "  Iterations: $NUM_ITER_LIST (increased)"
echo "  CFG: $CFG"
echo "  Temperature: $TEMPERATURE (lowered)"
echo "  Sparsity bias: $SPARSITY_BIAS (added to reduce density)"
echo "  Velocity threshold: $VELOCITY_THRESHOLD (increased)"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Note: These tuned parameters compensate for checkpoint-code mismatch"
echo "      For best results, retrain with the new code."
echo ""
echo "========================================================================"
echo "Starting generation..."
echo "========================================================================"
echo ""

# Run inference
python inference.py \
    --checkpoint "$CHECKPOINT" \
    --mode "$MODE" \
    --num_samples "$NUM_SAMPLES" \
    --generation_length "$GENERATION_LENGTH" \
    --target_width "$TARGET_WIDTH" \
    --num_iter_list $NUM_ITER_LIST \
    --cfg "$CFG" \
    --temperature "$TEMPERATURE" \
    --sparsity_bias "$SPARSITY_BIAS" \
    --velocity_threshold "$VELOCITY_THRESHOLD" \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_IMAGES \
    $SAVE_GIF \
    --device cuda:0

echo ""
echo "========================================================================"
echo "Generation complete!"
echo "========================================================================"
echo ""
echo "Generated files saved to: $OUTPUT_DIR/"
echo ""
echo "Compare with original:"
echo "  Original (dense): outputs/inference_arararar_row_major/"
echo "  Tuned (sparse):   $OUTPUT_DIR/"
echo ""
echo "If results still not good, consider retraining with new code."
echo "========================================================================"
echo ""

