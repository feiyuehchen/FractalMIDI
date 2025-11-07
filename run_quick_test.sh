#!/bin/bash
# Quick test script for FractalGen MIDI model
# Usage: bash run_quick_test.sh

# Exit on error
set -e

echo "========================================================================"
echo "FractalGen MIDI Quick Test (5 epochs)"
echo "========================================================================"
echo ""

# Activate conda environment (if needed)
# conda activate frac

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Quick test configuration
TRAIN_BATCH_SIZE=4  # Smaller batch for faster testing
VAL_BATCH_SIZE=4
MAX_STEPS=2000  # Short run for validation
LEARNING_RATE=1e-4
WARMUP_STEPS=200

# Data paths
TRAIN_DATA="dataset/train.txt"
VAL_DATA="dataset/valid.txt"

# Output directory
OUTPUT_DIR="outputs/fractalgen_test"

echo "Quick Test Configuration:"
echo "  Model: FractalGen (128→16→4→1, ~30M params)"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Val batch size: $VAL_BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "This is a quick validation run to ensure everything works."
echo "For full training, use: bash run_training_fractalgen.sh"
echo ""
echo "========================================================================"
echo "Starting quick test..."
echo "========================================================================"
echo ""

# Run quick test
python main.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --val_check_interval_steps 200 \
    --checkpoint_every_n_steps 400 \
    --save_top_k -1 \
    --lr "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --devices 0,1 \
    --num_workers 2 \
    --prefetch_factor 2 \
    --precision 32 \
    --log_every_n_steps 10 \
    --log_images_every_n_steps 0 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "Quick test complete!"
echo "========================================================================"
echo ""
echo "If training completed successfully, you can proceed with full training:"
echo "  bash run_training_fractalgen.sh"
echo ""
echo "View logs:"
echo "  tensorboard --logdir $OUTPUT_DIR"
echo ""

