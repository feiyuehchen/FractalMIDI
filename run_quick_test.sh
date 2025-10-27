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
BATCH_SIZE=4  # Smaller batch for faster testing
MAX_EPOCHS=5  # Just 5 epochs for quick validation
LEARNING_RATE=1e-4

# Data paths
TRAIN_DATA="dataset/train.txt"
VAL_DATA="dataset/valid.txt"

# Output directory
OUTPUT_DIR="outputs/fractalgen_test"

echo "Quick Test Configuration:"
echo "  Model: FractalGen (128→16→4→1, ~30M params)"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $MAX_EPOCHS"
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
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --warmup_epochs 2 \
    --devices 0,1 \
    --num_workers 2 \
    --precision 32 \
    --output_dir "$OUTPUT_DIR" \
    --log_every_n_steps 10 \
    --save_top_k 1

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

