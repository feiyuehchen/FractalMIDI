#!/bin/bash
# Training script for FractalGen MIDI model
# Usage: bash run_training_fractalgen.sh

# Exit on error
set -e

echo "========================================================================"
echo "FractalGen MIDI Training Script"
echo "========================================================================"
echo ""

# Activate conda environment (if needed)
# conda activate frac

# Set visible GPUs (use first two 4090s)
export CUDA_VISIBLE_DEVICES=0,1

# Model configuration
MODEL_SIZE="small"  # Options: small, base, large
BATCH_SIZE=4        # Adjust based on GPU memory
MAX_EPOCHS=50       # Training epochs

# Learning rate configuration
LEARNING_RATE=1e-4
WARMUP_EPOCHS=2
GRAD_CLIP=3.0

# Data paths
TRAIN_DATA="dataset/train.txt"
VAL_DATA="dataset/valid.txt"

# Output directory
OUTPUT_DIR="outputs/fractalgen_${MODEL_SIZE}"

# Hardware configuration
NUM_WORKERS=16
PRECISION="32"  # Options: 32, 16, bf16

# Logging
LOG_EVERY_N_STEPS=50

echo "Configuration:"
echo "  Model: FractalGen (128→16→4→1)"
echo "  Batch size: $BATCH_SIZE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Precision: $PRECISION"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "========================================================================"
echo "Starting training..."
echo "========================================================================"
echo ""

# Run training
python main.py \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --grad_clip "$GRAD_CLIP" \
    --devices 0,1 \
    --num_workers "$NUM_WORKERS" \
    --precision "$PRECISION" \
    --output_dir "$OUTPUT_DIR" \
    --log_every_n_steps "$LOG_EVERY_N_STEPS" \
    --save_top_k -1

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo ""
echo "View logs with:"
echo "  tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "Checkpoints saved to:"
echo "  $OUTPUT_DIR/checkpoints/"
echo ""

