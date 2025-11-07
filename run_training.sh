#!/bin/bash
# Training script for FractalGen MIDI model
# Usage: bash run_training.sh

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

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Generator types: "mar" or "ar" for each level (4 levels total)
# - mar: Masked Autoregressive (parallel generation with masking)
# - ar: Autoregressive (sequential generation)
GENERATOR_TYPES="ar,ar,ar,ar"

# Scan order (only applies to AR generators):
# - row_major: Left-to-right, then top-to-bottom (emphasizes temporal continuity)
# - column_major: Top-to-bottom, then left-to-right (emphasizes harmonic structure)
SCAN_ORDER="row_major"

# Mask ratio (only applies to MAR generators):
# - mask_ratio_loc: Mean mask ratio (1.0 = mask 100%, 0.7 = mask 70%, 0.5 = mask 50%)
# - mask_ratio_scale: Standard deviation (higher = more variation in mask ratio)
# Default: loc=1.0, scale=0.5 (same as before, won't affect existing training)
# To make training easier: try loc=0.7 or loc=0.5
MASK_RATIO_LOC=1.0
MASK_RATIO_SCALE=0.5

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
TRAIN_BATCH_SIZE=4        # Adjust based on GPU memory
VAL_BATCH_SIZE=4          # Validation batch size
MAX_STEPS=100000           # Training steps
WARMUP_STEPS=10000
ACCUM_GRAD=8
VAL_INTERVAL_STEPS=500
CHECKPOINT_EVERY_N_STEPS=20000
LOG_IMAGES_EVERY_N_STEPS=10000

# Learning rate configuration
LEARNING_RATE=1e-4
GRAD_CLIP=3.0
WEIGHT_DECAY=0.005

# Data paths
TRAIN_DATA="dataset/train.txt"
VAL_DATA="dataset/valid.txt"

# Output directory (automatically includes generator types and scan order)
OUTPUT_DIR="outputs/fractalgen_${GENERATOR_TYPES//,/_}_${SCAN_ORDER}"

# Hardware configuration
NUM_WORKERS=16
PRECISION="32"  # Options: 32, 16, bf16

# Logging
LOG_EVERY_N_STEPS=50

echo "Configuration:"
echo "  Model: FractalGen (128→16→4→1)"
echo "  Generator Types: $GENERATOR_TYPES"
echo "  Scan Order: $SCAN_ORDER"
echo "  Train batch size: $TRAIN_BATCH_SIZE"
echo "  Val batch size: $VAL_BATCH_SIZE"
echo "  Max steps: $MAX_STEPS"
echo "  Val interval: every $VAL_INTERVAL_STEPS steps"
echo "  Checkpoint interval: every $CHECKPOINT_EVERY_N_STEPS steps"
echo "  Learning rate: $LEARNING_RATE"
echo "  Warmup steps: $WARMUP_STEPS"
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
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --augment_factor 1 \
    --max_steps "$MAX_STEPS" \
    --lr "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_steps "$WARMUP_STEPS" \
    --grad_clip "$GRAD_CLIP" \
    --accumulate_grad_batches "$ACCUM_GRAD" \
    --devices 0,1 \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor 2 \
    --pitch_shift_min -12 \
    --pitch_shift_max 12 \
    --generator_types "$GENERATOR_TYPES" \
    --scan_order "$SCAN_ORDER" \
    --mask_ratio_loc "$MASK_RATIO_LOC" \
    --mask_ratio_scale "$MASK_RATIO_SCALE" \
    --precision "$PRECISION" \
    --val_check_interval_steps "$VAL_INTERVAL_STEPS" \
    --checkpoint_every_n_steps "$CHECKPOINT_EVERY_N_STEPS" \
    --save_top_k -1 \
    --log_every_n_steps "$LOG_EVERY_N_STEPS" \
    --log_images_every_n_steps "$LOG_IMAGES_EVERY_N_STEPS" \
    --output_dir "$OUTPUT_DIR" \

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

