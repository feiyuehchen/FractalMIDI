#!/bin/bash
# Training script for FractalGen MIDI model
# Usage: bash run_training.sh [config_file]

set -e

echo "========================================================================"
echo "FractalGen MIDI Training Script"
echo "========================================================================"
echo ""

# Activate conda environment (if needed)
# conda activate frac

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default config file
CONFIG_FILE="${1:-config/train_default.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Usage: bash run_training.sh [config_file]"
    echo "Example: bash run_training.sh config/train_ar.yaml"
    echo ""
    echo "Available configs:"
    ls -1 config/*.yaml 2>/dev/null || echo "  No config files found in config/"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""
echo "========================================================================"
echo "Starting training..."
echo "========================================================================"
echo ""

# Run training with config file
python main.py --config "$CONFIG_FILE"

echo ""
echo "========================================================================"
echo "Training complete!"
echo "========================================================================"
echo ""
