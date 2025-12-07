#!/bin/bash
# Inference script for FractalGen MIDI model
# Usage: bash run_inference.sh [checkpoint_path] [config_file]

set -e

echo "========================================================================"
echo "FractalGen MIDI Inference Script"
echo "========================================================================"
echo ""

# Check arguments
if [ -z "$1" ]; then
    echo "Error: Checkpoint path is required"
    echo ""
    echo "Usage: bash run_inference.sh <checkpoint_path> [config_file]"
    echo "Example: bash run_inference.sh outputs/fractalgen/checkpoints/step_00100000.ckpt"
    echo "Example: bash run_inference.sh outputs/fractalgen/checkpoints/step_00100000.ckpt config/inference_default.yaml"
    exit 1
fi

CHECKPOINT="$1"
CONFIG_FILE="${2:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"

# Build command
CMD="python src/inference/inference.py --checkpoint \"$CHECKPOINT\""

# Add config if provided
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    echo "Config: $CONFIG_FILE"
    CMD="$CMD --config \"$CONFIG_FILE\""
else
    echo "Config: Using command line defaults"
    # Add default parameters when no config is provided
    CMD="$CMD --mode unconditional --num_samples 10 --generation_length 256 --save_gif"
fi

echo ""
echo "========================================================================"
echo "Starting inference..."
echo "========================================================================"
echo ""

# Run inference
eval $CMD

echo ""
echo "========================================================================"
echo "Inference complete!"
echo "========================================================================"
echo ""
