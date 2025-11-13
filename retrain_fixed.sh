#!/bin/bash
# Retrain with fixed dropout configuration
# This script will start training version_3 with proper dropout settings

set -e

echo "========================================================================"
echo "FractalMIDI Retraining with Fixed Configuration"
echo "========================================================================"
echo ""
echo "Changes applied:"
echo "  ✓ Fixed dropout configuration (0.1 instead of 0.0)"
echo "  ✓ Fixed v_weight and num_conds reading from config"
echo "  ✓ Improved inference settings (sparsity_bias, velocity_threshold)"
echo ""
echo "This will start training version_3 with proper regularization."
echo "Expected validation loss: 0.05-0.15 (not 0.001 like before)"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frac

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Run training
cd /home/feiyueh/FractalMIDI
echo ""
echo "Starting training..."
echo ""
python main.py --config config/train_small.yaml

echo ""
echo "========================================================================"
echo "Training script completed!"
echo "========================================================================"
echo ""

