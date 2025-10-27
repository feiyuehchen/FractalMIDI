#!/bin/bash
# Single GPU test to isolate DDP issues
# Usage: bash run_single_gpu_test.sh

set -e

echo "========================================================================"
echo "Single GPU Test (Isolate DDP issues)"
echo "========================================================================"
echo ""

export CUDA_VISIBLE_DEVICES=0

echo "Configuration:"
echo "  Model: FractalGen (128→16→4→1)"
echo "  Batch size: 4"
echo "  Epochs: 2"
echo "  GPU: 0 (single)"
echo ""
echo "This test uses only 1 GPU to check if the issue is DDP-related."
echo ""
echo "========================================================================"
echo "Starting single GPU test..."
echo "========================================================================"
echo ""

python main.py \
    --train_data dataset/train.txt \
    --val_data dataset/valid.txt \
    --batch_size 4 \
    --max_epochs 2 \
    --lr 1e-4 \
    --warmup_epochs 1 \
    --devices 0 \
    --num_workers 2 \
    --precision 32 \
    --output_dir outputs/single_gpu_test \
    --log_every_n_steps 10

echo ""
echo "========================================================================"
echo "Single GPU test complete!"
echo "========================================================================"
echo ""
echo "If this works, the issue is DDP-related."
echo "If this fails, the issue is in the model or data pipeline."
echo ""

