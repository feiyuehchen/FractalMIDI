#!/bin/bash
# Inference script for FractalGen MIDI model
# Usage: bash run_inference_fractalgen.sh

# Exit on error
set -e

echo "========================================================================"
echo "FractalGen MIDI Inference Script"
echo "========================================================================"
echo ""

# Activate conda environment (if needed)
# conda activate frac

# Set visible GPU
export CUDA_VISIBLE_DEVICES=2

# Model checkpoint
CHECKPOINT="/home/feiyueh/FractalMIDI/outputs/fractalgen_ar_ar_ar_ar/checkpoints/step_00005000-val_loss_0.0414.ckpt"

# Generation mode
MODE="unconditional"  # Options: unconditional, conditional, inpainting

# Generation parameters
NUM_SAMPLES=10
GENERATION_LENGTH=256

# Sampling parameters
NUM_ITER_LIST="12 8 4 1"  # Iterations per level (4 levels: 128→16→4→1)
CFG=1.0                  # Classifier-free guidance strength
TEMPERATURE=1.0          # Sampling temperature
SPARSITY_BIAS=0.0        # No bias - use raw model output

# Output directory
OUTPUT_DIR="outputs/inference_arararar"

# Save options
SAVE_IMAGES="--save_images"  # Comment out to disable

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Mode: $MODE"
echo "  Num samples: $NUM_SAMPLES"
echo "  Generation length: $GENERATION_LENGTH"
echo "  Iterations: $NUM_ITER_LIST"
echo "  CFG: $CFG"
echo "  Temperature: $TEMPERATURE"
echo "  Sparsity bias: $SPARSITY_BIAS"
echo "  Output: $OUTPUT_DIR"
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
    --num_iter_list $NUM_ITER_LIST \
    --cfg "$CFG" \
    --temperature "$TEMPERATURE" \
    --sparsity_bias "$SPARSITY_BIAS" \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_IMAGES \
    --device cuda:0

echo ""
echo "========================================================================"
echo "Generation complete!"
echo "========================================================================"
echo ""
echo "Generated files saved to:"
echo "  $OUTPUT_DIR/"
echo ""
echo "To generate with different modes:"
echo "  Unconditional: MODE=\"unconditional\""
echo "  Conditional:   MODE=\"conditional\" (also set --condition_midi)"
echo "  Inpainting:    MODE=\"inpainting\" (also set --input_midi --mask_start --mask_end)"
echo ""

