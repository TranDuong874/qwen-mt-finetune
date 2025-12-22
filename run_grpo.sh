#!/bin/bash
set -e  # Exit on error

# =============================================================================
# GRPO Training + Evaluation Script for Remote Server
# =============================================================================

# Configuration
CONFIG="config_grpo.yaml"
OUTPUT_DIR="outputs_grpo"
NUM_GPUS="${NUM_GPUS:-1}"  # Default to 1 GPU, override with NUM_GPUS=2

echo "=============================================="
echo "GRPO Training Pipeline"
echo "=============================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Step 1: GRPO Training
# =============================================================================
echo ""
echo "[1/2] Starting GRPO training..."
echo "----------------------------------------------"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --mixed_precision=bf16 \
        train_grpo.py --config "$CONFIG"
else
    # Single GPU training
    python train_grpo.py --config "$CONFIG"
fi

echo ""
echo "[1/2] GRPO training complete!"
echo "----------------------------------------------"

# =============================================================================
# Step 2: Evaluation
# =============================================================================
echo ""
echo "[2/2] Evaluating final model..."
echo "----------------------------------------------"

# Check if final model exists
FINAL_MODEL="$OUTPUT_DIR/final_model"
if [ ! -d "$FINAL_MODEL" ]; then
    echo "Warning: $FINAL_MODEL not found, trying checkpoints..."
    # Find latest checkpoint
    FINAL_MODEL=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
    if [ -z "$FINAL_MODEL" ]; then
        echo "Error: No model found for evaluation!"
        exit 1
    fi
    echo "Using checkpoint: $FINAL_MODEL"
fi

python evaluate.py \
    --config "$CONFIG" \
    --adapter_model_path "$FINAL_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32

echo ""
echo "[2/2] Evaluation complete!"
echo "----------------------------------------------"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON results yet)"
echo ""

# Show evaluation results if available
RESULTS_FILE="$OUTPUT_DIR/eval_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo "Evaluation Results:"
    cat "$RESULTS_FILE"
fi
