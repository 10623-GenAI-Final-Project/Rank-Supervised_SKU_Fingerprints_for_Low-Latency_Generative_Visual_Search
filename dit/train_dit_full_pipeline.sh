#!/usr/bin/env bash
# Full pipeline for DiT fine-tuning on DeepFashion2
# Usage: bash dit/train_dit_full_pipeline.sh [GPU_ID] [NUM_IMAGES]
# Example: bash dit/train_dit_full_pipeline.sh 0 20000

set -euo pipefail

# Parse arguments
GPU_ID="${1:-0}"
TARGET_IMAGES="${2:-20000}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "$PROJECT_ROOT"

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
DATA_DIR="data/dit_training_subset_${TARGET_IMAGES}"
OUTPUT_DIR="checkpoints/dit_lora_${TARGET_IMAGES}"

echo "======================================================================"
echo "DiT Fine-tuning Pipeline for DeepFashion2"
echo "======================================================================"
echo "GPU:                ${CUDA_VISIBLE_DEVICES}"
echo "Project Root:       ${PROJECT_ROOT}"
echo "SKU Root:           ${SKU_ROOT}"
echo "Target Images:      ${TARGET_IMAGES}"
echo "Data Directory:     ${DATA_DIR}"
echo "Output Directory:   ${OUTPUT_DIR}"
echo "======================================================================"
echo ""

# Check if SKU_ROOT exists
if [ ! -d "$SKU_ROOT" ]; then
    echo "❌ Error: SKU_ROOT does not exist: $SKU_ROOT"
    echo "Please set SKU_ROOT environment variable to your DeepFashion2_SKU path"
    echo "Example: export SKU_ROOT=/path/to/DeepFashion2_SKU"
    exit 1
fi

# Step 1: Sample training data
echo "======================================================================"
echo "[1/2] Sampling training images..."
echo "======================================================================"
echo ""

if [ -f "${DATA_DIR}/dit_training_images.txt" ]; then
    echo "⚠️  Data already exists at ${DATA_DIR}"
    read -p "Do you want to re-sample? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$DATA_DIR"
    else
        echo "Skipping data sampling..."
    fi
fi

if [ ! -f "${DATA_DIR}/dit_training_images.txt" ]; then
    python -m dit.prepare_dit_training_subset \
        --sku_root "$SKU_ROOT" \
        --split train \
        --target_images "$TARGET_IMAGES" \
        --output_dir "$DATA_DIR" \
        --seed 42
    
    echo ""
    echo "✓ Data sampling completed"
else
    echo "✓ Using existing data at ${DATA_DIR}"
fi

echo ""

# Step 2: Fine-tune LoRA
echo "======================================================================"
echo "[2/2] Fine-tuning DiT with LoRA..."
echo "======================================================================"
echo ""

python -m dit.train_dit_lora_df2 \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model runwayml/stable-diffusion-v1-5 \
    --rank 8 \
    --lr 1e-4 \
    --epochs 5 \
    --batch_size 4 \
    --grad_accum 2 \
    --save_every 500 \
    --wandb_project "10623-dit-finetune-df2"

echo ""
echo "======================================================================"
echo "✓ Training Pipeline Completed!"
echo "======================================================================"
echo "Model checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Check wandb for training metrics"
echo "  2. Test the fine-tuned model with gen_dit_aug_df2.py"
echo "  3. Update gen_dit_aug_df2.py to load LoRA weights:"
echo "     --sd_model runwayml/stable-diffusion-v1-5"
echo "     --lora_weights ${OUTPUT_DIR}/lora_best.pt"
echo "======================================================================"
echo ""

