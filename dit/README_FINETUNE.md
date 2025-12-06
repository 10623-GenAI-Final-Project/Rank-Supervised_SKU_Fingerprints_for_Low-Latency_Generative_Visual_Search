# DiT Fine-tuning for DeepFashion2

Fine-tune Stable Diffusion with LoRA on DeepFashion2 catalog images.

---

## Quick Start

### 1. Setup Environment

```bash
# Set paths
export SKU_ROOT=/path/to/DeepFashion2_SKU
export HF_HOME=/path/to/hf_cache  # optional

# Install dependencies (if not already installed)
pip install -r requirements.txt

```

### 2. Run Training Pipeline

```bash
# Full pipeline: sample data + train LoRA
# Usage: bash dit/train_dit_full_pipeline.sh [GPU_ID] [NUM_IMAGES]
bash dit/train_dit_full_pipeline.sh 0 20000
```

This script will:
1. Sample high-quality catalog images from DeepFashion2_SKU
2. Fine-tune LoRA adapter (rank=8, 5 epochs)
3. Save checkpoints to `checkpoints/dit_lora_20000/`

---

## Training Configurations

| Config | Images | Time | Use Case |
|--------|--------|------|----------|
| Quick test | 10,000 | 1-2h | Fast validation |
| **Recommended**  | 20,000 | 4-6h | Best balance |
| Full training | 50,000 | 10-12h | Maximum quality |

### Exp: RTX 5090 32GB Settings

```bash
python -m dit.train_dit_lora_df2 \
  --data_dir data/dit_training_subset_20000 \
  --output_dir checkpoints/dit_lora \
  --rank 8 \
  --lr 1e-4 \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum 2 \
  --wandb_project "dit-finetune-df2"
```

**Memory usage**: ~18GB (batch_size=4)  
**Speed**: ~3.0 it/s  
**Effective batch size**: 8

---

## Output Checkpoints

Training saves to `checkpoints/dit_lora_20000/`:

```
checkpoints/dit_lora_20000/
├── lora_best.pt        # Best checkpoint (lowest loss)
├── lora_final.pt       # Final checkpoint
├── lora_epoch_*.pt     # Per-epoch checkpoints
└── lora_step_*.pt      # Intermediate checkpoints (every 500 steps)
```

**Recommended**: Use `lora_best.pt`

---


## Troubleshooting

### Out of memory

```bash
# Reduce batch size
python -m dit.train_dit_lora_df2 --batch_size 2 --grad_accum 4

# Or reduce LoRA rank
python -m dit.train_dit_lora_df2 --rank 4

# Or use fewer images
bash dit/train_dit_full_pipeline.sh 0 10000
```

### Loss not decreasing

- Learning rate too small → try `--lr 2e-4`
- Learning rate too large → try `--lr 5e-5`
- Check data quality

### Poor generation quality

- Train more epochs: `--epochs 10`
- Use more data: `bash dit/train_dit_full_pipeline.sh 0 50000`
- Increase LoRA rank: `--rank 16`

---

## Files

- `dit/prepare_dit_training_subset.py` - Data sampling
- `dit/train_dit_lora_df2.py` - LoRA training
- `dit/train_dit_full_pipeline.sh` - Full pipeline
- `gen/gen_dit_aug_df2.py` - Augmentation (needs LoRA support)

---

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Diffusers Docs](https://huggingface.co/docs/diffusers)
