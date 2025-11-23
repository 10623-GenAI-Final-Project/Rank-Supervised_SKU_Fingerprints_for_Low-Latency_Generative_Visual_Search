# Prepare Dataset for all
./scripts/prepare_deepfashion2_sku.sh

# Prepare Dataset for Baseline1
./scripts/prepare_df2_reid_splits.sh

# Train Baseline1
./scripts/train_baseline1_reid.sh

# Eval Baseline1
./scripts/eval_baseline1_reid_val.sh

# Train Baseline2/3
./scripts/train_clip_sku_df2.sh

# Eval Baseline2/3
./scripts/eval_clip_sku_df2.sh

# DiT Augmented Multi-view (only inference now)
./scripts/gen_dit_aug_df2.sh