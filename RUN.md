# Prepare Dataset for all
./scripts/prepare_deepfashion2_sku.sh

# Prepare Dataset for Baseline1
./scripts/prepare_df2_reid_splits.sh

# Train Baseline1
./scripts/train_baseline1_reid.sh

# Eval Baseline1
./scripts/eval_baseline1_reid_val.sh

# Train Baseline2/3/4
./scripts/train_clip_sku_df2.sh

# Eval Baseline2/3/4
./scripts/eval_clip_sku_df2.sh

# DiT Augmented Multi-view (only inference now)
# ./scripts/gen_dit_aug_df2.sh
./scripts/gen_dit_aug_df2_light.sh 3

# Train Baseline5
./scripts/train_baseline.only4.sh 5