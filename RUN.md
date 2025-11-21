# Prepare Dataset for all
./scripts/prepare_deepfashion2_sku.sh

# Prepare Dataset for Baseline1
./scripts/prepare_df2_reid_splits.sh*

# Train Baseline1
./scripts/train_baseline1_reid.sh

# Eval Baseline1
./scripts/eval_baseline1_reid_val.sh