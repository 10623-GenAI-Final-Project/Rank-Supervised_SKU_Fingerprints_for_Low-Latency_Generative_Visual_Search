# Rank-Supervised SKU Fingerprints for Low-Latency Generative Visual Search

Final project for **10-623 Generative AI (CMU)**.

We aim to build a practical **fashion visual search system** where a user can:
- upload a noisy phone photo (user / consumer image), or  
- type a short text description,

and the system retrieves the **exact product SKU** from a catalog with **low latency** (a single embedding lookup in FAISS plus a small amount of pre-processing).

The core idea is to learn a single, high-quality **SKU fingerprint vector** per product, trained directly under retrieval supervision on **DeepFashion2**.

---

## 1. Problem Setting

Given:
- A set of **catalog images** (shop photos) and **query images** (user photos) from DeepFashion2.
- A set of **text queries** derived from metadata (category, viewpoint, occlusion, etc.).

We want to support:
1. **Image → SKU retrieval** (user photo → product)
2. **Text → SKU retrieval** (short text → product)
3. Fast FAISS search with **one vector per SKU** (“SKU-per-vector”).

This is different from many existing systems that store several vectors per product, which increases index size and slows down search.

---

## 2. Dataset: DeepFashion2 and SKU Definition

We use **DeepFashion2 (DF2)** as the only dataset for both training and evaluation.

Each DF2 image has:
- image-level labels: `source` (shop or user), `pair_id`
- item-level labels: `style`, `category_id`, `category_name`, `bounding_box`, `occlusion`, `viewpoint`, etc.

We define a **visual SKU** as:

\[
  \text{SKU} = (\text{pair\_id}, \text{style} > 0, \text{category\_id})
\]

- Items with `style = 0` are discarded (cannot form positive commercial–consumer pairs).
- All crops from images with `source = "shop"` are treated as **catalog** images.
- All crops from images with `source = "user"` are treated as **query** images.
- SKUs are split into train / validation / test following the original DF2 split, so no SKU leaks across splits.

### 2.1 Preprocessing Pipeline

All preprocessing is driven by a single script:

```bash
./scripts/prepare_deepfashion2_sku.sh
```

This script:

1. Reads the original DF2 data under `data/DeepFashion2_original/` (configurable via `DF2_ROOT`).
2. For each split (train/validation/test):
   - Crops each clothing item using its bounding box.
   - Groups crops into **SKUs** based on `(pair_id, style, category_id)`.
   - Saves cropped images into (configurable via `SKU_ROOT`):
     - `<SKU_ROOT>/<split>/catalog/<sku_id>/*.jpg`
     - `<SKU_ROOT>/<split>/query/<sku_id>/*.jpg`
   - Writes a JSON file `<split>_sku_metadata.json` with all SKU-level metadata
     (including `occlusion` and `viewpoint` for each crop).
3. Reads the SKU metadata and generates image–text pairs stored in
   `<split>_image_text.jsonl`, where each line describes a single crop:

   ```json
   {
     "split": "train",
     "sku_id": "010001_01_01",
     "pair_id": 1,
     "style": 1,
     "category_id": 1,
     "category_name": "short sleeve top",
     "domain": "query",
     "image_path": "train/query/010001_01_01/000001_item1.jpg",
     "orig_image_path": "train/image/000001.jpg",
     "image_id": "000001",
     "item_idx": 1,
     "bbox": [x1, y1, x2, y2],
     "occlusion": 3,
     "viewpoint": 3,
     "text": "A user photo of a person wearing a heavily occluded short sleeve top from the side or back."
   }
   ```

These JSONL files are the main entry point for training vision-language models.

---

## 3. Model Overview

Our system consists of four main components:

1. **Baseline CLIP / SigLIP retriever**  
   - A standard dual-encoder model for image–text retrieval.
   - Provides an initial image encoder and text encoder.
   - Used as the starting point for both catalog and query embeddings.

2. **Generative DiT for catalog view augmentation (offline)**  
   - A diffusion transformer (DiT) generates additional catalog views:
     - small viewpoint changes,
     - mild background edits,
     - optional “counterfactual” edits as hard negatives.
   - All generation is done offline; **no DiT calls at query time**.
   - Generated views are filtered by similarity checks to ensure SKU identity is preserved.

3. **Rank-Supervised SKU Fingerprints**  
   - Each SKU has many catalog view embeddings (real + DiT-generated).
   - For each view, we evaluate retrieval performance on held-out user queries when that **single view** represents the SKU.
   - The best-performing view becomes a **teacher fingerprint**.
   - A small aggregation module (e.g. attention or MLP) is trained to map all view embeddings to a single **SKU fingerprint vector**, using:
     - distillation toward the best-shot teacher embedding, and
     - direct retrieval loss (contrastive, margin-based, or ranking loss).
   - The FAISS index stores only these compact SKU fingerprint vectors.

4. **VLA-style Query Pre-processing Policy (optional)**  
   - For user images, we consider a discrete set of pre-processing actions
     (e.g., smart crop, background cleanup, denoise, color correction).
   - Offline, for each query we evaluate all actions and label the one that
     yields the best retrieval rank.
   - A small vision-based policy network is trained to predict the best action.
   - At inference, we apply at most one action before encoding the query image,
     keeping latency overhead small.

---

## 4. Repository Structure

A simplified view of the repository:

```text
.
├── dataset/
│   ├── build_deepfashion2_sku_crops.py   # DF2 → SKU crops + metadata
│   ├── build_deepfashion2_text_prompts.py# DF2 → image-text JSONL
│   └── ...                               # Dataset / dataloader code
├── models/
│   ├── clip_baseline.py                  # CLIP / SigLIP baseline
│   ├── sku_fingerprint_head.py           # SKU aggregation / distillation
│   ├── dit_generator.py                  # DiT for catalog view synthesis
│   └── vla_policy.py                     # Query pre-processing policy (optional)
├── train/
│   ├── train_retriever.py                # Train CLIP-style retriever
│   ├── train_sku_fingerprints.py         # Train SKU fingerprint head
│   ├── train_vla_policy.py               # Train query pre-processing policy
│   └── ...
├── scripts/
│   ├── prepare_deepfashion2_sku.py       # One-command DF2 preprocessing
│   ├── eval_retrieval.py                 # Compute Recall@K, mAP, etc.
│   └── faiss_index_utils.py              # Build / query FAISS index
├── RUN.md                                # Detailed preprocessing instructions
└── README.md                             # (this file)
```

*(Some file names may differ slightly from the actual implementation; check the code for the exact naming.)*

---

## 5. How to Get Started

### 5.1 Environment

Create a conda environment (example):

```bash
conda create -n 10623_final_proj python=3.10
conda activate 10623_final_proj

pip install -r requirements.txt
```

### 5.2 Prepare DeepFashion2 and SKU data

1. Download and unzip DeepFashion2 into `DF2_ROOT` (or any
   directory of your choice).

2. Run the preprocessing script from the project root:

```bash
chmod +x ./scripts/prepare_deepfashion2_sku.sh

DF2_ROOT=/path/to/DeepFashion2_original SKU_ROOT=/path/to_your_generated/DeepFashion2_SKU SPLITS="train validation test" ./scripts/prepare_deepfashion2_sku.sh
```

3. After this step you should see:
   - Cropped images under `${SKU_ROOT}/<split>/{catalog,query}/<sku_id>/*.jpg`
   - Metadata files: `${SKU_ROOT}/<split>_sku_metadata.json`
   - Image–text files: `${SKU_ROOT}/<split>_image_text.jsonl`

### 5.3 Train the baseline retriever

```bash
python train/train_retriever.py   --data_root data/DeepFashion2_SKU   --train_split train   --val_split validation   --model_name clip-base   --output_dir outputs/retriever_clip_base
```

### 5.4 Train SKU fingerprint head

```bash
python train/train_sku_fingerprints.py   --data_root data/DeepFashion2_SKU   --retriever_ckpt outputs/retriever_clip_base/best.pt   --output_dir outputs/sku_fingerprints
```

(The exact flags may vary; check the corresponding training scripts.)

---

## 6. Evaluation

We evaluate the system on DeepFashion2 using:

- **Recall@K** and **mAP** for:
  - query image → SKU retrieval,
  - text → SKU retrieval.
- Latency metrics:
  - time per query for encoding + FAISS search,
  - size of the FAISS index (number of vectors × dimension).

Baselines to compare against:

1. CLIP/SigLIP with multiple catalog embeddings per SKU.
2. CLIP/SigLIP with naive average pooling over catalog views.
3. Our **rank-supervised SKU fingerprints** with one vector per SKU.

---

## 7. Acknowledgements

- DeepFashion2 dataset authors for providing a rich consumer–to–shop
  dataset with detailed annotations.
- OpenAI / Google for CLIP / SigLIP models and open-sourced codebases
  that inspired the retriever baseline.
- The 10-623 teaching staff and classmates for feedback on this project.

---

## 8. License

This project is for educational and research purposes as part of
10-623 Generative AI at CMU. Please check the DeepFashion2
license and any pretrained model licenses before using this code in
commercial applications.
