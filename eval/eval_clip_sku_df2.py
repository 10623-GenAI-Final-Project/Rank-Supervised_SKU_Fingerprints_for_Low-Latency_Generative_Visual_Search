# eval/eval_clip_sku_df2.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip

import torch.nn.functional as F
from dataset.df2_clip_sku_dataset import (
    DeepFashion2ImageSkuEvalDataset,
)
from models.clip_sku_baseline import ClipSkuBaseline

# Reuse the metrics function from your existing ReID eval.
from eval.eval_reid_df2 import compute_metrics  # type: ignore
from train.train_clip_sku_df2 import build_mean_sku_embeddings, compute_embeddings_clip


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP/SigLIP SKU baseline on DeepFashion2_SKU (no FAISS)."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint from train_clip_sku_df2.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device.",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="K for NDCG@K.",
    )
    parser.add_argument(
        "--recall_ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of K values for Recall@K.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=None,
        help="Optional: override CLIP model name (else load from checkpoint args).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=None,
        help="Optional: override CLIP pretrained tag (else load from checkpoint args).",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="bestshot",
        choices=["bestshot", "mean_sku"],
        help=(
            "Evaluation mode: 'bestshot' uses max over gallery images per SKU "
            "inside compute_metrics; 'mean_sku' uses mean catalog embedding per SKU."
        ),
    )
    parser.add_argument(
        "--image_text_suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix for image_text jsonl filename. "
            "If not set, will fall back to the suffix stored in the checkpoint args."
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sku_root = args.sku_root

    # Load checkpoint first (we may need its args to know the suffix)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sku2idx: Dict[str, int] = ckpt["sku2idx"]
    num_skus = len(sku2idx)

    ckpt_args: Dict = ckpt.get("args", {})

    # Decide which image_text suffix to use:
    #   1) if user passes --image_text_suffix, use that
    #   2) else fall back to what was used during training (stored in ckpt_args)
    suffix = args.image_text_suffix
    if suffix is None:
        suffix = ckpt_args.get("image_text_suffix", "")
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    split_jsonl = sku_root / f"{args.split}_image_text{suffix}.jsonl"

    # Restore clip model/pretrained from checkpoint args (with optional override).
    clip_model_name = args.clip_model or ckpt_args.get("clip_model", "ViT-B-16")
    clip_pretrained_tag = args.clip_pretrained or ckpt_args.get(
        "clip_pretrained", "laion2b_s34b_b88k"
    )

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"clip_model         = {clip_model_name}, pretrained = {clip_pretrained_tag}")
    print(f"num_skus           = {num_skus}")
    print(f"eval_mode          = {args.eval_mode}")
    print(f"image_text_suffix  = '{suffix}'")
    print(f"split_jsonl        = {split_jsonl}")

    # Rebuild CLIP model and preprocess.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained_tag
    )

    # Wrap with ClipSkuBaseline and load weights.
    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Build gallery/query datasets: catalog as gallery, query as query.
    gallery_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="catalog",
    )
    query_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="query",
    )

    print(
        f"[{args.split}] gallery images={len(gallery_ds)}, "
        f"query images={len(query_ds)}"
    )

    # Extract embeddings.
    gallery_embs, gallery_labels, _ = compute_embeddings_clip(
        model, gallery_ds, args.batch_size, device, args.num_workers
    )
    query_embs, query_labels, _ = compute_embeddings_clip(
        model, query_ds, args.batch_size, device, args.num_workers
    )

    gallery_embs = gallery_embs.to(device)
    query_embs = query_embs.to(device)
    gallery_labels = gallery_labels.to(device)
    query_labels = query_labels.to(device)

    # Choose evaluation mode.
    if args.eval_mode == "mean_sku":
        # Per-SKU mean catalog embedding as gallery.
        sku_embs = build_mean_sku_embeddings(
            gallery_embs=gallery_embs,
            gallery_labels=gallery_labels,
            num_skus=num_skus,
        )
        sku_labels = torch.arange(num_skus, device=device, dtype=torch.long)

        metrics = compute_metrics(
            gallery_embs=sku_embs,
            gallery_labels=sku_labels,
            query_embs=query_embs,
            query_labels=query_labels,
            ndcg_k=args.ndcg_k,
            recall_ks=tuple(args.recall_ks),
        )
    else:
        # bestshot: image-level gallery; compute_metrics will aggregate per SKU by max.
        metrics = compute_metrics(
            gallery_embs=gallery_embs,
            gallery_labels=gallery_labels,
            query_embs=query_embs,
            query_labels=query_labels,
            ndcg_k=args.ndcg_k,
            recall_ks=tuple(args.recall_ks),
        )

    print(f"=== CLIP-SKU Evaluation results ({args.split}) ===")
    for k, v in metrics.items():
        if "latency" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
