#!/usr/bin/env python3
"""
Inference script for single image SKU prediction using trained ClipSkuBaseline model.

Usage:
    python inference_single_image.py \
        --checkpoint checkpoints/clip_sku_baseline_final.pt \
        --image path/to/your/image.jpg \
        --clip_model ViT-B-16 \
        --clip_pretrained laion2b_s34b_b88k
"""
##/home/soinew/genAIdata/SKU/train/query/000003_02_02/000010_item1.jpg
import argparse
from pathlib import Path
from PIL import Image
import torch
import open_clip

from models.clip_sku_baseline import ClipSkuBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference on a single image using trained ClipSkuBaseline model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default= "data/modelforVLA.pt",
        required=True,
        help="Path to trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="CLIP model name (must match training).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="CLIP pretrained tag (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top SKU predictions to return.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, clip_model_name: str, clip_pretrained: str, device: str):
    """
    Load trained ClipSkuBaseline model from checkpoint.
    
    Returns:
        model: Loaded ClipSkuBaseline model
        sku2idx: Dictionary mapping SKU index to SKU ID (inverse of training's sku2idx)
        args: Training arguments from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args
    args = ckpt.get("args", {})
    num_skus = len(ckpt["sku2idx"])
    sku2idx = ckpt["sku2idx"]  # sku_id (string) -> index (int)
    
    # Create inverse mapping: index -> sku_id
    idx2sku = {idx: sku_id for sku_id, idx in sku2idx.items()}
    
    print(f"Model trained with {num_skus} SKUs")
    print(f"CLIP model: {args.get('clip_model', clip_model_name)}")
    print(f"CLIP pretrained: {args.get('clip_pretrained', clip_pretrained)}")
    
    # Create CLIP model (must match training)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.get("clip_model", clip_model_name),
        pretrained=args.get("clip_pretrained", clip_pretrained)
    )
    
    # Create ClipSkuBaseline model
    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=args.get("freeze_towers", True),
    )
    
    # Load trained weights
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model, idx2sku, preprocess


def predict_sku(image_path: Path, model: ClipSkuBaseline, preprocess, device: str, top_k: int = 5):
    """
    Predict SKU for a single image.
    
    Args:
        image_path: Path to input image
        model: Trained ClipSkuBaseline model
        preprocess: Image preprocessing function
        device: Device to run on
        top_k: Number of top predictions to return
    
    Returns:
        predictions: List of (sku_idx, sku_id, score) tuples, sorted by score
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension: (1, 3, H, W)
    
    # Get image embedding
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)  # (1, D)
        
        # Get all SKU embeddings
        sku_embs = model.sku_embeddings()  # (num_skus, D)
        
        # Get logit scale
        logit_scale = model.logit_scale.exp()
        
        # Compute similarity scores: image embedding @ SKU embeddings^T
        scores = logit_scale * (img_emb @ sku_embs.t())  # (1, num_skus)
        scores = scores.squeeze(0)  # (num_skus,)
    
    # Get top-k predictions
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)), dim=0)
    
    predictions = [
        (int(idx.item()), float(score.item()))
        for idx, score in zip(top_indices, top_scores)
    ]
    
    return predictions


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Check if image exists
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Load model
    model, idx2sku, preprocess = load_model(
        args.checkpoint,
        args.clip_model,
        args.clip_pretrained,
        device
    )
    
    # Predict SKU
    print(f"\nPredicting SKU for image: {args.image}")
    predictions = predict_sku(args.image, model, preprocess, device, args.top_k)
    
    # Print results
    print(f"\nTop {args.top_k} SKU predictions:")
    print("-" * 60)
    for rank, (sku_idx, score) in enumerate(predictions, 1):
        sku_id = idx2sku[sku_idx]
        print(f"Rank {rank}: SKU ID = {sku_id}, Score = {score:.4f}")
    print("-" * 60)
    
    # Return top prediction
    top_sku_idx, top_score = predictions[0]
    top_sku_id = idx2sku[top_sku_idx]
    print(f"\nPredicted SKU: {top_sku_id} (score: {top_score:.4f})")


if __name__ == "__main__":
    main()

