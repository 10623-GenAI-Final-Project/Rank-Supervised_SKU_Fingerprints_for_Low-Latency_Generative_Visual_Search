#!/usr/bin/env python3
"""
Sample a subset of catalog images from DeepFashion2_SKU for DiT fine-tuning.

This script selects high-quality catalog images with balanced category distribution.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import shutil
import argparse
from tqdm import tqdm


def sample_catalog_images_for_dit_training(
    sku_root: Path,
    split: str = "train",
    target_images: int = 20000,
    output_dir: Path = None,
    seed: int = 42,
):
    """
    Sample catalog images from DeepFashion2 SKU for DiT fine-tuning.
    
    Strategy:
    1. Only select catalog domain (shop images)
    2. Balanced sampling across categories
    3. Prioritize images with no occlusion and frontal viewpoint
    
    Args:
        sku_root: Root directory of DeepFashion2_SKU
        split: Dataset split (train/validation/test)
        target_images: Target number of images to sample
        output_dir: Output directory for sampled data
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"\n[DiT Sampling] Target: {target_images} images from {split} split")
    
    # Load metadata
    meta_path = sku_root / f"{split}_sku_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    with meta_path.open("r") as f:
        meta = json.load(f)
    
    # Group all catalog images by category
    category_images = defaultdict(list)
    for sku_id, sku_info in tqdm(meta["skus"].items(), desc="Processing SKUs"):
        category = sku_info["category_name"]
        category_id = sku_info["category_id"]
        
        for entry in sku_info.get("catalog", []):
            # Skip DiT-augmented images
            if entry.get("dit_aug", False):
                continue
            
            img_path = sku_root / entry["crop_path"]
            if not img_path.exists():
                continue
            
            # Calculate quality score (prefer high quality)
            quality_score = 0
            if entry.get("occlusion", 3) == 1:  # no occlusion
                quality_score += 2
            if entry.get("viewpoint", 3) == 2:  # frontal
                quality_score += 2
            if entry.get("scale", 1) >= 2:      # medium/large scale
                quality_score += 1
            
            category_images[category].append({
                "path": img_path,
                "category": category,
                "category_id": category_id,
                "sku_id": sku_id,
                "quality": quality_score,
                "occlusion": entry.get("occlusion", 3),
                "viewpoint": entry.get("viewpoint", 3),
                "scale": entry.get("scale", 1),
            })
    
    # Statistics
    total_available = sum(len(imgs) for imgs in category_images.values())
    num_categories = len(category_images)
    
    if total_available < target_images:
        print(f"⚠️  Warning: Only {total_available} available, adjusting target")
        target_images = total_available
    
    print(f"Found {total_available} images across {num_categories} categories")
    
    # Balanced sampling per category
    per_category = target_images // num_categories
    
    sampled_images = []
    for category, imgs in category_images.items():
        # Sort by quality score, prioritize high quality
        imgs_sorted = sorted(imgs, key=lambda x: x["quality"], reverse=True)
        
        # Sample from top half (high quality images)
        pool_size = max(per_category * 2, len(imgs_sorted))
        pool = imgs_sorted[:pool_size]
        
        # Sample
        n_sample = min(per_category, len(pool))
        sampled = random.sample(pool, n_sample)
        sampled_images.extend(sampled)
    
    # If not enough, randomly supplement
    if len(sampled_images) < target_images:
        all_imgs = [img for imgs in category_images.values() for img in imgs]
        remaining = [img for img in all_imgs if img not in sampled_images]
        need = target_images - len(sampled_images)
        if remaining:
            sampled_images.extend(random.sample(remaining, min(need, len(remaining))))
    
    # Limit to target and shuffle
    sampled_images = sampled_images[:target_images]
    random.shuffle(sampled_images)
    
    print(f"✓ Sampled {len(sampled_images)} images\n")
    
    # Save to output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image path list
        list_file = output_dir / "dit_training_images.txt"
        with list_file.open("w") as f:
            for img in sampled_images:
                f.write(f"{img['path']}\t{img['category']}\t{img['category_id']}\n")
        
        # Copy images
        img_dir = output_dir / "images"
        img_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(tqdm(sampled_images, desc="Copying images")):
            src = img["path"]
            dst = img_dir / f"{i:06d}_{img['category_id']:02d}_{img['sku_id']}.jpg"
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                if i < 5:  # Only print first few errors
                    print(f"⚠️  Failed to copy {src}: {e}")
        
        # Save metadata
        meta_file = output_dir / "metadata.json"
        metadata = {
            "sku_root": str(sku_root),
            "split": split,
            "target_images": target_images,
            "actual_images": len(sampled_images),
            "seed": seed,
            "num_categories": num_categories,
        }
        with meta_file.open("w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved to: {output_dir}")
    
    return sampled_images


def main():
    parser = argparse.ArgumentParser(
        description="Sample catalog images for DiT fine-tuning"
    )
    parser.add_argument(
        "--sku_root",
        type=str,
        required=True,
        help="Root directory of DeepFashion2_SKU",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (train/validation/test)",
    )
    parser.add_argument(
        "--target_images",
        type=int,
        default=20000,
        help="Target number of catalog images to sample",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/dit_training_subset",
        help="Output directory for sampled data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    sample_catalog_images_for_dit_training(
        sku_root=Path(args.sku_root),
        split=args.split,
        target_images=args.target_images,
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
    
    print("✓ Data preparation completed!\n")


if __name__ == "__main__":
    main()

