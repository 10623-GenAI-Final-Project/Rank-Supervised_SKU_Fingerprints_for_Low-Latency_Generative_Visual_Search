#!/usr/bin/env python3
"""
Quick test script for fine-tuned LoRA weights.

Usage:
    python -m dit.test_lora \
        --image path/to/catalog_image.jpg \
        --lora_weights checkpoints/dit_lora_20000/lora_best.pt \
        --output output_finetuned.jpg
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def test_lora(
    image_path: Path,
    lora_weights: Path,
    output_path: Path,
    prompt: str = "a catalog product photo of clothing, studio lighting, high quality, plain background",
    strength: float = 0.3,
    guidance_scale: float = 7.5,
):
    """Test fine-tuned LoRA weights on a single image."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Load LoRA weights
    if lora_weights:
        print(f"Loading LoRA: {lora_weights}")
        pipe.unet.load_attn_procs(str(lora_weights))
    
    # Load input image
    init_image = Image.open(image_path).convert("RGB")
    
    # Generate
    print(f"Generating (strength={strength}, guidance={guidance_scale})...")
    
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=50,
    )
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    
    print(f"✓ Saved: {output_path}")
    
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned LoRA weights"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input catalog image path",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_finetuned.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a catalog product photo of clothing, studio lighting, high quality, plain background",
        help="Generation prompt",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Denoising strength (0.0-1.0)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    
    args = parser.parse_args()
    
    test_lora(
        image_path=Path(args.image),
        lora_weights=Path(args.lora_weights),
        output_path=Path(args.output),
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance,
    )


if __name__ == "__main__":
    main()

