#!/usr/bin/env python3
"""
Test script for fine-tuned LoRA weights (supports both peft and legacy formats).

Usage:
    # For peft format (directory with adapter_model.safetensors)
    python -m dit.test_lora \
        --image path/to/catalog_image.jpg \
        --lora_weights checkpoints/dit_lora_100k/lora_best \
        --output output_finetuned.jpg

    # For comparison without LoRA
    python -m dit.test_lora \
        --image path/to/catalog_image.jpg \
        --output output_baseline.jpg
"""

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def test_lora(
    image_path: Path,
    lora_weights: Path = None,
    output_path: Path = Path("output.jpg"),
    prompt: str = "a catalog product photo of clothing, studio lighting, high quality, plain background",
    strength: float = 0.3,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
):
    """Test fine-tuned LoRA weights on a single image."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print("DiT LoRA Test")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Input image: {image_path}")
    if lora_weights:
        print(f"LoRA weights: {lora_weights}")
    else:
        print("LoRA weights: None (using baseline SD)")
    print(f"{'='*70}\n")
    
    print("Loading Stable Diffusion v1.5...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to(device)
    
    # Load LoRA weights (supports peft format)
    if lora_weights:
        lora_path = Path(lora_weights)
        
        # Check if it's a peft directory (contains adapter_model.safetensors)
        if lora_path.is_dir() and (lora_path / "adapter_model.safetensors").exists():
            print(f"Loading peft LoRA from: {lora_path}")
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_path))
            print("✓ Peft LoRA loaded")
        # Legacy .pt format
        elif lora_path.suffix == ".pt":
            print(f"Loading legacy LoRA from: {lora_path}")
            pipe.unet.load_attn_procs(str(lora_path))
            print("✓ Legacy LoRA loaded")
        else:
            raise ValueError(
                f"Unsupported LoRA format. Expected:\n"
                f"  - Directory with adapter_model.safetensors (peft)\n"
                f"  - .pt file (legacy)\n"
                f"Got: {lora_path}"
            )
    
    # Load input image
    print(f"\nLoading input image...")
    init_image = Image.open(image_path).convert("RGB")
    print(f"  Image size: {init_image.size}")
    
    # Generate
    print(f"\nGenerating image...")
    print(f"  Prompt: {prompt}")
    print(f"  Strength: {strength}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Inference steps: {num_inference_steps}")
    
    result = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path, quality=95)
    
    print(f"\n✓ Output saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned LoRA weights (peft or legacy format)"
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
        default=None,
        help="Path to LoRA weights (directory with adapter_model.safetensors or .pt file). "
             "If not provided, uses baseline SD without LoRA.",
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
        help="Denoising strength (0.0-1.0, lower=more similar to input)",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (higher=more faithful to prompt)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    
    args = parser.parse_args()
    
    test_lora(
        image_path=Path(args.image),
        lora_weights=Path(args.lora_weights) if args.lora_weights else None,
        output_path=Path(args.output),
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
    )


if __name__ == "__main__":
    main()

