#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion with LoRA on DeepFashion2 catalog images.

This script trains a lightweight LoRA adapter for fashion product image generation.
"""

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import wandb
import argparse
import os
from datetime import datetime


class CatalogImageDataset(Dataset):
    """Dataset for catalog images with text prompts."""
    
    def __init__(self, image_list_file, transform=None):
        self.images = []
        self.categories = []
        self.category_ids = []
        
        with open(image_list_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.images.append(parts[0])
                    self.categories.append(parts[1])
                    self.category_ids.append(int(parts[2]) if len(parts) > 2 else 1)
        
        self.transform = transform
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        category = self.categories[idx]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a blank image as fallback
            img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        
        if self.transform:
            img = self.transform(img)
        
        # Simple prompt template for catalog images
        prompt = f"a catalog product photo of {category}, studio lighting, high quality, plain background"
        
        return img, prompt


def train_lora(
    data_dir: Path,
    output_dir: Path,
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    rank: int = 8,
    learning_rate: float = 1e-4,
    num_epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    save_every: int = 500,
    use_wandb: bool = True,
    wandb_project: str = "dit-finetune-df2",
    max_grad_norm: float = 1.0,
):
    """
    Train LoRA adapter for Stable Diffusion on catalog images.
    
    Args:
        data_dir: Directory containing dit_training_images.txt
        output_dir: Directory to save checkpoints
        pretrained_model: Pretrained SD model path
        rank: LoRA rank (4/8/16, higher = more capacity)
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        save_every: Save checkpoint every N steps
        use_wandb: Whether to use wandb for logging
        wandb_project: Wandb project name
        max_grad_norm: Max gradient norm for clipping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if use_wandb:
        run_name = f"lora_r{rank}_lr{learning_rate}_bs{batch_size}x{gradient_accumulation_steps}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "pretrained_model": pretrained_model,
                "rank": rank,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "max_grad_norm": max_grad_norm,
            }
        )
        print(f"Wandb: {wandb.run.url}")
    
    # Load pretrained model
    print(f"\nLoading {pretrained_model}...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=torch.float16,
    )
    
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Freeze base parameters
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Add LoRA layers to UNet
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    
    unet.set_attn_processor(lora_attn_procs)
    
    # Only train LoRA parameters
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.to(device, dtype=torch.float16)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in lora_layers.parameters() if p.requires_grad)
    print(f"LoRA parameters: {trainable_params:,}")
    
    # Move models to GPU
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    unet.to(device, dtype=torch.float16)
    
    # Set eval mode for frozen modules
    vae.eval()
    text_encoder.eval()
    unet.train()
    
    # Prepare dataset
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = CatalogImageDataset(
        image_list_file=data_dir / "dit_training_images.txt",
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Learning rate scheduler (cosine)
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.1
    )
    
    # Training info
    print(f"\nTraining: {len(dataset):,} images, {num_epochs} epochs, BS={batch_size}x{gradient_accumulation_steps}, LR={learning_rate}\n")
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_losses = []
        
        optimizer.zero_grad()
        
        for step, (images, prompts) in enumerate(progress_bar):
            images = images.to(device, dtype=torch.float16)
            
            # Encode text
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), noise.float(), reduction="mean"
            )
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            
            # Optimization step
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Update progress bar
                current_loss = loss.item() * gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })
                
                # Log to wandb
                if use_wandb:
                    wandb.log({
                        "train/loss": current_loss,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + (step / len(dataloader)),
                        "train/step": global_step,
                    }, step=global_step)
                
                # Save checkpoint
                if global_step % save_every == 0:
                    save_path = output_dir / f"lora_step_{global_step}.pt"
                    torch.save(lora_layers.state_dict(), save_path)
                    print(f"\nSaved: {save_path.name}")
                    
                    if use_wandb:
                        wandb.save(str(save_path))
        
        # Epoch summary
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1}: loss={epoch_avg_loss:.4f}, step={global_step}")
        
        # Save epoch checkpoint
        save_path = output_dir / f"lora_epoch_{epoch+1}.pt"
        torch.save(lora_layers.state_dict(), save_path)
        
        if use_wandb:
            wandb.log({
                "epoch/avg_loss": epoch_avg_loss,
                "epoch/number": epoch + 1,
            }, step=global_step)
            wandb.save(str(save_path))
        
        # Save best model
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_path = output_dir / "lora_best.pt"
            torch.save(lora_layers.state_dict(), best_path)
            print(f"  ✓ Best model updated (loss: {best_loss:.4f})")
            
            if use_wandb:
                wandb.save(str(best_path))
    
    # Save final model
    final_path = output_dir / "lora_final.pt"
    torch.save(lora_layers.state_dict(), final_path)
    
    print(f"\n✓ Training completed!")
    print(f"  Best: {output_dir / 'lora_best.pt'} (loss: {best_loss:.4f})")
    print(f"  Final: {final_path.name}")
    
    if use_wandb:
        wandb.save(str(final_path))
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with LoRA on DeepFashion2"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/dit_training_subset",
        help="Data directory containing dit_training_images.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/dit_lora",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank (4/8/16, higher = more capacity but slower)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dit-finetune-df2",
        help="Wandb project name",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    
    args = parser.parse_args()
    
    train_lora(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        pretrained_model=args.model,
        rank=args.rank,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_every=args.save_every,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()

