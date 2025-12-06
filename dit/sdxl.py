import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import matplotlib.pyplot as plt
import wandb
import os

def load_pipeline():
    print("Loading SDXL Img2Img model...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    print("Model loaded.\n")
    return pipe

def edit_image(pipe, input_path, output_path,
               strength=0.15, steps=30, guidance=5.0):

    assert os.path.exists(input_path), f"Image not found: {input_path}"

    init_image = Image.open(input_path).convert("RGB")

    prompt = (
        "a photo of the same clothing item, slightly different lighting, "
        "subtle change of background, high image quality"
    )
    negative_prompt = (
        "change shape, wrong pattern, artifacts, distortion, deformed clothing"
    )

    print(f"Running SDXL Img2Img on {input_path} ...")

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance,
        num_inference_steps=steps,
    )

    edited = result.images[0]
    edited.save(output_path)

    print(f"Saved edited image to: {output_path}")

    return init_image, edited

if __name__ == "__main__":

    INPUT_IMAGE = "../data/DeepFashion2_SKU/train/catalog/000001_item_1.jpg"
    # INPUT_IMAGE = "../data/DeepFashion2_original/train/image/000001.jpg"
    OUTPUT_IMAGE = "sdxl_output.jpg"

    wandb.init(
        project="sdxl-img2img-demo",
        name="deepfashion2_sdxl_test",
        config={
            "strength": 0.15,
            "steps": 30,
            "guidance": 5.0,
            "input_image": INPUT_IMAGE
        }
    )

    pipe = load_pipeline()

    orig, edited = edit_image(
        pipe,
        input_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        strength=wandb.config["strength"],
        steps=wandb.config["steps"],
        guidance=wandb.config["guidance"],
    )

    wandb.log({
        "original_image": wandb.Image(orig, caption="Original"),
        "edited_image": wandb.Image(edited, caption="Edited by SDXL"),
    })

    artifact = wandb.Artifact("sdxl_edit", type="image-edit")
    artifact.add_file(OUTPUT_IMAGE)
    wandb.log_artifact(artifact)

    print("\nUpload finished!\n")
    wandb.finish()
