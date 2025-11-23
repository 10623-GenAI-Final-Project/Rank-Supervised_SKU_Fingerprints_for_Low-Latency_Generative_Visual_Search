import pickle
from train.policy_train import train_policy
from Image_processing.image_process import VLAAction
import argparse
import open_clip
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image-label pairs for VLA"
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",  
        help="open_clip model name (e.g., ViT-B-16).",
    )
    """
    !!! Notice: please keep the model same as what you used for 'main_gen_label' 
    """
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag.",
    
    )

def main():    
    args = parse_args()
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    visual_dim = clip_model.visual.output_dim
    quality_dim = 10
    num_actions = len(VLAAction)

    samples = pickle.load(open("labels.pkl", "rb"))

    model = train_policy(samples, visual_dim, quality_dim, num_actions)

    torch.save(model.state_dict(), "vla_policy.pt")
    print("Saved model to vla_policy.pt")

if __name__ == "__main__":
    main()

