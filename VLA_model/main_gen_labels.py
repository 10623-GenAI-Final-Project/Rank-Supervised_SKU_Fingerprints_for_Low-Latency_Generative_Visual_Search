from train.generate_labels import generate_labels
import argparse
import pickle
import open_clip

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
    samples = generate_labels(base_dataset, model_P, clip_model, preprocess)

    with open("labels.pkl", "wb") as f:
        pickle.dump(samples, f)

    print("Saved label samples to labels.pkl")

if __name__ == "__main__":
    main()