"""
VLA Model Inference Script

This script loads a trained VLA policy model and predicts the best action
for a given input image.
"""

import torch
from PIL import Image
from pathlib import Path
import argparse
import open_clip
import sys
import os

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Try importing with VLA_model prefix first, fallback to direct import
try:
    from VLA_model.model.policy import VLAPolicy
    from VLA_model.Image_processing.image_feature import compute_quality_features
    from VLA_model.Image_processing.image_process import VLAAction
except ImportError:
    # If running from within VLA_model directory
    from model.policy import VLAPolicy
    from Image_processing.image_feature import compute_quality_features
    from Image_processing.image_process import VLAAction


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA Model Inference - Predict action label for an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/soinew/Rank-Supervised_SKU_Fingerprints_for_Low-Latency_Generative_Visual_Search/VLA_model/checkpoint_vla/vla_policy.pt",
        help="Path to VLA policy checkpoint file (default: vla_policy.pt).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="open_clip model name (e.g., ViT-B-16). Must match training configuration.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag. Must match training configuration.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def load_vla_model(checkpoint_path: str, clip_model_name: str, clip_pretrained: str, device: str):
    """
    Load the VLA policy model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        clip_model_name: CLIP model name (must match training)
        clip_pretrained: CLIP pretrained tag (must match training)
        device: Device to load model on
    
    Returns:
        policy_model: Loaded VLAPolicy model
        clip_model: CLIP model for feature extraction
        preprocess: Image preprocessing function
    """
    print(f"Loading VLA model from {checkpoint_path}...")
    
    # Create CLIP model (must match training configuration)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Get dimensions from CLIP model
    visual_dim = clip_model.visual.output_dim
    quality_dim = 10  # Fixed quality feature dimension
    num_actions = len(VLAAction)
    
    # Create VLA policy model
    policy_model = VLAPolicy(
        visual_dim=visual_dim,
        quality_dim=quality_dim,
        num_actions=num_actions
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_model.load_state_dict(checkpoint)
    policy_model.to(device)
    policy_model.eval()
    
    print("Model loaded successfully!")
    print(f"  - Visual dimension: {visual_dim}")
    print(f"  - Quality dimension: {quality_dim}")
    print(f"  - Number of actions: {num_actions}")
    
    return policy_model, clip_model, preprocess


@torch.no_grad()
def predict_action(image_path: str, policy_model: VLAPolicy, clip_model, preprocess, device: str):
    """
    Predict the best action for a given image.
    
    Args:
        image_path: Path to input image
        policy_model: Trained VLA policy model
        clip_model: CLIP model for visual features
        preprocess: Image preprocessing function
        device: Device to run inference on
    
    Returns:
        action: VLAAction enum value (the predicted action)
        action_id: Integer ID of the action
        logits: Raw logits from the model (for debugging)
    """
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    
    # Extract visual features using CLIP
    clip_input = preprocess(img).unsqueeze(0).to(device)
    vfeat = clip_model.encode_image(clip_input)
    vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)  # Normalize
    
    # Extract quality features
    qfeat = compute_quality_features(img)
    qfeat = torch.tensor(qfeat, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get action prediction
    logits = policy_model(vfeat, qfeat)
    action_id = int(logits.argmax(dim=-1).item())
    action = VLAAction(action_id)
    
    return action, action_id, logits


def main():
    args = parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate checkpoint path (try relative to project root if not absolute)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = Path(project_root) / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    policy_model, clip_model, preprocess = load_vla_model(
        str(checkpoint_path),
        args.clip_model,
        args.clip_pretrained,
        str(device)
    )
    
    # Predict action
    print(f"\nPredicting action for image: {image_path}")
    action, action_id, logits = predict_action(
        str(image_path),
        policy_model,
        clip_model,
        preprocess,
        str(device)
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"Predicted Action: {action.name} (ID: {action_id})")
    print("="*60)
    print("\nAll action scores:")
    for i, action_enum in enumerate(VLAAction):
        score = logits[0][i].item()
        marker = " <-- SELECTED" if i == action_id else ""
        print(f"  {action_enum.name:20s}: {score:8.4f}{marker}")
    print("="*60)
    
    return action, action_id


if __name__ == "__main__":
    main()

