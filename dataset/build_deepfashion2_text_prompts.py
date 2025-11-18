# dataset/build_deepfashion2_text_prompts.py

import argparse
import json
from pathlib import Path


def occlusion_prefix(occlusion: int) -> str:
    """Return a short prefix describing occlusion level, or empty string."""
    if occlusion == 2:
        return "partially occluded "
    if occlusion == 3:
        return "heavily occluded "
    return ""  # occlusion == 1 or unknown


def viewpoint_suffix(viewpoint: int) -> str:
    """
    Return a short suffix describing viewpoint.

    1 = no wear, 2 = frontal viewpoint, 3 = side or back viewpoint.
    For viewpoint == 1 we handle text separately in build_prompt.
    """
    if viewpoint == 2:
        return "from the front"
    if viewpoint == 3:
        return "from the side or back"
    return ""


def build_prompt(category_name: str, domain: str,
                 occlusion: int, viewpoint: int) -> str:
    """
    Build a text prompt for one crop, using category, occlusion and viewpoint.

    domain: "catalog" or "query".
    occlusion: 1 = slight/none, 2 = medium, 3 = heavy.
    viewpoint: 1 = no wear, 2 = frontal, 3 = side/back.
    """
    occ = occlusion_prefix(occlusion)
    view = viewpoint_suffix(viewpoint)

    if domain == "catalog":
        if viewpoint == 1:
            return f"A catalog product photo of a {occ}{category_name} that is not being worn."
        if view:
            return f"A catalog product photo of a {occ}{category_name} {view}."
        return f"A catalog product photo of a {occ}{category_name}."

    elif domain == "query":
        if viewpoint == 1:
            return f"A user photo of a {occ}{category_name} that is not being worn."
        if view:
            return f"A user photo of a person wearing a {occ}{category_name} {view}."
        return f"A user photo of a person wearing a {occ}{category_name}."

    else:
        # Fallback for unexpected domains.
        if viewpoint == 1:
            return f"A photo of a {occ}{category_name} that is not being worn."
        if view:
            return f"A photo of a {occ}{category_name} {view}."
        return f"A photo of a {occ}{category_name}."


def process_split(split: str, sku_root: Path):
    meta_path = sku_root / f"{split}_sku_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    skus = meta["skus"]

    out_jsonl = sku_root / f"{split}_image_text.jsonl"
    num_entries = 0

    with open(out_jsonl, "w") as f_out:
        for sku_id, info in skus.items():
            pair_id = info["pair_id"]
            style = info["style"]
            category_id = info["category_id"]
            category_name = info["category_name"]

            for domain in ["catalog", "query"]:
                for entry in info[domain]:
                    crop_path = entry["crop_path"]
                    orig_image_path = entry["orig_image_path"]
                    image_id = entry["image_id"]
                    item_idx = entry["item_idx"]
                    bbox = entry["bbox"]
                    occlusion = int(entry.get("occlusion", 1))
                    viewpoint = int(entry.get("viewpoint", 1))

                    text = build_prompt(category_name, domain, occlusion, viewpoint)

                    record = {
                        "split": split,
                        "sku_id": sku_id,
                        "pair_id": pair_id,
                        "style": style,
                        "category_id": category_id,
                        "category_name": category_name,
                        "domain": domain,
                        "image_path": crop_path,         # relative to sku_root
                        "orig_image_path": orig_image_path,  # relative to df2_root
                        "image_id": image_id,
                        "item_idx": item_idx,
                        "bbox": bbox,
                        "occlusion": occlusion,
                        "viewpoint": viewpoint,
                        "text": text,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    num_entries += 1

    print(f"[{split}] wrote {num_entries} image-text records to {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Build image-text pairs for DeepFashion2 SKU crops."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        default=Path("data/DeepFashion2_SKU"),
        help="Root directory where SKU crops and *_sku_metadata.json are stored.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to process.",
    )

    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.sku_root)


if __name__ == "__main__":
    main()
