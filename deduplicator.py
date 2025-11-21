import os
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import argparse

from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# 1. Load CLIP model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# -----------------------------
# 2. Compute CLIP embeddings
# -----------------------------
def compute_clip_embeddings(image_paths: List[Path]):
    embeddings = []

    for path in tqdm(image_paths, desc="Embedding images"):
        try:
            image = Image.open(path).convert("RGB")
        except:
            print(f"Warning: failed to load {path}")
            embeddings.append(np.zeros(512))
            continue

        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)

        emb = emb.cpu().numpy().flatten()
        emb = emb / np.linalg.norm(emb)  # normalize for cosine similarity
        embeddings.append(emb)

    return np.vstack(embeddings)


# -----------------------------
# 3. Cluster embeddings using DBSCAN
# -----------------------------
def cluster_embeddings(embeddings: np.ndarray, sim_threshold: float = 0.12):
    """
    DBSCAN uses a distance metric. Cosine distance = 1 - cosine similarity.
    For burst photos, distances < 0.12 are usually appropriate.
    """
    dbscan = DBSCAN(
        eps=sim_threshold,        # smaller = tighter clusters
        min_samples=2,
        metric="cosine"
    )

    labels = dbscan.fit_predict(embeddings)
    return labels


# -----------------------------
# 4. Group images by cluster ID
# -----------------------------
def group_by_cluster(image_paths: List[Path], labels: np.ndarray) -> Dict[int, List[Path]]:
    clusters = {}
    for path, label in zip(image_paths, labels):
        if label == -1:
            # -1 = noise, treat as single-photo cluster
            clusters.setdefault(-1, []).append(path)
        else:
            clusters.setdefault(label, []).append(path)
    return clusters


# -----------------------------
# 5. Full pipeline
# -----------------------------
def find_similar_photos(input_folder: str):
    folder = Path(input_folder)

    image_paths = sorted(
        [p for p in folder.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    print(f"Found {len(image_paths)} images")

    # 1) Compute embeddings
    embeddings = compute_clip_embeddings(image_paths)

    # 2) Cluster
    labels = cluster_embeddings(embeddings)

    # 3) Group results
    clusters = group_by_cluster(image_paths, labels)

    return clusters


# -----------------------------
# 6. Main function with path argument
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Find visually similar photos using CLIP.")
    parser.add_argument("--path", type=str, help="Folder with photos")

    args = parser.parse_args()

    # Default path if not provided
    default_path = r"c:\Users\Z004JR9Y\OneDrive - Siemens AG\Dokumenty\msveda\github\tst-foto"

    input_path = args.path if args.path else default_path
    print(f"Using input path: {input_path}")

    clusters = find_similar_photos(input_path)

    print("\n=== CLUSTERS ===")
    for cid, items in clusters.items():
        print(f"\nCluster {cid} ({len(items)} images):")
        for p in items:
            print("   ", p)


if __name__ == "__main__":
    main()