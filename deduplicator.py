#!/usr/bin/env python3
"""
deduplicator_clip_aesthetic.py

Windows + Python script (single-folder scan) which:
 - computes CLIP ViT-L/14 embeddings
 - builds FAISS index (faiss-cpu) or sklearn fallback
 - clusters visually similar photos (neighbor graph + union-find)
 - scores images using HuggingFace aesthetic predictor + sharpness + exposure
 - prints the chosen best photo for each found cluster

Requirements:
    pip install torch torchvision transformers pillow faiss-cpu scikit-learn tqdm opencv-python
"""

import argparse
from pathlib import Path
import os
from typing import List
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from tqdm import tqdm

# Transformers CLIP + Aesthetic predictor
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, AutoModelForImageClassification, AutoImageProcessor

# Optional libs
try:
    import faiss
    _have_faiss = True
except:
    _have_faiss = False

try:
    from sklearn.neighbors import NearestNeighbors
    _have_sklearn = True
except:
    _have_sklearn = False

try:
    import cv2
    _have_cv2 = True
except:
    _have_cv2 = False

# -----------------------------
# Parameters
# -----------------------------
BATCH_SIZE = 32
K_NEIGHBORS = 8
SIMILARITY_THRESHOLD = 0.92
MIN_CLUSTER_SIZE = 1
DEFAULT_INPUT = r"c:\Users\Z004JR9Y\OneDrive - Siemens AG\Dokumenty\msveda\github\tst-foto"

# -----------------------------
# Helpers
# -----------------------------
def list_images_flat(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])

# -----------------------------
# Load models
# -----------------------------
def load_models(device: torch.device):
    print("Loading CLIP ViT-L/14...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("Loading HF aesthetic predictor (cafeai/aesthetic-v2)...")
    aesthetic_processor = AutoImageProcessor.from_pretrained("cafeai/cafe_aesthetic")
    aesthetic_model = AutoModelForImageClassification.from_pretrained("cafeai/cafe_aesthetic")
    aesthetic_model.eval()

    return clip_model, clip_processor, aesthetic_processor, aesthetic_model

# -----------------------------
# Compute CLIP embeddings
# -----------------------------
def compute_clip_embeddings(paths: List[Path], clip_model, clip_processor, device) -> np.ndarray:
    embeddings = []
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding batches"):
            batch_paths = paths[i:i+BATCH_SIZE]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except:
                    imgs.append(None)
            valid_idx = [j for j,img in enumerate(imgs) if img is not None]
            if not valid_idx:
                embeddings.extend([np.zeros(clip_model.config.projection_dim, dtype=np.float32)] * len(batch_paths))
                continue
            batch_imgs = [imgs[j] for j in valid_idx]
            inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype("float32")
            # assemble
            b_emb = [None]*len(batch_paths)
            idx_map = 0
            for j in range(len(batch_paths)):
                if j in valid_idx:
                    b_emb[j] = feats[idx_map]
                    idx_map +=1
                else:
                    b_emb[j] = np.zeros(feats.shape[1], dtype=np.float32)
            embeddings.extend(b_emb)
    return np.vstack(embeddings)

# -----------------------------
# Neighbors index
# -----------------------------
def build_and_query_index(embeddings: np.ndarray, top_k:int=K_NEIGHBORS):
    n, d = embeddings.shape
    if _have_faiss:
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        sims, inds = index.search(embeddings, top_k)
        return inds, sims
    elif _have_sklearn:
        nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
        nn.fit(embeddings)
        dists, inds = nn.kneighbors(embeddings, n_neighbors=top_k)
        sims = 1.0 - dists
        return inds, sims
    else:
        raise RuntimeError("Install faiss-cpu or scikit-learn")

# -----------------------------
# Union-Find clustering
# -----------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self,a):
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]
            a=self.parent[a]
        return a
    def union(self,a,b):
        ra=self.find(a)
        rb=self.find(b)
        if ra==rb: return
        if self.rank[ra]<self.rank[rb]:
            self.parent[ra]=rb
        else:
            self.parent[rb]=ra
            if self.rank[ra]==self.rank[rb]:
                self.rank[ra]+=1

def cluster_by_similarity(inds: np.ndarray, sims: np.ndarray, threshold: float=SIMILARITY_THRESHOLD):
    n=inds.shape[0]
    uf=UnionFind(n)
    for i in range(n):
        for j_idx, neighbor in enumerate(inds[i]):
            neighbor=int(neighbor)
            if neighbor==i: continue
            sim=float(sims[i][j_idx])
            if sim>=threshold:
                uf.union(i, neighbor)
    clusters={}
    for i in range(n):
        r=uf.find(i)
        clusters.setdefault(r,[]).append(i)
    return [c for c in clusters.values() if len(c)>=MIN_CLUSTER_SIZE]

# -----------------------------
# Scorers: aesthetic, sharpness, exposure
# -----------------------------
def compute_aesthetic_hf(img_path: Path, processor, model, device):
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # expected score: sum(prob_i * i)
            score = sum([i*probs[0,i].item() for i in range(probs.shape[1])])
        return float(score)
    except:
        return 0.0

def sharpness_score_path(path: Path) -> float:
    if _have_cv2:
        try:
            im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None: return 0.0
            return float(cv2.Laplacian(im, cv2.CV_64F).var())
        except:
            return 0.0
    else:
        try:
            img = Image.open(path).convert("L").resize((256,256))
            arr = np.asarray(img).astype(np.float32)
            gy, gx = np.gradient(arr)
            return float((gx**2 + gy**2).mean())
        except:
            return 0.0

def exposure_score_path(path: Path) -> float:
    try:
        if _have_cv2:
            im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None: return 0.0
            mean = float(im.mean())/255.0
        else:
            img = Image.open(path).convert("L").resize((256,256))
            arr = np.asarray(img).astype(np.float32)/255.0
            mean = float(arr.mean())
        return 1.0 - min(1.0, abs(mean-0.5)*2.0)
    except:
        return 0.0

def normalize_sharpness(s: float) -> float:
    s=max(0.0,float(s))
    return min(1.0,np.log10(s+1.0)/3.0)

def final_score(idx, emb, path, aesthetic_proc, aesthetic_model, device):
    aest = compute_aesthetic_hf(path, aesthetic_proc, aesthetic_model, device)
    sharp = normalize_sharpness(sharpness_score_path(path))
    expo = exposure_score_path(path)
    return 0.7*aest + 0.25*sharp + 0.05*expo

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Deduplicate photos (CLIP + HF aesthetic)")
    parser.add_argument("--path", type=str, help="Folder with images (flat). If omitted, uses default.")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    args = parser.parse_args()

    folder = Path(args.path) if args.path else Path(DEFAULT_INPUT)
    if not folder.exists():
        print("Input folder does not exist:", folder)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_processor, aesthetic_processor, aesthetic_model = load_models(device)

    image_paths = list_images_flat(folder)
    if not image_paths:
        print("No images found in", folder)
        return
    print(f"Found {len(image_paths)} images.")

    embeddings = compute_clip_embeddings(image_paths, clip_model, clip_processor, device)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0]=1.0
    embeddings = embeddings / norms

    inds, sims = build_and_query_index(embeddings, top_k=args.k)
    clusters = cluster_by_similarity(inds, sims, threshold=args.threshold)
    print(f"Formed {len(clusters)} clusters.")

    kept=[]
    for ci, cluster in enumerate(clusters):
        if len(cluster)==0: continue
        best_score=-1e9
        best_idx=None
        for idx in cluster:
            path=image_paths[idx]
            score=final_score(idx, embeddings[idx], path, aesthetic_processor, aesthetic_model, device)
            if score>best_score:
                best_score=score
                best_idx=idx
        kept.append((best_idx,image_paths[best_idx]))
        print(f"[Cluster {ci}] {len(cluster)} items -> keep: {image_paths[best_idx].name} (score {best_score:.2f})")

    out_file = folder/"kept_best.txt"
    with open(out_file,"w",encoding="utf-8") as f:
        for idx,p in kept:
            f.write(str(p)+"\n")
    print("Kept list written to", out_file)
    print("Done.")

if __name__=="__main__":
    main()
