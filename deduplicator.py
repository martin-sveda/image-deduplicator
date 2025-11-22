#!/usr/bin/env python3
"""
best_photo_fullpower.py

Full-power local "ChatGPT-like" best-photo selector (family shots), single-folder, non-destructive.

Features:
 - time-based grouping (EXIF DateTimeOriginal or mtime)
 - CLIP ViT-L/14 embeddings + FAISS/sklearn neighbor graph clustering
 - PickScore human-preference model (via imscore) or CLIP-proxy fallback
 - MUSIQ TF-Hub technical quality
 - facenet-pytorch: MTCNN face detection + InceptionResnetV1 face embeddings
 - OpenCV Haar for eye/smile heuristics + fer for emotion (happiness) detection
 - Weighted scoring prioritizing faces + human preference
 - Disk caching of heavy computations (optional)
"""

import argparse
from pathlib import Path
import os
import sys
import math
import pickle
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import time

import numpy as np
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from tqdm import tqdm

# Transformers CLIP
from transformers import CLIPModel, CLIPProcessor

# Face models
from facenet_pytorch import MTCNN, InceptionResnetV1

# Emotion detector
try:
    from fer import FER
except ImportError:
    from fer.fer import FER


# TF-Hub MUSIQ
import tensorflow as tf
import tensorflow_hub as hub

# Optional libs: faiss, sklearn, cv2, imscore
_have_faiss = False
_have_sklearn = False
_have_cv2 = False
_have_imscore = False
try:
    import faiss
    _have_faiss = True
except Exception:
    _have_faiss = False

try:
    from sklearn.neighbors import NearestNeighbors
    _have_sklearn = True
except Exception:
    _have_sklearn = False

try:
    import cv2
    _have_cv2 = True
except Exception:
    _have_cv2 = False

# PickScore via imscore (optional)
PickScorer = None
try:
    # try the likely import paths
    try:
        from imscore.preference.model import PickScorer as _PickScorer
    except Exception:
        from imscore import PickScorer as _PickScorer  # fallback
    PickScorer = _PickScorer
    _have_imscore = True
except Exception:
    PickScorer = None
    _have_imscore = False

# -------------------------
# Configurable parameters
# -------------------------
DEFAULT_INPUT = r"c:\Users\Z004JR9Y\OneDrive - Siemens AG\Dokumenty\msveda\github\tst-foto"
TIME_WINDOW_SECONDS = 120        # group photos taken within X seconds
CLIP_BATCH = 16
K_NEIGHBORS = 6
SIMILARITY_THRESHOLD = 0.92
MIN_CLUSTER_SIZE = 1

# Scoring weights (tweak to taste)
WEIGHT_PICK = 0.45
WEIGHT_FACE = 0.30
WEIGHT_MUSIQ = 0.15
WEIGHT_SHARP = 0.10

# MUSIQ TF-HUB model URL (paq2piq recommended)
MUSIQ_HUB = "https://tfhub.dev/google/musiq/paq2piq/1"

# cache paths
CACHE_DIR = Path(".bestphoto_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Haar cascades
if _have_cv2:
    HAAR_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
    HAAR_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"
else:
    HAAR_EYE = HAAR_SMILE = None

# -------------------------
# Utilities
# -------------------------
def list_images_flat(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])

def read_exif_datetime(p: Path) -> Optional[datetime]:
    try:
        img = Image.open(p)
        exif = img._getexif()
        if not exif:
            return None
        dto_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == 'DateTimeOriginal':
                dto_tag = k
                break
        dto = exif.get(dto_tag) or exif.get(36867) or exif.get(306)
        if not dto:
            return None
        return datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None

def file_mtime_datetime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime)

# -------------------------
# Model loaders
# -------------------------
def load_clip(device: torch.device):
    print("Loading CLIP ViT-L/14...")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip.eval()
    return clip, proc

def load_musiq():
    print("Loading MUSIQ (TF-Hub)...")
    return hub.load(MUSIQ_HUB)

def load_face_models(device: torch.device):
    print("Loading face models (MTCNN + InceptionResnetV1)...")
    mtcnn = MTCNN(keep_all=True, device=device if device.type!='cpu' else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

def load_pickscore(device: torch.device):
    """
    Try to load PickScorer via imscore. If not available, return None and script uses CLIP-proxy fallback.
    """
    if not _have_imscore:
        print("imscore (PickScorer) not installed — will use CLIP-proxy fallback.")
        return None
    try:
        # Example: PickScorer.from_pretrained("RE-N-Y/pickscore") if implemented.
        # Many imscore versions expose a convenience loader.
        # Try multiple common APIs gracefully.
        try:
            model = PickScorer.from_pretrained("RE-N-Y/pickscore")
        except Exception:
            # fallback: direct instantiation may work
            model = PickScorer()
        model.to(device)
        model.eval()
        print("Loaded PickScorer.")
        return model
    except Exception as e:
        print("Failed to init PickScorer, falling back to proxy. Error:", e)
        return None

# -------------------------
# CLIP embeddings (batched) + caching
# -------------------------
def compute_clip_embeddings(paths: List[Path], clip_model, clip_proc, device, cache_key: Optional[str]=None) -> np.ndarray:
    # caching: key by list hash if provided
    if cache_key:
        cache_file = CACHE_DIR / f"emb_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    emb = pickle.load(f)
                if emb.shape[0] == len(paths):
                    return emb
            except Exception:
                pass

    clip_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), CLIP_BATCH), desc="CLIP embedding batches"):
            batch = paths[i:i+CLIP_BATCH]
            imgs = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(None)
            valid_idx = [j for j,im in enumerate(imgs) if im is not None]
            if not valid_idx:
                # zero vectors for missing
                d = clip_model.config.projection_dim
                embeddings.extend([np.zeros(d, dtype=np.float32)] * len(batch))
                continue
            batch_imgs = [imgs[j] for j in valid_idx]
            inputs = clip_proc(images=batch_imgs, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype("float32")
            d = feats.shape[1]
            outb = [None]*len(batch)
            idx_map = 0
            for j in range(len(batch)):
                if j in valid_idx:
                    outb[j] = feats[idx_map]
                    idx_map += 1
                else:
                    outb[j] = np.zeros(d, dtype=np.float32)
            embeddings.extend(outb)
    emb_arr = np.vstack(embeddings)
    if cache_key:
        try:
            with open(CACHE_DIR / f"emb_{cache_key}.pkl", "wb") as f:
                pickle.dump(emb_arr, f)
        except Exception:
            pass
    return emb_arr

# -------------------------
# Neighbor search & clustering
# -------------------------
def build_and_query_index(embeddings: np.ndarray, top_k: int = K_NEIGHBORS):
    n,d = embeddings.shape
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

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a
    def union(self, a, b):
        ra = self.find(a); rb = self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1

def cluster_by_similarity(inds: np.ndarray, sims: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> List[List[int]]:
    n = inds.shape[0]
    uf = UnionFind(n)
    for i in range(n):
        for j_idx, neighbor in enumerate(inds[i]):
            neighbor = int(neighbor)
            if neighbor == i:
                continue
            sim = float(sims[i][j_idx])
            if sim >= threshold:
                uf.union(i, neighbor)
    clusters = {}
    for i in range(n):
        r = uf.find(i)
        clusters.setdefault(r, []).append(i)
    return [c for c in clusters.values() if len(c) >= MIN_CLUSTER_SIZE]

# -------------------------
# MUSIQ scoring (TF-Hub) with caching
# -------------------------
def load_musiq():
    print("Loading MUSIQ TF-Hub model...")
    return hub.load(MUSIQ_HUB)

def musiq_score(musiq_model, path: Path, cache_key: Optional[str]=None) -> float:
    # cache by path name if requested
    if cache_key:
        cache_file = CACHE_DIR / f"musiq_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    d = pickle.load(f)
                if path.name in d:
                    return d[path.name]
            except Exception:
                pass
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        out = musiq_model(arr)
        val = float(tf.convert_to_tensor(out).numpy().flatten()[0])
    except Exception:
        val = 0.0
    if cache_key:
        try:
            cache_file = CACHE_DIR / f"musiq_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    d = pickle.load(f)
            else:
                d = {}
            d[path.name] = val
            with open(cache_file, "wb") as f:
                pickle.dump(d, f)
        except Exception:
            pass
    return val

# -------------------------
# Face detection + face-level signals
# -------------------------
def detect_faces_and_signals(path: Path, mtcnn: MTCNN, resnet: InceptionResnetV1, fer_detector: FER, cache_key: Optional[str]=None):
    """
    Returns list of dicts per face:
      {box, embedding, sharpness, eyes_count, smile_haar (0/1), happy (0..1)}
    """
    # caching optional
    if cache_key:
        cfile = CACHE_DIR / f"faces_{cache_key}.pkl"
        if cfile.exists():
            try:
                d = pickle.load(open(cfile,"rb"))
                if path.name in d:
                    return d[path.name]
            except Exception:
                pass

    out = []
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return out

    # detect boxes
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        # cache empty
        if cache_key:
            try:
                d = {}
                if cfile.exists():
                    d = pickle.load(open(cfile,"rb"))
                d[path.name] = out
                pickle.dump(d, open(cfile,"wb"))
            except Exception:
                pass
        return out

    faces = []
    for b in boxes:
        x1,y1,x2,y2 = [int(round(v)) for v in b]
        x1=max(0,x1); y1=max(0,y1)
        try:
            face = img.crop((x1,y1,x2,y2)).resize((160,160))
        except Exception:
            continue
        faces.append(((x1,y1,x2,y2), face))

    if not faces:
        if cache_key:
            try:
                d = {}
                if cfile.exists():
                    d = pickle.load(open(cfile,"rb"))
                d[path.name] = out
                pickle.dump(d, open(cfile,"wb"))
            except Exception:
                pass
        return out

    # compute embeddings
    import torchvision.transforms as T
    trans = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    batch = torch.stack([trans(f[1]).to(resnet.device) for f in faces])
    with torch.no_grad():
        embs = resnet(batch).cpu().numpy()

    # Haar cascades for eyes/smile
    eye_cascade = cv2.CascadeClassifier(HAAR_EYE) if HAAR_EYE else None
    smile_cascade = cv2.CascadeClassifier(HAAR_SMILE) if HAAR_SMILE else None

    for i, ((x1,y1,x2,y2), face_img) in enumerate(faces):
        face_np = np.array(face_img.convert("RGB"))
        if _have_cv2:
            gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) if eye_cascade is not None else []
            smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=22) if smile_cascade is not None else []
            eyes_count = len(eyes)
            smile_haar = 1.0 if len(smiles) > 0 else 0.0
        else:
            sharpness = 0.0; eyes_count = 0; smile_haar = 0.0
        # FER emotion detection
        try:
            em = fer_detector.detect_emotions(face_np)
            if em and len(em)>0 and isinstance(em[0], dict):
                emotions = em[0]["emotions"]
                happy = float(emotions.get("happy", 0.0))
            else:
                happy = 0.0
        except Exception:
            happy = 0.0

        emb = embs[i] if i < len(embs) else np.zeros(512, dtype=float)
        out.append({
            "box": (x1,y1,x2,y2),
            "embedding": emb,
            "sharpness": sharpness,
            "eyes": eyes_count,
            "smile_haar": smile_haar,
            "happy": happy
        })

    # cache per-path
    if cache_key:
        try:
            d = {}
            if cfile.exists():
                d = pickle.load(open(cfile,"rb"))
            d[path.name] = out
            pickle.dump(d, open(cfile,"wb"))
        except Exception:
            pass

    return out

# -------------------------
# Helpers: normalize
# -------------------------
def normalize_sharpness(s: float) -> float:
    s = max(0.0, float(s))
    return min(1.0, math.log10(s + 1.0) / 3.0)

def face_quality_score(faces: List[Dict]) -> float:
    if not faces:
        return 0.0
    per = []
    for f in faces:
        eyes_score = min(2, f.get("eyes",0))/2.0
        smile_score = max(f.get("smile_haar",0.0), f.get("happy",0.0))
        sharp_n = normalize_sharpness(f.get("sharpness",0.0))
        s = 0.4*eyes_score + 0.45*sharp_n + 0.15*smile_score
        per.append(s)
    avg = float(np.mean(per))
    count = len(per)
    count_factor = min(1.0, 0.6 + 0.1*count)
    return avg * count_factor

# -------------------------
# PickScore / fallback
# -------------------------
def compute_pickscore_for_paths(pick_model, clip_model, clip_proc, device, paths: List[Path], cache_key: Optional[str]=None) -> List[float]:
    """
    If pick_model is available (imscore PickScorer), use it (often accepts list of file paths).
    If not available, fallback to deterministic CLIP-proxy based on dot with fixed vector.
    Caches per-subgroup to speed repeated runs.
    """
    # simple cache by joined names
    key = None
    if cache_key:
        key = CACHE_DIR / f"pick_{cache_key}.pkl"
        if key.exists():
            try:
                d = pickle.load(open(key,"rb"))
                # return in same order
                return [d.get(p.name, 0.0) for p in paths]
            except Exception:
                pass

    scores = []
    if pick_model is not None:
        # Many PickScorer APIs allow scoring lists of filepaths — try that, else fallback to per-image inference.
        try:
            # If the pick_model has .score(paths) method:
            if hasattr(pick_model, "score"):
                scores = pick_model.score([str(p) for p in paths])
                scores = [float(s) for s in scores]
            else:
                # per-image inference: assume pick_model accepts CLIP embeddings
                scores = []
                for p in paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        inputs = clip_proc(images=img, return_tensors="pt").to(device)
                        with torch.no_grad():
                            emb = clip_model.get_image_features(**inputs)
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                            out = pick_model(emb).cpu().numpy().item()
                        scores.append(float(out))
                    except Exception:
                        scores.append(0.0)
        except Exception:
            # fallback to proxy
            scores = []
            rng = np.random.RandomState(12345)
            vec = rng.normal(size=(clip_model.config.projection_dim,))
            vec = vec / np.linalg.norm(vec)
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    inputs = clip_proc(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        emb = clip_model.get_image_features(**inputs)
                        emb_np = emb.cpu().numpy().flatten()
                    scores.append(float(np.dot(emb_np, vec)))
                except Exception:
                    scores.append(0.0)
    else:
        # proxy - deterministic vector projection
        rng = np.random.RandomState(12345)
        vec = rng.normal(size=(clip_model.config.projection_dim,))
        vec = vec / np.linalg.norm(vec)
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                inputs = clip_proc(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = clip_model.get_image_features(**inputs)
                    emb_np = emb.cpu().numpy().flatten()
                scores.append(float(np.dot(emb_np, vec)))
            except Exception:
                scores.append(0.0)

    # cache
    if key:
        try:
            d = {}
            if key.exists():
                d = pickle.load(open(key,"rb"))
            for p,s in zip(paths, scores):
                d[p.name] = float(s)
            pickle.dump(d, open(key,"wb"))
        except Exception:
            pass

    return scores

# -------------------------
# Final scoring per image
# -------------------------
def final_score_for_image(path: Path, emb: np.ndarray,
                          pick_model, clip_model, clip_proc, device,
                          musiq_model, mtcnn, resnet, fer_detector,
                          cache_prefix: Optional[str]=None):
    # pickscore (proxy if None)
    pick_val = compute_pickscore_for_paths(pick_model, clip_model, clip_proc, device, [path], cache_key=(cache_prefix + "_pick" if cache_prefix else None))[0]
    # musiq
    musiq_val = musiq_score(musiq_model, path, cache_key=(cache_prefix + "_musiq" if cache_prefix else None))
    # faces
    faces = detect_faces_and_signals(path, mtcnn, resnet, fer_detector, cache_key=(cache_prefix + "_faces" if cache_prefix else None))
    face_val = face_quality_score(faces)
    # whole-image sharpness
    if _have_cv2:
        try:
            im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            whole_sharp = float(cv2.Laplacian(im, cv2.CV_64F).var()) if im is not None else 0.0
        except Exception:
            whole_sharp = 0.0
    else:
        whole_sharp = 0.0
    sharp_n = normalize_sharpness(whole_sharp)
    # normalize musiq to 0..1
    musiq_norm = max(0.0, min(1.0, float(musiq_val) / 100.0))
    # combine — note pick_val may need normalization across subgroup; we normalize later per subgroup
    raw = (WEIGHT_PICK * float(pick_val) + WEIGHT_FACE * float(face_val) + WEIGHT_MUSIQ * float(musiq_norm) + WEIGHT_SHARP * float(sharp_n))
    meta = {"pick": pick_val, "face": face_val, "musiq": musiq_val, "sharp": sharp_n}
    return float(raw), meta

# -------------------------
# Grouping functions (time -> CLIP split)
# -------------------------
def group_by_time(paths: List[Path], time_window_seconds: int = TIME_WINDOW_SECONDS) -> List[List[Path]]:
    items = []
    for p in paths:
        dt = read_exif_datetime(p)
        if dt is None:
            dt = file_mtime_datetime(p)
        items.append((p, dt))
    items.sort(key=lambda x: x[1])
    if not items:
        return []
    groups = []
    cur = [items[0]]
    for i in range(1, len(items)):
        prev = items[i-1][1]
        curdt = items[i][1]
        if (curdt - prev).total_seconds() <= time_window_seconds:
            cur.append(items[i])
        else:
            groups.append([x[0] for x in cur])
            cur = [items[i]]
    if cur:
        groups.append([x[0] for x in cur])
    return groups

def split_time_group_by_clip(time_group_paths: List[Path], clip_model, clip_proc, device, threshold=SIMILARITY_THRESHOLD):
    if len(time_group_paths) <= 5:
        return [time_group_paths]
    embs = compute_clip_embeddings(time_group_paths, clip_model, clip_proc, device, cache_key=None)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0]=1.0
    embs = embs / norms
    inds, sims = build_and_query_index(embs, top_k=K_NEIGHBORS)
    clusters_idx = cluster_by_similarity(inds, sims, threshold=threshold)
    clusters = [[time_group_paths[i] for i in c] for c in clusters_idx]
    return clusters

# -------------------------
# Main
# -------------------------
def main():
    global CACHE_DIR
    parser = argparse.ArgumentParser(description="Full-power best-photo selector (PickScore + MUSIQ + face signals).")
    parser.add_argument("--path", type=str, help="Folder with images (flat). If omitted uses default.")
    parser.add_argument("--time-window", type=int, default=TIME_WINDOW_SECONDS)
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--dry", action="store_true", help="Dry run (no output file)")
    args = parser.parse_args()

    folder = Path(args.path) if args.path else Path(DEFAULT_INPUT)
    if not folder.exists():
        print("Input folder not found:", folder)
        return

    # update global cache dir if user provided
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load models
    clip_model, clip_proc = load_clip(device)
    musiq_model = load_musiq()
    mtcnn, resnet = load_face_models(device)
    fer_detector = FER(mtcnn=False)  # emotion detector
    pick_model = load_pickscore(device)

    # list images (flat)
    image_paths = list_images_flat(folder)
    if not image_paths:
        print("No images found in", folder)
        return
    print(f"Found {len(image_paths)} images")

    # group by time
    time_groups = group_by_time(image_paths, time_window_seconds=args.time_window)
    print(f"Grouped into {len(time_groups)} time groups (window={args.time_window}s)")

    kept = []
    for gi, tg in enumerate(time_groups):
        subgroups = split_time_group_by_clip(tg, clip_model, clip_proc, device)
        for sg in subgroups:
            # precompute clip embeddings for subgroup and pickscore if needed
            emb = compute_clip_embeddings(sg, clip_model, clip_proc, device, cache_key=None)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            emb = emb / norms

            # compute raw scores per-image
            raw_scores = []
            metas = []
            # Optionally compute pickscore vector for the subgroup in batch (better normalization)
            pickvals = compute_pickscore_for_paths(pick_model, clip_model, clip_proc, device, sg, cache_key=f"group_{gi}_{len(sg)}")
            for i, p in enumerate(sg):
                raw, meta = final_score_for_image(p, emb[i], pick_model, clip_model, clip_proc, device,
                                                  musiq_model, mtcnn, resnet, fer_detector,
                                                  cache_prefix=f"group_{gi}_{len(sg)}")
                raw_scores.append(raw)
                metas.append(meta)
            arr = np.array(raw_scores, dtype=float)
            if np.all(np.isfinite(arr)) and (arr.max() - arr.min() > 1e-8):
                arr_n = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                arr_n = np.zeros_like(arr)
            best_idx = int(np.argmax(arr_n))
            best_path = sg[best_idx]
            kept.append(best_path)
            print(f"[Moment {gi}] subgroup size {len(sg)} -> keep: {best_path.name} score={arr_n[best_idx]:.3f} meta={metas[best_idx]}")

    # write kept list
    if not args.dry:
        out = folder / "kept_best.txt"
        with open(out, "w", encoding="utf-8") as f:
            for p in kept:
                f.write(str(p) + "\n")
        print("Wrote kept list to", out)
    else:
        print("Dry run; no file written.")

    print("Done.")

if __name__ == "__main__":
    main()
