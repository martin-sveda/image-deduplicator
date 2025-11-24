#!/usr/bin/env python3
"""
best_photo_with_internvl2.py

Full-power best-photo selector (family shots) with InternVL2-2B as final chooser.

Features:
 - time-based grouping (EXIF or mtime)
 - CLIP ViT-L/14 embeddings -> clustering (FAISS or sklearn fallback)
 - PickScore via imscore (optional) or CLIP-proxy fallback
 - MUSIQ TF-Hub technical score
 - Face detection & signals (MTCNN, InceptionResnetV1, Haar + FER/DeepFace fallback)
 - InternVL2-2B (OpenGVLab/InternVL2-2B) used to pick best among prefiltered candidates
 - Caching of heavy computations
"""

import argparse
from pathlib import Path
import os
import sys
import math
import pickle
from datetime import datetime
from typing import List, Optional, Dict
import time
import re

import numpy as np
from PIL import Image, ExifTags, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from tqdm import tqdm

# transformers CLIP + InternVL2 pipeline
from transformers import CLIPModel, CLIPProcessor, pipeline

# facenet
from facenet_pytorch import MTCNN, InceptionResnetV1

# TF-Hub MUSIQ
import tensorflow as tf
import tensorflow_hub as hub

# emotion detector (try fer, fallback to deepface)
_has_fer = False
try:
    from fer import FER
    _has_fer = True
except Exception:
    try:
        from deepface import DeepFace
        _has_fer = False
    except Exception:
        _has_fer = False

# optional libs
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

# optional imscore
PickScorer = None
try:
    try:
        from imscore.preference.model import PickScorer as _PickScorer
    except Exception:
        from imscore import PickScorer as _PickScorer
    PickScorer = _PickScorer
    _have_imscore = True
except Exception:
    PickScorer = None
    _have_imscore = False

# -------------------------
# Configurable params
# -------------------------
DEFAULT_INPUT = r"c:\Users\Z004JR9Y\OneDrive - Siemens AG\Dokumenty\msveda\github\tst-foto"
TIME_WINDOW_SECONDS = 120
CLIP_BATCH = 16
K_NEIGHBORS = 6
SIMILARITY_THRESHOLD = 0.92
MIN_CLUSTER_SIZE = 1

WEIGHT_PICK = 0.45
WEIGHT_FACE = 0.30
WEIGHT_MUSIQ = 0.15
WEIGHT_SHARP = 0.10

MUSIQ_HUB = "https://tfhub.dev/google/musiq/paq2piq/1"

CACHE_DIR = Path(".bestphoto_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Haar cascades if cv2 available
if _have_cv2:
    HAAR_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
    HAAR_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"
else:
    HAAR_EYE = HAAR_SMILE = None

# InternVL2 model id (OpenGVLab)
INTERNVL2_MODEL_ID = "OpenGVLab/InternVL2-2B"

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
    mtcnn = MTCNN(keep_all=True, device=device if device.type != 'cpu' else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn, resnet

def load_pickscore(device: torch.device):
    if not _have_imscore:
        print("imscore not installed; PickScore unavailable, using CLIP-proxy fallback.")
        return None
    try:
        try:
            model = PickScorer.from_pretrained("RE-N-Y/pickscore")
        except Exception:
            model = PickScorer()
        model.to(device)
        model.eval()
        print("Loaded PickScorer.")
        return model
    except Exception as e:
        print("PickScorer load failed:", e)
        return None

# -------------------------
# InternVL2 loader + selector
# -------------------------
def load_internvl2(model_id: str = INTERNVL2_MODEL_ID, device: Optional[torch.device] = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe_device = 0 if device.type == "cuda" else -1
    print(f"Loading InternVL2 pipeline '{model_id}' on device {device} ...")

    try:
        intern_pipe = pipeline(
            task="image-to-text",
            model=model_id,
            device=pipe_device,
            trust_remote_code=True
        )
    except Exception as e:
        print("Failed to load pipeline with 'image-to-text':", e)
        try:
            intern_pipe = pipeline(
                task="vision-text-generation",
                model=model_id,
                device=pipe_device,
                trust_remote_code=True
            )
        except Exception as e2:
            print("Failed to load InternVL2 pipeline:", e2)
            return None

    print("InternVL2 pipeline ready.")
    return intern_pipe



def internvl2_select_best(pipe, candidate_paths: List[Path], max_candidates: int = 8, verbose: bool = False) -> int:
    """
    Use InternVL2 pipeline to pick best image among candidate_paths.
    Returns index relative to candidate_paths.
    """
    if pipe is None:
        raise RuntimeError("InternVL2 pipeline is None")

    if len(candidate_paths) == 0:
        raise ValueError("No candidates")

    # limit candidates
    if len(candidate_paths) > max_candidates:
        step = max(1, len(candidate_paths) // max_candidates)
        candidate_paths = candidate_paths[::step][:max_candidates]
        if verbose:
            print(f"Trimmed to {len(candidate_paths)} candidates for InternVL2")

    # load PIL images
    pil_images = []
    for p in candidate_paths:
        try:
            pil_images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            if verbose:
                print("Failed to open", p, e)

    # build concise prompt expecting an integer index
    prompt = (
        "You will be shown several similar photos. "
        "Choose the single best photo that you would keep as the family 'keeper'. "
        "Criteria: natural/genuine smiles, eyes open, everyone visible and in-focus, pleasant lighting, balanced composition. "
        "Return exactly the index (0-based) of the best photo and nothing else."
    )

    # call pipeline with images and prompt
    try:
        # many vision-text pipelines accept a list of images + prompt via "image" or "images" param
        out = pipe(prompt, images=pil_images, max_new_tokens=32)  # signature may vary
    except TypeError:
        try:
            out = pipe(images=pil_images, text=prompt, max_new_tokens=32)
        except Exception as e:
            if verbose:
                print("InternVL2 multi-image call failed:", e)
            # fallback: rate each image individually
            ratings = []
            rate_prompt = (
                "Rate this photo from 0 to 10 for how good it is as a candidate for family album. "
                "Consider natural smile, eyes open, clarity and composition. Return only a single number."
            )
            for img in pil_images:
                try:
                    r = pipe(rate_prompt, images=img, max_new_tokens=16)
                    if isinstance(r, list) and isinstance(r[0], dict):
                        txt = r[0].get('generated_text', '') or r[0].get('text', '')
                    elif isinstance(r, dict):
                        txt = r.get('generated_text', '') or r.get('text', '')
                    else:
                        txt = str(r)
                    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", txt)
                    val = float(m.group(1)) if m else 0.0
                except Exception:
                    val = 0.0
                ratings.append(val)
            best_idx = int(np.argmax(ratings)) if ratings else 0
            return best_idx
    except Exception as e:
        if verbose:
            print("InternVL2 call exception:", e)
        return 0

    # parse text
    txt = ""
    if isinstance(out, list):
        # often pipeline returns list of dicts
        first = out[0]
        if isinstance(first, dict):
            txt = first.get('generated_text', '') or first.get('text', '') or str(first)
        else:
            txt = str(first)
    elif isinstance(out, dict):
        txt = out.get('generated_text', '') or out.get('text', '')
    else:
        txt = str(out)

    if verbose:
        print("InternVL2 output:", txt)

    m = re.search(r"\b([0-9]+)\b", txt)
    if not m:
        m = re.search(r"(index|photo|image)\s*[:#]?\s*([0-9]+)", txt, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(2))
        else:
            if verbose:
                print("Could not parse index from InternVL2 output; defaulting to 0")
            idx = 0
    else:
        idx = int(m.group(1))

    idx = max(0, min(idx, len(candidate_paths) - 1))
    return idx

# -------------------------
# Embedding / neighbor / cluster helpers
# -------------------------
def compute_clip_embeddings(paths: List[Path], clip_model, clip_proc, device, cache_key: Optional[str] = None) -> np.ndarray:
    if cache_key:
        cache_file = CACHE_DIR / f"emb_{cache_key}.pkl"
        if cache_file.exists():
            try:
                arr = pickle.load(open(cache_file, "rb"))
                if arr.shape[0] == len(paths):
                    return arr
            except Exception:
                pass

    clip_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), CLIP_BATCH), desc="CLIP embedded batches"):
            batch = paths[i:i+CLIP_BATCH]
            imgs = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(None)
            valid_idx = [j for j, im in enumerate(imgs) if im is not None]
            if not valid_idx:
                d = clip_model.config.projection_dim
                embeddings.extend([np.zeros(d, dtype=np.float32)] * len(batch))
                continue
            batch_imgs = [imgs[j] for j in valid_idx]
            inputs = clip_proc(images=batch_imgs, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype("float32")
            d = feats.shape[1]
            outb = [None] * len(batch)
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
            pickle.dump(emb_arr, open(CACHE_DIR / f"emb_{cache_key}.pkl", "wb"))
        except Exception:
            pass
    return emb_arr

def build_and_query_index(embeddings: np.ndarray, top_k: int = K_NEIGHBORS):
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
# MUSIQ + faces + pickscore
# -------------------------
def load_musiq():
    print("Loading MUSIQ TF-Hub model...")
    return hub.load(MUSIQ_HUB)

def musiq_score(musiq_model, path: Path, cache_key: Optional[str] = None) -> float:
    if cache_key:
        cf = CACHE_DIR / f"musiq_{cache_key}.pkl"
        if cf.exists():
            try:
                d = pickle.load(open(cf, "rb"))
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
            d = {}
            if cf.exists():
                d = pickle.load(open(cf, "rb"))
            d[path.name] = val
            pickle.dump(d, open(cf, "wb"))
        except Exception:
            pass
    return val

def detect_faces_and_signals(path: Path, mtcnn: MTCNN, resnet: InceptionResnetV1, fer_detector, cache_key: Optional[str] = None):
    if cache_key:
        cf = CACHE_DIR / f"faces_{cache_key}.pkl"
        if cf.exists():
            try:
                d = pickle.load(open(cf, "rb"))
                if path.name in d:
                    return d[path.name]
            except Exception:
                pass

    out = []
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return out

    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        if cache_key:
            try:
                d = {}
                if cf.exists(): d = pickle.load(open(cf, "rb"))
                d[path.name] = out
                pickle.dump(d, open(cf, "wb"))
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

    # embeddings
    import torchvision.transforms as T
    trans = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    batch = torch.stack([trans(f[1]).to(resnet.device) for f in faces])
    with torch.no_grad():
        embs = resnet(batch).cpu().numpy()

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

        # emotion detection
        happy = 0.0
        try:
            if _has_fer:
                em = fer_detector.detect_emotions(face_np)
                if em and len(em)>0 and isinstance(em[0], dict):
                    emotions = em[0]["emotions"]
                    happy = float(emotions.get("happy", 0.0))
            else:
                # deepface fallback
                res = DeepFace.analyze(face_np, actions=['emotion'], enforce_detection=False)
                if isinstance(res, list) and len(res)>0:
                    emotions = res[0].get("emotion", {})
                    happy = float(emotions.get("happy", 0.0))/100.0
                elif isinstance(res, dict):
                    emotions = res.get("emotion", {})
                    happy = float(emotions.get("happy", 0.0))/100.0
        except Exception:
            happy = 0.0

        emb = embs[i] if i < len(embs) else np.zeros(512)
        out.append({
            "box": (x1,y1,x2,y2),
            "embedding": emb,
            "sharpness": sharpness,
            "eyes": eyes_count,
            "smile_haar": smile_haar,
            "happy": happy
        })

    if cache_key:
        try:
            d = {}
            if cf.exists(): d = pickle.load(open(cf, "rb"))
            d[path.name] = out
            pickle.dump(d, open(cf, "wb"))
        except Exception:
            pass

    return out

def normalize_sharpness(s: float) -> float:
    s = max(0.0, float(s))
    return min(1.0, math.log10(s + 1.0) / 3.0)

def face_quality_score(faces: List[Dict]) -> float:
    if not faces:
        return 0.0
    per = []
    for f in faces:
        eyes_score = min(2, f.get("eyes", 0)) / 2.0
        smile_score = max(f.get("smile_haar", 0.0), f.get("happy", 0.0))
        sharp_n = normalize_sharpness(f.get("sharpness", 0.0))
        s = 0.4 * eyes_score + 0.45 * sharp_n + 0.15 * smile_score
        per.append(s)
    avg = float(np.mean(per))
    count = len(per)
    count_factor = min(1.0, 0.6 + 0.1 * count)
    return avg * count_factor

def compute_pickscore_for_paths(pick_model, clip_model, clip_proc, device, paths: List[Path], cache_key: Optional[str] = None) -> List[float]:
    key = None
    if cache_key:
        keyf = CACHE_DIR / f"pick_{cache_key}.pkl"
        if keyf.exists():
            try:
                d = pickle.load(open(keyf, "rb"))
                return [d.get(p.name, 0.0) for p in paths]
            except Exception:
                pass

    scores = []
    if pick_model is not None:
        try:
            if hasattr(pick_model, "score"):
                scores = pick_model.score([str(p) for p in paths])
                scores = [float(s) for s in scores]
            else:
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
            # fallback proxy
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

    if cache_key:
        try:
            d = {}
            keyf = CACHE_DIR / f"pick_{cache_key}.pkl"
            if keyf.exists(): d = pickle.load(open(keyf, "rb"))
            for p, s in zip(paths, scores):
                d[p.name] = float(s)
            pickle.dump(d, open(keyf, "wb"))
        except Exception:
            pass

    return scores

def final_score_for_image(path: Path, emb: np.ndarray,
                          pick_model, clip_model, clip_proc, device,
                          musiq_model, mtcnn, resnet, fer_detector,
                          cache_prefix: Optional[str] = None):
    pick_val = compute_pickscore_for_paths(pick_model, clip_model, clip_proc, device, [path], cache_key=(cache_prefix + "_pick" if cache_prefix else None))[0]
    musiq_val = musiq_score(musiq_model, path, cache_key=(cache_prefix + "_musiq" if cache_prefix else None))
    faces = detect_faces_and_signals(path, mtcnn, resnet, fer_detector, cache_key=(cache_prefix + "_faces" if cache_prefix else None))
    face_val = face_quality_score(faces)
    if _have_cv2:
        try:
            im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            whole_sharp = float(cv2.Laplacian(im, cv2.CV_64F).var()) if im is not None else 0.0
        except Exception:
            whole_sharp = 0.0
    else:
        whole_sharp = 0.0
    sharp_n = normalize_sharpness(whole_sharp)
    musiq_norm = max(0.0, min(1.0, float(musiq_val) / 100.0))
    raw = (WEIGHT_PICK * float(pick_val) + WEIGHT_FACE * float(face_val) + WEIGHT_MUSIQ * float(musiq_norm) + WEIGHT_SHARP * float(sharp_n))
    meta = {"pick": pick_val, "face": face_val, "musiq": musiq_val, "sharp": sharp_n}
    return float(raw), meta

# -------------------------
# Grouping & split by CLIP
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
    norms[norms == 0] = 1.0
    embs = embs / norms
    inds, sims = build_and_query_index(embs, top_k=K_NEIGHBORS)
    clusters_idx = cluster_by_similarity(inds, sims, threshold=threshold)
    clusters = [[time_group_paths[i] for i in c] for c in clusters_idx]
    return clusters

# -------------------------
# Prefilter candidates for InternVL2 (fast heuristics)
# -------------------------
def prefilter_candidates_by_sharpness_or_musiq(paths: List[Path], musiq_model, top_k: int = 8):
    scores = []
    for p in paths:
        # quick sharpness
        if _have_cv2:
            try:
                im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                sharp = float(cv2.Laplacian(im, cv2.CV_64F).var()) if im is not None else 0.0
            except Exception:
                sharp = 0.0
        else:
            sharp = 0.0
        # musiq (fast-ish)
        musiq_v = musiq_score(musiq_model, p, cache_key=None)
        # combine crude
        scores.append((p, 0.6 * sharp + 0.4 * musiq_v))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [x[0] for x in scores[:top_k]]
    return top

# -------------------------
# Main
# -------------------------
def main():
    global CACHE_DIR
    parser = argparse.ArgumentParser(description="Best photo selector with InternVL2 final chooser.")
    parser.add_argument("--path", type=str, help="Folder with images (flat).")
    parser.add_argument("--time-window", type=int, default=TIME_WINDOW_SECONDS)
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--intern-model", type=str, default=INTERNVL2_MODEL_ID, help="InternVL2 model id on HF")
    args = parser.parse_args()

    folder = Path(args.path) if args.path else Path(DEFAULT_INPUT)
    if not folder.exists():
        print("Input folder not found:", folder)
        return

    # update cache dir
    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load models
    clip_model, clip_proc = load_clip(device)
    musiq_model = load_musiq()
    mtcnn, resnet = load_face_models(device)
    fer_detector = None
    if _has_fer:
        try:
            fer_detector = FER(mtcnn=False)
        except Exception:
            fer_detector = None
    else:
        try:
            # DeepFace doesn't provide a small detector object; we'll call analyze per-face
            # ensure import succeeded earlier
            pass
        except Exception:
            pass
    pick_model = load_pickscore(device)

    # InternVL2 pipeline
    intern_pipe = load_internvl2(model_id=args.intern_model, device=device)

    # images
    image_paths = list_images_flat(folder)
    if not image_paths:
        print("No images found in", folder)
        return
    print(f"Found {len(image_paths)} images in {folder}")

    time_groups = group_by_time(image_paths, time_window_seconds=args.time_window)
    print(f"Grouped into {len(time_groups)} time groups (window={args.time_window}s)")

    kept = []
    for gi, tg in enumerate(time_groups):
        subgroups = split_time_group_by_clip(tg, clip_model, clip_proc, device)
        for sg in subgroups:
            # compute embeddings (for scoring fallback)
            emb = compute_clip_embeddings(sg, clip_model, clip_proc, device, cache_key=None)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms

            # prefilter candidates for InternVL2 (fast)
            prefiltered = prefilter_candidates_by_sharpness_or_musiq(sg, musiq_model, top_k=8)
            # ensure deterministic order mapping back if prefiltered is subset
            # If intern_pipe available, call it; else fallback to hybrid scoring
            if intern_pipe is not None:
                try:
                    best_rel = internvl2_select_best(intern_pipe, prefiltered, max_candidates=8, verbose=False)
                    best_path = prefiltered[best_rel]
                    kept.append(best_path)
                    print(f"[Moment {gi}] subgroup {len(sg)} -> keep (InternVL2): {best_path.name}")
                    continue
                except Exception as e:
                    print("InternVL2 selection failed, falling back to hybrid scoring:", e)

            # fallback: hybrid scoring (PickScore + face + musiq + sharpness)
            raw_scores = []
            metas = []
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
            print(f"[Moment {gi}] subgroup {len(sg)} -> keep (hybrid): {best_path.name} score={arr_n[best_idx]:.3f} meta={metas[best_idx]}")

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
