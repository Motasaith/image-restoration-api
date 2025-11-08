# app/model_wrappers.py
import os
import sys
import threading
import numpy as np
from PIL import Image
import cv2
import tempfile
import subprocess
import urllib.request
import torch
def download_weight(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading SwinIR weight to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Done!")

from .config import SWINIR_ROOT, GFPGAN_ROOT, RESTORMER_ROOT, SWINIR_WEIGHT, GFPGAN_WEIGHT, RESTORMER_WEIGHTS_DIR

_lock = threading.Lock()
_models = {"gfpgan": None, "restormer": None}

def _add_sys_path(path):
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# ────────────────────────────────
# REAL-ESRGAN REPLACES SWINIR ONLY
# ────────────────────────────────
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

print("Loading Real-ESRGAN (first run = 10 sec)...")
_realesrgan = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("Real-ESRGAN LOADED!")

def apply_realesrgan(img_np):
    output, _ = _realesrgan.enhance(img_np)
    return output

# THIS LINE REPLACES SWINIR
apply_swinir = apply_realesrgan

# --- GFPGAN wrapper ---
def load_gfpgan():
    with _lock:
        if _models["gfpgan"] is not None:
            return _models["gfpgan"]
        try:
            _add_sys_path(GFPGAN_ROOT)
            from gfpgan import GFPGANer
            gp = GFPGANer(
                model_path=GFPGAN_WEIGHT if os.path.exists(GFPGAN_WEIGHT) else None,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            _models["gfpgan"] = gp
            return gp
        except Exception as e:
            print("GFPGAN import/load failed:", e)
            _models["gfpgan"] = None
            return None

def apply_gfpgan(np_img):
    gp = _models.get("gfpgan") or load_gfpgan()
    if gp is None:
        return np_img
    try:
        cropped_faces, restored_faces, restored_img = gp.enhance(
            np_img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return restored_img
    except Exception as e:
        print("GFPGAN inference error:", e)
        return np_img


# --- Restormer wrapper ---
def load_restormer(model_name=None):
    with _lock:
        if _models.get("restormer") is not None:
            return _models["restormer"]
        try:
            _add_sys_path(RESTORMER_ROOT)
            import demo as restormer_demo
            _models["restormer"] = restormer_demo
            return restormer_demo
        except Exception as e:
            print("Restormer import failed:", e)
            _models["restormer"] = None
            return None

def apply_restormer(np_img, model_choice="real_denoising"):
    rest = _models.get("restormer") or load_restormer()
    if rest is None:
        den = cv2.fastNlMeansDenoisingColored(np_img, None, 10,10,7,21)
        return den
    try:
        if hasattr(rest, "restore_image"):
            return rest.restore_image(np_img, model_choice)
        if hasattr(rest, "test"):
            return rest.test(np_img, model_choice)
    except Exception as e:
        print("Restormer inference failed:", e)
    den = cv2.fastNlMeansDenoisingColored(np_img, None, 10,10,7,21)
    return den
