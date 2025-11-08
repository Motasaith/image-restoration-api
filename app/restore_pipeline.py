# app/restore_pipeline.py
from .analyzer import analyze
from .model_wrappers import apply_restormer, apply_swinir, apply_gfpgan
from .utils import pil_to_cv2, cv2_to_pil
from PIL import Image
import cv2
import numpy as np

def decide_and_run(pil_img: Image.Image):
    decisions = analyze(pil_img)
    applied = []
    img_np = pil_to_cv2(pil_img)

    # 1. Restormer — fixes blur & noise
    if (decisions["need_deblur"] or decisions["need_denoise"]) and decisions["has_text"]:
        choice = "motion_deblurring" if decisions["blur_score"] < 30 else "single_image_defocus_deblurring"
        if decisions["need_denoise"]: choice = "real_denoising"
        img_np = apply_restormer(img_np, choice)
        applied.append(f"restormer:{choice}")

    # 2. Real-ESRGAN — 4× upscale (via apply_swinir)
    if decisions["need_upscale"] or decisions["has_text"]:
        img_np = apply_swinir(img_np)
        applied.append("RealESRGAN_x4")

        # 3. GFPGAN — ONLY for old/damaged portraits
    if decisions["faces"] > 0 and decisions["blur_score"] < 50:
        img_np = apply_gfpgan(img_np)
        applied.append("GFPGAN")
    else:
        applied.append("GFPGAN_SKIPPED")

    # 4. SMART BLACK TEXT — keeps yellow/pink/blue paper!
    if decisions["has_text"]:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        text_mask = thresh < 127
        img_np[text_mask] = [0, 0, 0]  # BLACK TEXT ONLY
        applied.append("SMART_BLACK")

    # 5. Tiny edge boost
    if decisions["has_text"]:
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32)
        sharp = cv2.filter2D(img_np, -1, kernel)
        img_np = cv2.addWeighted(img_np, 0.9, sharp, 0.1, 0)
        applied.append("edge_boost")

    return cv2_to_pil(img_np), decisions, applied