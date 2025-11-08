# app/analyzer.py
import cv2
import numpy as np
from pytesseract import image_to_data
from PIL import Image
from .config import BLUR_LAPLACIAN_THRESH, NOISE_STD_THRESH, LOW_RES_PIXELS

def variance_of_laplacian(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def estimate_noise_std(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(lap))

def has_text(pil_img: Image.Image, conf_thresh=30):
    try:
        data = image_to_data(pil_img, output_type='dict')
    except Exception:
        return False, 0.0
    confs = data.get("conf", [])
    total = 0
    good = 0
    for c in confs:
        try:
            v = float(c)
        except:
            v = -1
        if v >= 0:
            total += 1
            if v >= conf_thresh:
                good += 1
    score = (good / total) if total > 0 else 0.0
    return (score > 0.12), score

def detect_faces(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    return len(faces)

def analyze(pil_img: Image.Image):
    np_img = np.array(pil_img)
    h, w = np_img.shape[:2]
    lap = variance_of_laplacian(np_img)
    noise = estimate_noise_std(np_img)
    text_present, text_score = has_text(pil_img)
    face_count = detect_faces(np_img)

    decisions = {
        "width": w,
        "height": h,
        "pixels": int(w*h),
        "blur_score": float(lap),
        "noise_std": float(noise),
        "has_text": bool(text_present),
        "text_score": float(text_score),
        "faces": int(face_count),
        "need_deblur": lap < BLUR_LAPLACIAN_THRESH,
        "need_denoise": noise > NOISE_STD_THRESH,
        "need_upscale": (w*h) < LOW_RES_PIXELS
    }
    return decisions
