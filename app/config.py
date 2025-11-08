# app/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Model local folders (your repo layout)
SWINIR_ROOT = os.path.join(BASE_DIR, "..", "models", "swinir")
RESTORMER_ROOT = os.path.join(BASE_DIR, "models", "restormer")
GFPGAN_ROOT = os.path.join(BASE_DIR, "models", "gfpgan")

# explicit model weight paths
SWINIR_WEIGHT = os.path.join(SWINIR_ROOT, "weights", "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
GFPGAN_WEIGHT = os.path.join(BASE_DIR, "models", "gfpgan", "weights", "GFPGANv1.3.pth")
RESTORMER_WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "restormer", "weights")

# Analyzer thresholds
BLUR_LAPLACIAN_THRESH = 100.0
NOISE_STD_THRESH = 12.0
LOW_RES_PIXELS = 800 * 600

# Upscale preference
DEFAULT_UPSCALE = 2  # final upscaling factor
