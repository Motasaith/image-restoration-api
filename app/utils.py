# app/utils.py
from io import BytesIO
from PIL import Image
import numpy as np

def read_imagefile_bytes(bytestr: bytes) -> Image.Image:
    return Image.open(BytesIO(bytestr)).convert("RGB")

def pil_to_cv2(img_pil):
    arr = np.array(img_pil)
    # convert RGB to BGR for cv2 if needed - but many libs accept RGB, we keep RGB
    return arr

def cv2_to_pil(img_arr):
    return Image.fromarray(img_arr.astype("uint8"))

def save_pil_to_bytes(pil_img: Image.Image, fmt="PNG"):
    buf = BytesIO()
    pil_img.save(buf, fmt)
    buf.seek(0)
    return buf
