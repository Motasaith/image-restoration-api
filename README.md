# Image Restoration API

A FastAPI-based RESTful API service for intelligent image restoration. It automatically analyzes uploaded images for issues like blur, noise, low resolution, faces, and text, then applies appropriate AI models to enhance them. The API supports multiple image formats and provides detailed metadata about the restoration process.

## Features

- **Intelligent Analysis**: Automatically detects image quality issues (blur, noise, resolution, faces, text) using computer vision and OCR.
- **Conditional Model Application**: Applies AI models only when needed based on analysis results.
- **RESTful API**: Simple endpoints for image upload and restoration.
- **Multiple Formats**: Supports common image formats (JPEG, PNG, etc.).
- **Metadata Logging**: Returns headers with analysis decisions and applied models.
- **Preview Mode**: Optional endpoint to analyze without processing.
- **Output Saving**: Saves restored images to disk for reference.

## Installation

### Prerequisites
- Python 3.8+
- System dependencies:
  - Tesseract OCR (for text detection): Install via package manager (e.g., `apt install tesseract-ocr` on Ubuntu, `brew install tesseract` on macOS, or download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) on Windows).
  - Optional: CUDA-compatible GPU for faster inference with PyTorch models.

### Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-restoration-api
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Setup**:
   - Models are pre-included in the `models/` directory (Real-ESRGAN, GFPGAN, Restormer repositories are cloned).
   - Weights are downloaded automatically on first run or can be pre-downloaded from their respective repos.
   - For Real-ESRGAN: Weights are fetched from GitHub releases.
   - For GFPGAN/Restormer: Ensure `models/gfpgan/weights/` and `models/restormer/weights/` contain the required PTH files.

4. **Run the server**:
   ```bash
   python app/main.py
   ```
   The API will be available at `http://localhost:8000`.

## Usage

Upload an image via POST request to `/restore`. The API analyzes the image and applies restoration models as needed.

### Example with curl:
```bash
curl -X POST "http://localhost:8000/restore" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg" \
     --output restored_image.png
```

### Response
- **Body**: The restored image (PNG format).
- **Headers**:
  - `X-Restoration-Decisions`: JSON string with analysis results (e.g., blur score, noise level, face count).
  - `X-Applied-Models`: Comma-separated list of models used (e.g., "restormer:motion_deblurring,RealESRGAN_x4,GFPGAN").
  - `X-Saved-File`: Path to the saved output file on the server.

### Python Example:
```python
import requests

url = "http://localhost:8000/restore"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)

with open("restored.png", "wb") as f:
    f.write(response.content)

print("Decisions:", response.headers.get("X-Restoration-Decisions"))
print("Models Used:", response.headers.get("X-Applied-Models"))
```

## API Endpoints

- `POST /restore`: Upload and restore an image. Returns the processed image with metadata headers.
- `GET /health`: Health check endpoint. Returns `{"status": "ok"}`.
- `POST /restore/info`: (Not implemented in current code, but mentioned in original README) – Analyze image and return JSON decisions without processing.

## Models Used

The API applies models conditionally based on image analysis:

- **Real-ESRGAN**: Super-resolution for upscaling low-resolution images (4x scale).
- **GFPGAN**: Face restoration for images with detected faces and low blur scores.
- **Restormer**: Denoising and deblurring (motion blur, defocus, real-world noise).
- **Text Sharpening**: Custom sharpening for text clarity (black text on colored backgrounds).
- **Edge Boost**: Light sharpening for text images.

Models are loaded on-demand and cached for performance.

## System Requirements

- **CPU/GPU**: PyTorch models run on CPU by default; use CUDA GPU for faster processing.
- **Memory**: At least 4GB RAM recommended; higher for large images or GPU inference.
- **Disk Space**: ~2GB for models and dependencies.

## Testing

- Use sample images from `tests/sample_images/` for various scenarios (blurry, noisy, low-res, with faces/text).
- Run `python -m pytest` if tests are added in the future.
- Check logs for applied models and saved output paths.

## Development Notes

- The pipeline was originally placeholders; now integrates real AI models via Python wrappers.
- TODO.md tracks planned refactor to replace Real-ESRGAN with SwinIR for better performance.
