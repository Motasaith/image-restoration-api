# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from .utils import read_imagefile_bytes, save_pil_to_bytes
from .restore_pipeline import decide_and_run
from pathlib import Path
import uuid
import uvicorn

# === CREATE OUTPUT FOLDER ===
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Image Restoration API",
    description="Upload blurry/text/low-res images → get crystal clear version",
    version="1.0"
)

@app.post("/restore")
async def restore(file: UploadFile = File(...)):
    content = await file.read()
    try:
        pil_img = read_imagefile_bytes(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    out_pil, decisions, applied = decide_and_run(pil_img)

    # === SAVE TO DISK ===
    filename = f"{uuid.uuid4().hex[:12]}_{file.filename or 'image.png'}"
    save_path = OUTPUT_DIR / filename
    out_pil.save(save_path, "PNG")
    print(f"SAVED → {save_path}")

    # === STREAM BACK ===
    buf = save_pil_to_bytes(out_pil, fmt="PNG")
    headers = {
        "X-Restoration-Decisions": str(decisions),
        "X-Applied-Models": ",".join(applied),
        "X-Saved-File": str(save_path),
    }
    return StreamingResponse(buf, media_type="image/png", headers=headers)

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)