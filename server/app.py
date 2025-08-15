# server/app.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from typing import List
import shutil
import uuid

from server.predict import predict, IMAGE_UPLOAD_DIR

# ---------------- FastAPI App ----------------
app = FastAPI(title="Multi-modal Sentiment API")

# Enable CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utility ----------------
def save_upload_file(upload_file: UploadFile, dest_folder: str) -> str:
    """Save uploaded file to disk and return the saved file path."""
    os.makedirs(dest_folder, exist_ok=True)
    ext = os.path.splitext(upload_file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(dest_folder, unique_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

# ---------------- Routes ----------------
@app.post("/predict")
async def predict_sentiment(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    images = []
    filenames = []

    # Save uploaded images temporarily
    for f in files:
        try:
            file_path = save_upload_file(f, IMAGE_UPLOAD_DIR)
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            filenames.append(os.path.basename(file_path))
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # Run prediction (OCR + sentiment)
    results = predict(images)  # predict() should return OCR text + labels

    # Attach filenames
    for i, res in enumerate(results):
        res["filename"] = filenames[i]

    return results

# ---------------- Health Check ----------------
@app.get("/")
async def root():
    return {"message": "Multi-modal Sentiment API is running"}