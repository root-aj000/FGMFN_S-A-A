# server/app.py
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from PIL import Image
import uuid
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from server.predict import predict, IMAGE_UPLOAD_DIR
from utils.path import LOG_DIR

# ------------------ CONFIG ------------------
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_DIR, exist_ok=True)

# ------------------ LOGGING ------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "server.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Multi-Modal Sentiment Classifier API")

# === CORS ===
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Pydantic Models ------------------
class Prediction(BaseModel):
    filename: str
    predicted_label_text: str
    predicted_label_num: int
    confidence_score: float
    ocr_text: str

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

# ------------------ UTILITY ------------------
def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def save_upload_file(upload_file: UploadFile, dest_folder: str) -> str:
    ext = os.path.splitext(upload_file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(dest_folder, unique_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path

# ------------------ PREDICT ENDPOINT ------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    images = []
    filenames = []
    uploaded_paths = []

    try:
        # Save uploaded files temporarily
        for file in files:
            file_path = save_upload_file(file, IMAGE_UPLOAD_DIR)
            uploaded_paths.append(file_path)
            img = Image.open(file_path).convert("RGB")
            images.append(img)
            filenames.append(file.filename)

        # Run predictions (OCR + sentiment)
        results = predict(images)

        # Attach filenames to results
        for i, res in enumerate(results):
            res["filename"] = filenames[i]

        logging.info(f"Predictions made for {len(images)} images.")
        return {"predictions": results}

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        # Delete only files uploaded in this request
        for file_path in uploaded_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

# ------------------ HEALTH CHECK ------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}

# # ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.app:app",
        port=8000,
        reload=True
    )