# server/app.py
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image

from predict import predict, IMAGE_UPLOAD_DIR
from utils.path import SAVED_MODEL_PATH, LOG_DIR

# ------------------ CONFIG ------------------
MODEL_PATH = SAVED_MODEL_PATH
UPLOAD_FOLDER = IMAGE_UPLOAD_DIR
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
LOG_DIR = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ LOGGING ------------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "server.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Multi-Modal Sentiment Classifier API")

def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

# ------------------ PREDICT ENDPOINT ------------------
@app.post("/predict")
async def predict_endpoint(files: List[UploadFile] = File(...), texts: List[str] = None):
    if texts is None:
        texts = ["" for _ in files]

    if len(texts) != len(files):
        raise HTTPException(status_code=400, detail="Number of texts must match number of images.")

    images = []
    try:
        # Save uploaded files temporarily
        for file in files:
            if not allowed_file(file.filename):
                raise HTTPException(status_code=400, detail=f"File type not allowed: {file.filename}")
            temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            img = Image.open(temp_path).convert("RGB")
            images.append(img)

        # Run predictions
        results = predict(images, texts)
        logging.info(f"Predictions made for {len(images)} images.")
        return JSONResponse(content={"predictions": results})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    finally:
        # Clean up uploaded images
        for img_file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, img_file))

# ------------------ HEALTH CHECK ------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}
