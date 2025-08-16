# server/predict.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json
from paddlex import create_pipeline
import traceback

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN
from preprocessing.text_preprocessing import tokenize_text
from utils.path import SAVED_MODEL_PATH, MODEL_CONFIG

# Optional: Tesseract OCR fallback (if needed)
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------ OCR MODEL ------------------
# Assuming ocr_model is defined globally somewhere in your project
ocr_model = create_pipeline(pipeline="ocr")

# ------------------ PATHS ------------------
MODEL_PATH = SAVED_MODEL_PATH
MODEL_CONFIG = MODEL_CONFIG
IMAGE_UPLOAD_DIR = "data/images/tmp_uploads/"

# ------------------ HYPERPARAMS ------------------
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LEN = 128

# ------------------ LABELS ------------------
LABELS = {0: "Neutral", 1: "Positive"}

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------ LOAD MODEL ------------------
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

model = FG_MFN(cfg).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------ OCR UTILITIES ------------------
def log_error(msg):
    print(f"[ERROR] {msg}")  # simple logger, replace with proper logging if needed

def extract_text(image_path_or_pil):
    try:
        # If PIL image, save temporarily
        if isinstance(image_path_or_pil, Image.Image):
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image_path_or_pil.save(tmp_file.name)
            image_path = tmp_file.name
        else:
            image_path = image_path_or_pil

        result = list(ocr_model.predict(image_path))

        rec_texts = []
        rec_scores = []

        if isinstance(result, list) and result and isinstance(result[0], dict):
            if "rec_texts" in result[0]:
                rec_texts = result[0]["rec_texts"]
                rec_scores = result[0].get("rec_scores", [])
            elif "data" in result[0]:
                for item in result[0]["data"]:
                    if isinstance(item, dict) and "text" in item:
                        rec_texts.append(item["text"])
                        rec_scores.append(item.get("score", None))
        elif isinstance(result, dict):
            if "rec_texts" in result:
                rec_texts = result["rec_texts"]
                rec_scores = result.get("rec_scores", [])
            elif "data" in result:
                for item in result["data"]:
                    if isinstance(item, dict) and "text" in item:
                        rec_texts.append(item["text"])
                        rec_scores.append(item.get("score", None))

        text = " ".join(rec_texts).strip()
        avg_score = sum([s for s in rec_scores if s is not None]) / len([s for s in rec_scores if s is not None]) if rec_scores else 0.0

        if not text:
            log_error(f"No text recognized for {image_path}")
            return "", avg_score

        return text, avg_score
    except Exception as e:
        log_error(f"OCR Failed for {image_path}: {str(e)}\n{traceback.format_exc()}")
        return "", 0.0

# ------------------ INFERENCE FUNCTION ------------------
def predict(images):
    """
    Args:
        images: list of PIL Images
    Returns:
        list of dicts: predicted_label_text, predicted_label_num, confidence_score, ocr_text
    """
    results = []

    # Perform OCR automatically using extract_text
    ocr_texts = []
    for img in images:
        text, score = extract_text(img)
        ocr_texts.append(text)

    # Batch processing
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i + BATCH_SIZE]
        batch_texts = ocr_texts[i:i + BATCH_SIZE]

        # Process images
        img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(DEVICE)

        # Tokenize text
        text_tensors = torch.stack([
            tokenize_text(t) for t in batch_texts
        ]).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensors, text_tensors)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        for j, prob in enumerate(probs):
            label_num = int(np.argmax(prob))
            label_text = LABELS[label_num]
            confidence_score = float(np.max(prob))
            results.append({
                "predicted_label_text": label_text,
                "predicted_label_num": label_num,
                "confidence_score": confidence_score,
                "ocr_text": batch_texts[j]  # include extracted OCR text
            })

    return results

# ------------------ EXAMPLE USAGE ------------------
if __name__ == "__main__":
    # Load images from upload dir
    uploaded_images = []
    for fname in os.listdir(IMAGE_UPLOAD_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            uploaded_images.append(Image.open(os.path.join(IMAGE_UPLOAD_DIR, fname)).convert("RGB"))

    predictions = predict(uploaded_images)
    for i, pred in enumerate(predictions):
        print(f"{uploaded_images[i].filename if hasattr(uploaded_images[i], 'filename') else i}: {pred}")
