# server/predict.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN
from utils.path import SAVED_MODEL_PATH , MODEL_CONFIG

# Optional: Tesseract OCR
import pytesseract

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
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ------------------ LOAD MODEL ------------------
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

model = FG_MFN(cfg).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------ INFERENCE FUNCTION ------------------
def predict(images):
    """
    Args:
        images: list of PIL Images
    Returns:
        list of dicts: predicted_label_text, predicted_label_num, confidence_score, ocr_text
    """
    results = []

    # Perform OCR automatically
    ocr_texts = [pytesseract.image_to_string(img).strip() for img in images]

    # Batch processing
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i+BATCH_SIZE]
        batch_texts = ocr_texts[i:i+BATCH_SIZE]

        # Process images
        img_tensors = torch.stack([transform(img) for img in batch_imgs]).to(DEVICE)

        # Tokenize text
        text_tensors = torch.stack([
            CustomDataset.tokenize_text(t, max_len=MAX_TEXT_LEN) for t in batch_texts
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
