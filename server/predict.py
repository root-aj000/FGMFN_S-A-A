# server/predict.py
import os
import re
import json
import torch
import numpy as np
import traceback
from PIL import Image
from torchvision import transforms
from paddlex import create_pipeline

from preprocessing.text_preprocessing import tokenize_text
from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
from utils.path import SAVED_MODEL_PATH, MODEL_CONFIG

# ------------------ OCR MODEL ------------------
ocr_model = create_pipeline(pipeline="ocr")

# ------------------ PATHS ------------------
MODEL_PATH = SAVED_MODEL_PATH
MODEL_CONFIG_PATH = MODEL_CONFIG
IMAGE_UPLOAD_DIR = "data/images/tmp_uploads/"

# ------------------ HYPERPARAMS ------------------
BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LEN = 128

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ LOAD CONFIG ------------------
try:
    with open(MODEL_CONFIG_PATH, "r") as f:
        CFG = json.load(f)
except Exception as e:
    print(f"[WARNING] Failed to load model config: {e}", flush=True)
    CFG = {}

# ------------------ LOAD MODEL (LAZY) ------------------
model = None


def load_model():
    global model
    if model is not None:
        return model

    try:
        loaded_model = FG_MFN(CFG).to(DEVICE)

        if os.path.exists(MODEL_PATH):
            loaded_model.load_state_dict(
                torch.load(MODEL_PATH, map_location=DEVICE)
            )
        else:
            print("[WARNING] Model checkpoint not found. Using random weights.")

        loaded_model.eval()
        model = loaded_model
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        loaded_model = FG_MFN(CFG).to(DEVICE)
        loaded_model.eval()
        model = loaded_model
        return model


# ------------------ LABEL MAPS ------------------
def get_label_maps():
    label_maps = {}
    attributes = CFG.get("ATTRIBUTES", {})
    for attr, cfg in attributes.items():
        label_maps[attr] = cfg.get("labels", [])
    return label_maps


LABEL_MAPS = get_label_maps()

# ------------------ TEXT UTILITIES ------------------
def extract_keywords(text):
    if not text:
        return ""

    stopwords = {
        "the", "a", "an", "is", "are", "and", "or", "to", "for",
        "of", "in", "on", "at", "with", "your", "you", "we",
        "our", "this", "that"
    }

    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    keywords = [w.capitalize() for w in words if w.lower() not in stopwords]

    seen = set()
    unique = []
    for w in keywords:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)

    return " ".join(unique[:5])


def extract_monetary_mention(text):
    if not text:
        return "None"

    patterns = [
        r"\d+%\s*(?:OFF|off|discount)",
        r"(?:Rs\.?|INR|USD|\$|₹)\s*\d+(?:,\d{3})*(?:\.\d{2})?",
        r"(?:FREE|Free|free)",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)

    return "None"


def extract_call_to_action(text):
    if not text:
        return "None"

    patterns = [
        r"(?:Buy|Shop|Order|Get|Grab|Claim)\s*(?:Now|Today)?",
        r"(?:Limited\s*Offer|Hurry|Act\s*Now)",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(0)

    return "None"


def extract_objects_mentioned(text):
    if not text:
        return "Unknown"

    mapping = {
        "Phone": r"\b(phone|iphone|mobile)\b",
        "Laptop": r"\b(laptop|computer|pc)\b",
        "Food": r"\b(food|burger|coffee|drink)\b",
        "Clothing": r"\b(shirt|dress|jeans)\b",
    }

    found = []
    text = text.lower()
    for k, p in mapping.items():
        if re.search(p, text):
            found.append(k)

    return ", ".join(found) if found else "General"


# ------------------ OCR ------------------
def extract_text(image):
    try:
        if isinstance(image, Image.Image):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(tmp.name)
            path = tmp.name
        else:
            path = image

        result = list(ocr_model.predict(path))

        texts, scores = [], []
        if result and isinstance(result[0], dict):
            texts = result[0].get("rec_texts", [])
            scores = result[0].get("rec_scores", [])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return " ".join(texts).strip(), avg_score

    except Exception:
        return "", 0.0


# ------------------ INFERENCE ------------------
def predict(images):
    results = []

    ocr_texts = []
    for img in images:
        text, _ = extract_text(img)
        ocr_texts.append(text)

    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i + BATCH_SIZE]
        batch_texts = ocr_texts[i:i + BATCH_SIZE]

        img_tensor = torch.stack(
            [transform(img) for img in batch_imgs]
        ).to(DEVICE)

        tokens = [tokenize_text(t) for t in batch_texts]
        text_ids = torch.stack([t["input_ids"] for t in tokens]).to(DEVICE)
        masks = torch.stack([t["attention_mask"] for t in tokens]).to(DEVICE)

        model_instance = load_model()
        with torch.no_grad():
            outputs = model_instance(
                img_tensor,
                text_ids,
                attention_mask=masks
            )

        for j in range(len(batch_imgs)):
            result = {"ocr_text": batch_texts[j]}

            primary_label = None
            primary_idx = None
            primary_conf = None

            for attr in ATTRIBUTE_NAMES:
                if attr not in outputs:
                    continue

                probs = torch.softmax(outputs[attr][j], dim=0)
                idx = int(torch.argmax(probs))
                conf = float(torch.max(probs))

                labels = LABEL_MAPS.get(attr, [])
                label = labels[idx] if idx < len(labels) else str(idx)

                result[attr] = label
                result[f"{attr}_confidence"] = conf

                if primary_label is None:
                    primary_label = label
                    primary_idx = idx
                    primary_conf = conf

            # legacy fields for tests
            result["predicted_label_text"] = primary_label or "Unknown"
            result["predicted_label_num"] = primary_idx if primary_idx is not None else -1
            result["confidence_score"] = primary_conf if primary_conf else 0.0

            result["keywords"] = extract_keywords(batch_texts[j])
            result["monetary_mention"] = extract_monetary_mention(batch_texts[j])
            result["call_to_action"] = extract_call_to_action(batch_texts[j])
            result["object_detected"] = extract_objects_mentioned(batch_texts[j])

            results.append(result)

    return results
