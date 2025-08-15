# 01_data_generation.py
import os
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from .preprocess import deduplicate, save_csv, log_error
from utils.path import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_OUTPUT_DIR, LOG_DIR


# Optional PaddleOCR import
try:
    from paddleocr import PaddleOCR
    paddle_available = True
except ImportError:
    paddle_available = False

import pytesseract

# ========================
# Config & Paths
# ========================
RAW_DATA_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
IMAGE_OUTPUT_DIR = IMAGE_OUTPUT_DIR
LOG_DIR = LOG_DIR

image_extensions = [".jpg", ".png"]
labels_map = {"Neutral": 0, "Positive": 1, "Negative": 2}

# Choose OCR engine: "tesseract" or "paddle"
OCR_ENGINE = "paddle"  # Change to "tesseract" if preferred

# Initialize PaddleOCR if selected
if OCR_ENGINE == "paddle" and paddle_available:
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
elif OCR_ENGINE == "paddle" and not paddle_available:
    raise ImportError("PaddleOCR selected but not installed.")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ========================
# Helper Functions
# ========================

def get_all_images(raw_dir):
    """Get all image file paths recursively."""
    images = []
    for ext in image_extensions:
        images.extend(glob(os.path.join(raw_dir, f"**/*{ext}"), recursive=True))
    return images

def extract_text(image_path):
    """Extract text from image using the chosen OCR engine."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unreadable image")

        if OCR_ENGINE == "tesseract":
            text = pytesseract.image_to_string(img)
        elif OCR_ENGINE == "paddle":
            result = ocr_model.ocr(image_path, cls=True)
            text = " ".join([line[1][0] for page in result for line in page])
        else:
            raise ValueError(f"Unknown OCR engine: {OCR_ENGINE}")

        return text.strip()
    except Exception as e:
        log_error(f"Failed OCR on {image_path}: {str(e)}")
        return None

def assign_label(text):
    """Simple placeholder for labeling logic."""
    if not text:
        return "Neutral"
    # Expand with actual sentiment logic if needed
    return "Positive" if len(text) > 20 else "Neutral"

def rename_and_save_image(src_path, idx):
    """Rename image to a consistent format and copy to IMAGE_OUTPUT_DIR."""
    ext = os.path.splitext(src_path)[1]
    new_name = f"{idx:04d}{ext}"
    dst_path = os.path.join(IMAGE_OUTPUT_DIR, new_name)
    try:
        img = cv2.imread(src_path)
        if img is None:
            raise ValueError("Unreadable image")
        cv2.imwrite(dst_path, img)
        return new_name
    except Exception as e:
        log_error(f"Failed to save image {src_path}: {str(e)}")
        return None

# ========================
# Main Data Generation
# ========================

def main():
    images = get_all_images(RAW_DATA_DIR)
    print(f"Found {len(images)} images.")

    data = []
    idx = 1
    for img_path in tqdm(images, desc="Processing images"):
        text = extract_text(img_path)
        if text is None:
            continue
        label_text = assign_label(text)
        label_num = labels_map[label_text]

        new_image_name = rename_and_save_image(img_path, idx)
        if new_image_name is None:
            continue

        data.append({
            "image_name": new_image_name,
            "text": text,
            "label_text": label_text,
            "label_num": label_num
        })
        idx += 1

    print(f"Collected {len(data)} valid samples.")

    # Deduplicate
    cleaned_data = deduplicate(data)
    print(f"{len(cleaned_data)} samples after deduplication.")

    # Split into train/val/test (80/10/10)
    df = pd.DataFrame(cleaned_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    n = len(df)
    train_df = df[:int(0.8*n)]
    val_df = df[int(0.8*n):int(0.9*n)]
    test_df = df[int(0.9*n):]

    save_csv(train_df, os.path.join(PROCESSED_DATA_DIR, "train.csv"))
    save_csv(val_df, os.path.join(PROCESSED_DATA_DIR, "val.csv"))
    save_csv(test_df, os.path.join(PROCESSED_DATA_DIR, "test.csv"))

    print("CSV files saved to", PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()
