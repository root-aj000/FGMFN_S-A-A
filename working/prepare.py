import os
import csv
import random
import logging
import traceback
from pathlib import Path
from sklearn.model_selection import train_test_split
from paddlex import create_pipeline
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log_info = logging.info
log_error = logging.error

# Create OCR model
ocr_model = create_pipeline(pipeline="ocr")

# Paths
RAW_DIR = Path("data/raw")
DATASET_DIR = Path("data/dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_CSV_DIR = Path("data")
OUTPUT_CSV_DIR.mkdir(exist_ok=True)

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def is_valid_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def extract_text(image_path: Path):
    try:
        result = list(ocr_model.predict(str(image_path)))

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
        avg_score = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0

        if not text:
            log_error(f"No text recognized for {image_path}")
            return None

        return text, avg_score
    except Exception as e:
        log_error(f"OCR Failed for {image_path}: {str(e)}\n{traceback.format_exc()}")
        return None

def save_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text", "score", "label_text", "label_num", "image_url"])
        writer.writerows(rows)
    log_info(f"Saved CSV: {path}")

def prepare_dataset():
    image_files = list(RAW_DIR.glob("*.*"))
    data_samples = []

    log_info(f"Found {len(image_files)} images")

    for img_path in image_files:
        if not is_valid_image(img_path):
            log_error(f"Invalid or unreadable image: {img_path}")
            continue

        result = extract_text(img_path)
        if result:
            text, score = result

            # Random label
            label_text = random.choice(["positive", "neutral"])
            label_num = 1 if label_text == "positive" else 2

            # Keep original path for image_url
            image_url = str(img_path)

            data_samples.append((None, text, score, label_text, label_num, image_url))
        else:
            log_error(f"No valid text for {img_path}")

    if not data_samples:
        log_error("No data samples collected. Exiting.")
        return

    # Split into train, val, test
    train_val, test = train_test_split(data_samples, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    # Helper to copy and rename images while preserving URL
    def process_split(split_data, split_dir):
        rows = []
        for idx, (old_name, text, score, label_text, label_num, image_url) in enumerate(split_data, start=1):
            new_filename = f"{idx:04d}.jpg"
            dest_path = split_dir / new_filename

            # Copy file from image_url
            try:
                Image.open(image_url).save(dest_path)
            except Exception as e:
                log_error(f"Error copying {image_url}: {e}")
                continue

            rows.append((str(dest_path), text, score, label_text, label_num, image_url))
        return rows

    train_rows = process_split(train, TRAIN_DIR)
    val_rows = process_split(val, VAL_DIR)
    test_rows = process_split(test, TEST_DIR)

    save_csv(OUTPUT_CSV_DIR / "train.csv", train_rows)
    save_csv(OUTPUT_CSV_DIR / "val.csv", val_rows)
    save_csv(OUTPUT_CSV_DIR / "test.csv", test_rows)

if __name__ == "__main__":
    prepare_dataset()
