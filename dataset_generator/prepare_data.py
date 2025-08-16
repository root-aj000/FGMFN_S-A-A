import os
import csv
import random
import logging
import traceback
from sklearn.model_selection import train_test_split
from paddlex import create_pipeline
from PIL import Image
from utils.path import *

# ---------------- PATH CONFIG ---------------- #
RAW_DATA_DIR = RAW_DATA_DIR
PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
IMAGE_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, "images")  # store images here
LOG_DIR = LOG_DIR

# Create directories
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------- SETTINGS ---------------- #
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1

# ---------------- LOGGING ---------------- #
log_file_path = os.path.join(LOG_DIR, "dataset_prep.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(message)s",
    force=True
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

log_info = lambda msg: (logger.info(msg), print(msg, flush=True))
log_error = lambda msg: (logger.error(msg), print(msg, flush=True))

# ---------------- OCR MODEL ---------------- #
ocr_model = create_pipeline(pipeline="ocr")

# ---------------- FUNCTIONS ---------------- #
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

def extract_text(image_path):
    try:
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
        avg_score = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0

        if not text:
            log_error(f"No text recognized for {image_path}")
            return None

        return text, avg_score
    except Exception as e:
        log_error(f"OCR Failed for {image_path}: {str(e)}\n{traceback.format_exc()}")
        return None

def save_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text", "score", "label_text", "label_num", "original_image"])
        writer.writerows(rows)
    log_info(f"Saved CSV: {csv_path}")

def process_split(split_data, split_label, start_index):
    rows = []
    for idx, (_, text, score, label_text, label_num, image_url) in enumerate(split_data, start=start_index):
        new_filename = f"{split_label}_{idx:04d}.jpg"
        dest_path = os.path.join(IMAGE_OUTPUT_DIR, new_filename)

        try:
            log_info(f"Processing image: {image_url}")
            with Image.open(image_url) as img:
                img.convert("RGB").save(dest_path, "JPEG")
        except Exception as e:
            log_error(f"Error copying {image_url}: {e}")
            continue

        # Store only filename in CSV
        rows.append((new_filename, text, score, label_text, label_num, image_url))
    return rows

# ---------------- MAIN ---------------- #
def prepare_dataset():
    log_info("Scanning for images...")
    image_files = [
        os.path.join(RAW_DATA_DIR, f)
        for f in os.listdir(RAW_DATA_DIR)
        if os.path.isfile(os.path.join(RAW_DATA_DIR, f))
    ]
    log_info(f"Found {len(image_files)} images.")

    data_samples = []

    for img_path in image_files:
        if not is_valid_image(img_path):
            log_error(f"Invalid or unreadable image: {img_path}")
            continue

        result = extract_text(img_path)
        if result:
            text, score = result
            label_text = random.choice(["positive", "neutral"])
            label_num = 2 if label_text == "positive" else 1
            data_samples.append((None, text, score, label_text, label_num, img_path))
        else:
            log_error(f"No valid text for {img_path}")

    log_info(f"Collected {len(data_samples)} valid samples.")

    if not data_samples:
        log_error("No data samples collected. Exiting.")
        return

    # Split dataset
    train_val, test = train_test_split(data_samples, test_size=TEST_SPLIT, random_state=42)
    train, val = train_test_split(train_val, test_size=VAL_SPLIT, random_state=42)

    index_counter = 1
    train_rows = process_split(train, "train", index_counter)
    index_counter += len(train_rows)
    val_rows = process_split(val, "val", index_counter)
    index_counter += len(val_rows)
    test_rows = process_split(test, "test", index_counter)

    save_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), train_rows)
    save_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), val_rows)
    save_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), test_rows)

    avg_score = sum(row[2] for row in (train_rows + val_rows + test_rows)) / len(train_rows + val_rows + test_rows)
    log_info(f"Average OCR confidence score: {avg_score:.4f}")

if __name__ == "__main__":
    prepare_dataset()
