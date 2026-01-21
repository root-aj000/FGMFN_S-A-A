import os
import csv
import random
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
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

# ---------------- ATTRIBUTE CONFIGURATION ---------------- #
# These must match the model_config.json
ATTRIBUTE_LABELS = {
    "theme": ["Food", "Fashion", "Tech", "Health", "Travel", "Finance", "Entertainment", "Sports", "Education", "Other"],
    "sentiment": ["Positive", "Negative", "Neutral"],
    "emotion": ["Excitement", "Trust", "Joy", "Fear", "Anger", "Sadness", "Surprise", "Anticipation"],
    "dominant_colour": ["Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Black", "White", "Brown", "Multi"],
    "attention_score": ["High", "Medium", "Low"],
    "trust_safety": ["Safe", "Unsafe", "Questionable"],
    "target_audience": ["General", "Food Lovers", "Tech Enthusiasts", "Fashionistas", "Parents", "Professionals", "Fitness Enthusiasts", "Students"],
    "predicted_ctr": ["High", "Medium", "Low"],
    "likelihood_shares": ["High", "Medium", "Low"]
}

ATTRIBUTE_NAMES = list(ATTRIBUTE_LABELS.keys())

# CSV Headers - all columns
CSV_HEADERS = [
    "image_path", "text", "score",
    # Trainable attributes (label + num)
    "theme", "theme_num",
    "sentiment", "sentiment_num", 
    "emotion", "emotion_num",
    "dominant_colour", "dominant_colour_num",
    "attention_score", "attention_score_num",
    "trust_safety", "trust_safety_num",
    "target_audience", "target_audience_num",
    "predicted_ctr", "predicted_ctr_num",
    "likelihood_shares", "likelihood_shares_num",
    # Text-extracted fields
    "keywords", "monetary_mention", "call_to_action", "object_detected",
    # Original path for reference
    "original_image"
]

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
        valid_scores = [s for s in rec_scores if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        if not text:
            log_error(f"No text recognized for {image_path}")
            return None

        return text, avg_score
    except Exception as e:
        log_error(f"OCR Failed for {image_path}: {str(e)}\n{traceback.format_exc()}")
        return None

def generate_random_labels():
    """Generate random labels for all attributes (for initial dataset creation).
    
    NOTE: In production, you should manually label your data instead of using random labels.
    This function is provided for demonstration/testing purposes.
    """
    labels = {}
    for attr, label_list in ATTRIBUTE_LABELS.items():
        idx = random.randint(0, len(label_list) - 1)
        labels[attr] = label_list[idx]
        labels[f"{attr}_num"] = idx
    return labels

def save_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    log_info(f"Saved CSV: {csv_path}")

def process_split(split_data, split_label, start_index):
    rows = []
    for idx, sample in enumerate(split_data, start=start_index):
        new_filename = f"{split_label}_{idx:04d}.jpg"
        dest_path = os.path.join(IMAGE_OUTPUT_DIR, new_filename)

        try:
            log_info(f"Processing image: {sample['original_image']}")
            with Image.open(sample['original_image']) as img:
                img.convert("RGB").save(dest_path, "JPEG")
        except Exception as e:
            log_error(f"Error copying {sample['original_image']}: {e}")
            continue

        # Update image path to new filename
        row = sample.copy()
        row["image_path"] = new_filename
        rows.append(row)
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

    def process_image(img_path):
        if not is_valid_image(img_path):
            log_error(f"Invalid or unreadable image: {img_path}")
            return None

        result = extract_text(img_path)
        if result:
            text, score = result
            
            # Generate random labels (REPLACE WITH MANUAL LABELING IN PRODUCTION)
            labels = generate_random_labels()
            
            sample = {
                "image_path": None,  # Will be set during split processing
                "text": text,
                "score": score,
                "original_image": img_path,
                # Text-extracted fields (placeholder - replace with actual extraction)
                "keywords": "",
                "monetary_mention": "None",
                "call_to_action": "None",
                "object_detected": "General",
            }
            sample.update(labels)
            return sample
        else:
            return None

    # Sequential processing to prevent PaddleX threading issues
    for img_path in image_files:
        res = process_image(img_path)
        if res:
            data_samples.append(res)

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

    all_rows = train_rows + val_rows + test_rows
    avg_score = sum(row["score"] for row in all_rows) / len(all_rows) if all_rows else 0
    log_info(f"Average OCR confidence score: {avg_score:.4f}")
    
    log_info(f"\n=== Dataset Preparation Complete ===")
    log_info(f"Train samples: {len(train_rows)}")
    log_info(f"Val samples: {len(val_rows)}")
    log_info(f"Test samples: {len(test_rows)}")
    log_info(f"\nNOTE: Labels were randomly generated for demonstration.")
    log_info(f"For actual training, please manually label your data in the CSV files.")

if __name__ == "__main__":
    prepare_dataset()
