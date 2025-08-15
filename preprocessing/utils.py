import os
import csv
import logging
from PIL import Image
import pytesseract
from torchvision import transforms
from typing import List, Tuple
from utils.path import LOG_DIR

# ------------------ PATHS ------------------
LOG_DIR = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------ LOGGING ------------------
def log_error(message: str, path: str = os.path.join(LOG_DIR, "errors.log")):
    logging.basicConfig(
        filename=path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.error(message)

# ------------------ CSV ------------------
def save_csv(data: List[dict], path: str, fieldnames: List[str]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    except Exception as e:
        log_error(f"Failed to save CSV {path}: {str(e)}")

# ------------------ OCR ------------------
def extract_text_from_image(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        log_error(f"OCR failed: {str(e)}")
        return ""

# ------------------ Deduplication ------------------
def deduplicate_images(text_list: List[str], image_list: List[Image.Image]) -> Tuple[List[str], List[Image.Image]]:
    unique_texts = []
    unique_images = []
    seen_texts = set()
    for text, img in zip(text_list, image_list):
        if text not in seen_texts:
            unique_texts.append(text)
            unique_images.append(img)
            seen_texts.add(text)
    return unique_texts, unique_images

# ------------------ Image Augmentation ------------------
def augment_image(image: Image.Image) -> Image.Image:
    try:
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomResizedCrop(size=image.size[0], scale=(0.9,1.0))
        ])
        return augment_transform(image)
    except Exception as e:
        log_error(f"Image augmentation failed: {str(e)}")
        return image

# ------------------ Tokenization Helper ------------------
def tokenize_text(text: str, tokenizer, max_len: int = 128):
    try:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        return encoding["input_ids"].squeeze(0)
    except Exception as e:
        log_error(f"Text tokenization failed: {str(e)}")
        return None
