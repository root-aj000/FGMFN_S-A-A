# preprocessing/dataset_preprocessing.py
import os
import cv2
import torch
import pandas as pd
from preprocessing.image_preprocessing import resize_image, normalize_image
from preprocessing.text_preprocessing import clean_text, tokenize_text
from preprocessing.augmentation import augment_image
from utils.path import PROCESSED_IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV


# Config / Paths
PROCESSED_IMAGE_DIR = PROCESSED_IMAGE_DIR
TRAIN_CSV = TRAIN_CSV
VAL_CSV = VAL_CSV
TEST_CSV = TEST_CSV

def process_dataset(csv_path, augment=False):
    """Load CSV, preprocess images and text, return dataset ready for PyTorch."""
    df = pd.read_csv(csv_path)
    processed_data = []

    for _, row in df.iterrows():
        image_path = os.path.join(PROCESSED_IMAGE_DIR, row["image_name"])
        img = cv2.imread(image_path)
        if img is None:
            continue

        img = resize_image(img)
        if augment:
            img = augment_image(img)
        img_tensor = normalize_image(img)

        text = clean_text(row["text"])
        token_ids = tokenize_text(text)

        processed_data.append({
            "image": img_tensor,
            "text_tokens": token_ids,
            "label": torch.tensor(row["label_num"], dtype=torch.long)
        })

    return processed_data

# Example usage
if __name__ == "__main__":
    train_data = process_dataset(TRAIN_CSV, augment=True)
    val_data = process_dataset(VAL_CSV, augment=False)
    test_data = process_dataset(TEST_CSV, augment=False)

    print(f"Processed train samples: {len(train_data)}")
    print(f"Processed val samples: {len(val_data)}")
    print(f"Processed test samples: {len(test_data)}")
