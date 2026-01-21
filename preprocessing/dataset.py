# preprocessing/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from preprocessing.image_preprocessing import resize_image, normalize_image
from preprocessing.text_preprocessing import clean_text, tokenize_text
from preprocessing.augmentation import augment_image
from utils.path import IMAGE_DIR, TRAIN_CSV

# Config / Paths
IMAGE_DIR = IMAGE_DIR
MAX_TEXT_LEN = 128  # Max length for tokenized text

# Attribute names that we train on (must match model config)
ATTRIBUTE_NAMES = [
    "theme", "sentiment", "emotion", "dominant_colour", "attention_score",
    "trust_safety", "target_audience", "predicted_ctr", "likelihood_shares"
]

class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir=IMAGE_DIR, augment=False):
        """
        Args:
            csv_path (str): Path to CSV file with columns: image_path, text, and attribute columns
            image_dir (str): Directory containing images
            augment (bool): Apply image augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.augment = augment
        
        # Determine which attributes are present in the CSV
        self.available_attributes = []
        for attr in ATTRIBUTE_NAMES:
            col_name = f"{attr}_num"
            if col_name in self.df.columns:
                self.available_attributes.append(attr)
        
        # Backwards compatibility: check for old label_num column
        self.legacy_mode = "label_num" in self.df.columns and len(self.available_attributes) == 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load and preprocess image ---
        image_path = os.path.join(self.image_dir, row["image_path"])
        img = cv2.imread(image_path)
        if img is None:
            # Return a black image if file missing/corrupted
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = resize_image(img)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augment:
                img = augment_image(img)
        img_tensor = normalize_image(img)

        # --- Preprocess text ---
        text = clean_text(row.get("text", ""))
        text_tensor = tokenize_text(text, max_length=MAX_TEXT_LEN)

        # Build output dict
        output = {
            "visual": img_tensor,        # [C,H,W]
            "text": text_tensor["input_ids"],         # [seq_len]
            "attention_mask": text_tensor["attention_mask"], # [seq_len]
        }
        
        # --- Process labels ---
        if self.legacy_mode:
            # Backwards compatibility: single label
            output["label"] = torch.tensor(row.get("label_num", 0), dtype=torch.long)
        else:
            # Multi-attribute labels
            for attr in ATTRIBUTE_NAMES:
                col_name = f"{attr}_num"
                if col_name in self.df.columns:
                    output[attr] = torch.tensor(row.get(col_name, 0), dtype=torch.long)
                else:
                    # Default to 0 if attribute not in CSV
                    output[attr] = torch.tensor(0, dtype=torch.long)

        return output

# Example usage
if __name__ == "__main__":
    train_csv = TRAIN_CSV
    dataset = CustomDataset(train_csv, augment=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Available attributes: {dataset.available_attributes}")
    print(f"Legacy mode: {dataset.legacy_mode}")

    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample.items()})
