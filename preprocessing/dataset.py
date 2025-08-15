# preprocessing/dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from image_preprocessing import resize_image, normalize_image
from text_preprocessing import clean_text, tokenize_text
from augmentation import augment_image
from utils.path import IMAGE_DIR , TRAIN_CSV

# Config / Paths
IMAGE_DIR = IMAGE_DIR
MAX_TEXT_LEN = 128  # Max length for tokenized text

class CustomDataset(Dataset):
    def __init__(self, csv_path, image_dir=IMAGE_DIR, augment=False):
        """
        Args:
            csv_path (str): Path to CSV file with columns: image_name, text, label_num
            image_dir (str): Directory containing images
            augment (bool): Apply image augmentation
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- Load and preprocess image ---
        image_path = os.path.join(self.image_dir, row["image_name"])
        img = cv2.imread(image_path)
        if img is None:
            # Return a black image if file missing/corrupted
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = resize_image(img)
            if self.augment:
                img = augment_image(img)
        img_tensor = normalize_image(img)

        # --- Preprocess text ---
        text = clean_text(row.get("text", ""))
        text_tensor = tokenize_text(text, max_length=MAX_TEXT_LEN)

        # --- Process label ---
        label_tensor = torch.tensor(row.get("label_num", 0), dtype=torch.long)

        return {
            "visual": img_tensor,        # [C,H,W]
            "text": text_tensor,         # [seq_len]
            "label": label_tensor        # scalar tensor
        }

# Example usage
if __name__ == "__main__":
    train_csv = TRAIN_CSV
    dataset = CustomDataset(train_csv, augment=True)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print({k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample.items()})
