import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AdvertisementDataset(Dataset):
    """
    Custom dataset for advertisement sentiment analysis.
    Expects:
        data_path/
            train.csv (or val.csv / test.csv) with columns: image_name,label_text,label_num,text
            images/
    """

    def __init__(self, data_path, split="train", transform=None, tokenizer=None, max_length=64, use_numeric_labels=True):
        self.data_path = data_path
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_numeric_labels = use_numeric_labels

        csv_file = os.path.join(data_path, f"{split}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV split file not found: {csv_file}")

        # Read CSV into dataframe
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_name"]
        text = self.df.iloc[idx].get("text", "")  # Default to empty string if no OCR
        if self.use_numeric_labels:
            label_val = self.df.iloc[idx]["label_num"]
        else:
            label_val = self.df.iloc[idx]["label_text"]

        img_full_path = os.path.join(self.data_path, "images", img_name)

        # Load and preprocess image
        image = Image.open(img_full_path).convert("RGB")
        image = self.transform(image)

        # Tokenize text
        if self.tokenizer:
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            text_tensor = text_tokens["input_ids"].squeeze(0)
        else:
            text_tensor = torch.tensor([ord(c) for c in text[:self.max_length]], dtype=torch.long)

        # Convert label to tensor
        if self.use_numeric_labels:
            label = torch.tensor(int(label_val), dtype=torch.long)
        else:
            # If label is text, keep as string (no tensor)
            label = label_val

        return {
            "visual": image,
            "text": text_tensor,
            "label": label
        }