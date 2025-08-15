import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd

class AdvertisementDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, tokenizer=None, max_length=64, use_numeric_labels=True):
        """
        data_path/
            images/
            train.csv
            val.csv
            test.csv
        """
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

        # Load CSV instead of TXT
        csv_file = os.path.join(data_path, f"{split}.csv")
        df = pd.read_csv(csv_file)

        # Store data
        self.image_names = df["image_name"].tolist()
        self.texts = df["text"].fillna("").tolist()
        if use_numeric_labels:
            self.labels = df["label_num"].tolist()
        else:
            self.labels = df["label_text"].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, "images", self.image_names[idx])

        # Load image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenize text
        text = self.texts[idx]
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

        label = self.labels[idx]
        if self.use_numeric_labels:
            label = torch.tensor(int(label), dtype=torch.long)

        return {"visual": image, "text": text_tensor, "label": label}