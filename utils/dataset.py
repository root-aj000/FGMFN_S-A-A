import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class AdvertisementDataset(Dataset):
    """
    Custom dataset for advertisement sentiment analysis.
    Expects:
        data_path/
            train.txt  (or val.txt / test.txt)  --> image_path \t text \t label
            images/
    """

    def __init__(self, data_path, split="train", transform=None, tokenizer=None, max_length=64):
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

        # Mapping for string labels to integers
        self.label_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }

        split_file = os.path.join(data_path, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            self.samples = [line.strip().split("\t") for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, label_val = self.samples[idx]
        img_full_path = os.path.join(self.data_path, img_path)

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

        # Convert label to integer
        try:
            # If label is already numeric string
            label = int(label_val)
        except ValueError:
            # If label is a string like "Positive"
            if label_val not in self.label_map:
                raise ValueError(f"Label '{label_val}' not recognized in label_map.")
            label = self.label_map[label_val]

        return {
            "visual": image,
            "text": text_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }