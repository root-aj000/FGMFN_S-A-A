# preprocessing/text_preprocessing.py
import re
from transformers import BertTokenizer

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def clean_text(text):
    """Clean OCR text: lowercase, strip, remove special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_text(text, max_length=128):
    """Tokenize text using BERT tokenizer and return tensor of token ids."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding["input_ids"].squeeze(0)
