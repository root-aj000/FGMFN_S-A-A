# preprocessing/tokenizer.py
from transformers import BertTokenizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
