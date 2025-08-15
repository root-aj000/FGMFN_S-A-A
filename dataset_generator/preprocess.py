# preprocessing/utils.py
import pandas as pd
import logging
import os

def save_csv(df, path):
    """Save DataFrame to CSV with UTF-8 encoding."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Saved CSV: {path}")

def log_error(message, path="data/logs/errors.log"):
    """Log errors to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(filename=path, level=logging.ERROR)
    logging.error(message)

def deduplicate(data_list):
    """Remove duplicate text or image_name entries."""
    seen_texts = set()
    cleaned = []
    for item in data_list:
        if item["text"] not in seen_texts:
            seen_texts.add(item["text"])
            cleaned.append(item)
    return cleaned
