import re


def clean_text(text: str) -> str:
    """
    Lowercase, remove special characters, and extra spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(file_path: str):
    """
    Load dataset split file with tab-separated values.
    Returns list of (image_path, text, label)
    """
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                samples.append((parts[0], parts[1], int(parts[2])))
    return samples