import os
import torch
from models.fg_mfn import FGMFN
from PIL import Image
import torchvision.transforms as transforms
import easyocr
from transformers import BertTokenizer
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = FGMFN(embed_dim=256)
model.load_state_dict(torch.load(
    os.path.join("saved_models", "fgmfn_sentiment.pth"),
    map_location=device
))
model.to(device)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# OCR reader
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    """Extract text from an image using EasyOCR."""
    return " ".join(reader.readtext(image_path, detail=0))

def preprocess_image(image_path):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def preprocess_text(text):
    """Tokenize text for BERT."""
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return encoding["input_ids"], encoding["attention_mask"]

def predict_ad_sentiment(image_path):
    """Predict sentiment for a single ad image."""
    text = extract_text_from_image(image_path)
    img_tensor = preprocess_image(image_path).to(device)
    input_ids, attention_mask = preprocess_text(text)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

    with torch.no_grad():
        logits = model(img_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[pred_label], confidence, text

def predict_from_csv(csv_path, images_root):
    """
    Run predictions for all images listed in a CSV file.
    CSV must have a column 'image_path' with relative paths.
    """
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        image_path = os.path.join(images_root, row["image_path"])
        sentiment, confidence, text = predict_ad_sentiment(image_path)
        results.append({
            "image_path": row["image_path"],
            "extracted_text": text,
            "predicted_sentiment": sentiment,
            "confidence": confidence
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Example: single image
    sentiment, confidence, extracted_text = predict_ad_sentiment("data/images/sample_ad.jpg")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print(f"Extracted Text: {extracted_text}")

    # Example: batch prediction from CSV
    results_df = predict_from_csv("data/test.csv", "data")
    print(results_df.head())