import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pytesseract  # OCR

# ======== CONFIG ========
INPUT_FOLDER = r"C:\Users\Aj\Documents\ad_images"         # Input ads folder
OUTPUT_FOLDER = r"C:\Users\Aj\Documents\ad_dataset"       # Output dataset folder
IMG_EXT = ".jpg"                                          # ".jpg" or ".png"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
LABELS_TEXT = ["Neutral", "Positive"]  # No Negative
LABELS_NUM = {"Neutral": 1, "Positive": 2}
# Optional: Point to Tesseract executable (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ========================

def extract_text_from_image(img_path):
    """Extract text from image using Tesseract OCR."""
    try:
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"⚠ OCR failed for {img_path}: {e}")
        return ""

def prepare_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_out_folder = os.path.join(output_folder, "images")
    os.makedirs(images_out_folder, exist_ok=True)

    # Gather all image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("⚠ No images found in input folder.")
        return

    image_files.sort()
    data_rows = []

    print("📦 Processing images with OCR...")
    counter = 1
    for img_name in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(input_folder, img_name)
        new_name = f"{counter:04d}{IMG_EXT}"
        new_path = os.path.join(images_out_folder, new_name)

        # Save image to dataset folder
        img = Image.open(img_path).convert("RGB")
        img.save(new_path)

        # Random label assignment
        label_text = random.choice(LABELS_TEXT)
        label_num = LABELS_NUM[label_text]

        # OCR extraction
        text_from_img = extract_text_from_image(img_path)

        data_rows.append({
            "image_path": f"images/{new_name}",
            "text": text_from_img,
            "label_text": label_text,
            "label_num": label_num
        })
        counter += 1

    if not data_rows:
        print("✅ No new images to process.")
        return

    df = pd.DataFrame(data_rows)

    # Split into train, val, test
    train_df, temp_df = train_test_split(df, train_size=TRAIN_SPLIT, shuffle=True, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42)

    # Save CSV files
    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_folder, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

    print("\n✅ Dataset prepared successfully!")
    print(f"📂 Saved in: {output_folder}")
    print(f"📄 CSV Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    prepare_dataset(INPUT_FOLDER, OUTPUT_FOLDER)