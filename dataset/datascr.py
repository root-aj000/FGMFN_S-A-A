import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import easyocr

# ======== CONFIG ========
INPUT_FOLDER = r"C:\Users\Aj\Documents\ad_images"   # Input ads folder
OUTPUT_FOLDER = r"C:\Users\Aj\Documents\data"       # Output dataset folder
IMG_EXT = ".jpg"                                    # ".jpg" or ".png"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
LABELS_TEXT = ["Neutral", "Positive"]               # No Negative
LABELS_NUM = {"Neutral": 1, "Positive": 2}
# ========================

# OCR reader
reader = easyocr.Reader(['en'])

def extract_text_from_image(image_path):
    """Extract visible text from the image using OCR."""
    try:
        return " ".join(reader.readtext(image_path, detail=0))
    except:
        return ""

def prepare_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images_folder = os.path.join(output_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Gather all image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("⚠ No images found in input folder.")
        return

    image_files.sort()
    data_rows = []

    print("📦 Processing images...")
    counter = 1
    for img_name in tqdm(image_files, desc="Renaming & OCR"):
        img_path = os.path.join(input_folder, img_name)
        new_name = f"{counter:04d}{IMG_EXT}"

        # Assign random label
        label_text = random.choice(LABELS_TEXT)
        label_num = LABELS_NUM[label_text]

        # Extract text from image
        ocr_text = extract_text_from_image(img_path)

        # Save image to central images folder
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(images_folder, new_name))

        data_rows.append({
            "image_name": new_name,
            "label_text": label_text,
            "label_num": label_num,
            "text": ocr_text
        })

        counter += 1

    df = pd.DataFrame(data_rows)

    # Split into train, val, test
    train_df, temp_df = train_test_split(df, train_size=TRAIN_SPLIT, shuffle=True, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42)

    # Save CSVs with all columns
    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_folder, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

    print("\n✅ Dataset prepared successfully!")
    print(f"📂 Images saved in: {images_folder}")
    print(f"📄 CSV files saved in: {output_folder}")

if __name__ == "__main__":
    prepare_dataset(INPUT_FOLDER, OUTPUT_FOLDER)