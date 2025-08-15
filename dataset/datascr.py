import os
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ======== CONFIG ========
INPUT_FOLDER = r"C:\Users\Aj\Documents\ad_images"         # Input ads folder
OUTPUT_FOLDER = r"C:\Users\Aj\Documents\ad_dataset"       # Output dataset folder
IMG_EXT = ".jpg"                                          # ".jpg" or ".img"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
LABELS_TEXT = ["Neutral", "Positive"]  # No Negative
LABELS_NUM = {"Neutral": 1, "Positive": 2}
# ========================

def prepare_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Gather all image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print("⚠ No images found in input folder.")
        return

    image_files.sort()
    data_rows = []

    # Track already processed images
    existing_names = set()
    for split in ["train", "val", "test"]:
        split_csv = os.path.join(output_folder, split, "annotations_text.csv")
        if os.path.exists(split_csv):
            try:
                existing_df = pd.read_csv(split_csv)
                existing_names.update(existing_df["image_name"].tolist())
            except:
                pass

    print("📦 Processing images...")
    counter = 1
    for img_name in tqdm(image_files, desc="Renaming"):
        # Skip if already processed
        if any(img_name in existing for existing in existing_names):
            continue

        img_path = os.path.join(input_folder, img_name)
        new_name = f"{counter:04d}{IMG_EXT}"

        label_text = random.choice(LABELS_TEXT)
        label_num = LABELS_NUM[label_text]

        data_rows.append({
            "image_name": new_name,
            "label_text": label_text,
            "label_num": label_num,
            "original_file": img_name
        })
        counter += 1

    if not data_rows:
        print("✅ No new images to process. Dataset already up-to-date.")
        return

    df = pd.DataFrame(data_rows)

    # Split into train, val, test
    train_df, temp_df = train_test_split(df, train_size=TRAIN_SPLIT, shuffle=True, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=VAL_SPLIT / (1 - TRAIN_SPLIT), random_state=42)

    # Save images & annotations
    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        split_folder = os.path.join(output_folder, split_name, "images")
        os.makedirs(split_folder, exist_ok=True)

        for _, row in split_df.iterrows():
            img = Image.open(os.path.join(input_folder, row["original_file"])).convert("RGB")
            img.save(os.path.join(split_folder, row["image_name"]))

        # Save both annotation files
        split_df[["image_name", "label_text"]].to_csv(
            os.path.join(output_folder, split_name, "annotations_text.csv"), index=False
        )
        split_df[["image_name", "label_num"]].to_csv(
            os.path.join(output_folder, split_name, "annotations_numeric.csv"), index=False
        )

    print("\n✅ Dataset prepared successfully!")
    print(f"📂 Saved in: {output_folder}")

if __name__ == "__main__":
    prepare_dataset(INPUT_FOLDER, OUTPUT_FOLDER)