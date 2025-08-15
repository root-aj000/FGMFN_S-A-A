Perfect! I can now map your **step-by-step modules** into the **production-ready folder structure** you provided. Here’s a detailed integration, showing which file lives where, their tasks, and expected outputs for **clarity and reproducibility**.

---

# **Step-by-Step Modules in Production Structure**

---

### **Step 1: Data Generation**

**File:** `01_data_generation.py` 
**Tasks:**

* Collect images in `data/raw/`.
* Run OCR (Tesseract/PaddleOCR) → extract text.
* Assign sentiment labels (Neutral, Positive).
* Deduplicate images & text (`deduplication.py`).
* Handle missing/unreadable images.
* Generate CSVs (`train.csv`, `val.csv`, `test.csv`) in `data/processed/`:

```text
image_name, text, label_text, label_num
```

* Rename images to a consistent format (`image_name` → `0001.jpg`  ).

**Data Types:**

* Images: `.jpg`, `.png`
* Text: String
* Labels: Text + Numeric
* CSVs: UTF-8 encoded

##  Data Generation (preprocessing)

Paths:

RAW_DATA_DIR = "data/raw/"                   # Original datasets
PROCESSED_DATA_DIR = "data/processed/"      # CSV outputs and cleaned data
IMAGE_OUTPUT_DIR = "data/images/"           # Renamed and saved images
LOG_DIR = "data/logs/"                      # Logs, duplicates, missing reports


Variables:

image_extensions = [".jpg", ".png"]         # Allowed image types
ocr_engine = "pytesseract"                  # OCR engine choice
labels = {"Neutral": 0, "Positive": 1}     # Sentiment label mapping


Functions / Outputs:

extract_text(image_path) -> str → OCR text

deduplicate(images_list, texts_list) -> cleaned_list

save_csv(dataframe, split_name) → saves train.csv, val.csv, test.csv
---

### **Step 2: Data Preprocessing**

**File:** `02_data_preprocessing.py` → `preprocessing/preprocessing.py` + `preprocessing/augmentation.py`
**Tasks:**

* Resize images (e.g., 224×224).
* Normalize images (mean/std for pretrained CNNs).
* Clean OCR text: nulls, whitespace, lowercase, remove special chars.
* Tokenize text (BERT tokenizer / char embeddings) → `tokenizer.py`.
* Optional augmentation: flip, rotate, crop.

**Output:** Clean dataset ready for PyTorch Dataset.

## Data Preprocessing (preprocessing/preprocessing.py & preprocessing/augmentation.py)

Paths:

PROCESSED_IMAGE_DIR = "data/images/"
TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"


Variables:

IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


Functions / Outputs:

resize_image(image) -> image_resized

normalize_image(image) -> normalized_tensor

clean_text(text) -> cleaned_text

augment_image(image) -> augmented_image

tokenize_text(text) -> token_ids


---

### **Step 3: Dataset Class**

**File:** `03_dataset.py` → `preprocessing/dataset.py`
**Tasks:**

* Custom PyTorch Dataset:

  * Loads CSV + images from `data/images/`.
  * Tokenizes text.
  * Converts labels to tensors.
* Returns dictionary:

```python
{
  "visual": image_tensor,
  "text": text_tensor,
  "label": label_tensor
}
```

* Handles missing/corrupted files gracefully.

## Dataset Class (preprocessing/dataset.py)

Paths:

IMAGE_DIR = "data/images/"
CSV_PATH = "data/processed/train.csv"   # Can switch to val/test


Variables:

MAX_TEXT_LEN = 128                 # Max length for tokenized text


Dataset Output:

{
  "visual": image_tensor,          # torch.FloatTensor [C,H,W]
  "text": text_tensor,             # torch.LongTensor [seq_len]
  "label": label_tensor            # torch.LongTensor scalar
}












---

### **Step 4: Model Architecture**

**File:** `04_model.py` → `models/fg_mfn.py`, `models/visual_module.py`, `models/text_module.py`
**Tasks:**

* Multi-modal:

  * CNN backbone for images (ResNet50/efficientnet)
  * Text encoder (BERT/LSTM/Transformer)
  * Fusion layer: Concatenation / Attention
  * Classifier → 2 classes (Neutral, Positive)
* Configurable hyperparameters in `models/configs/model_config.json`.


## Model Architecture (models/*.py)

Paths:

MODEL_CONFIG = "models/configs/model_config.json"
SAVED_MODEL_DIR = "models/saved_models/"


Variables:

IMAGE_BACKBONE = "resnet50"       # CNN backbone
TEXT_ENCODER = "bert-base-uncased" # Text encoder
FUSION_TYPE = "concat"            # Fusion layer type
NUM_CLASSES = 2                   # Neutral, Positive


Outputs:

forward(image_tensor, text_tensor) -> logits (shape: [batch, NUM_CLASSES])




---

### **Step 5: Training**

**File:** `05_train.py` → `training/train.py`
**Tasks:**

* Load datasets (train/val).
* Define optimizer, scheduler, loss function (CrossEntropy).
* Training loop:

  * Forward + backward pass
  * Validation metrics per epoch
  * Early stopping & checkpoint saving (`models/saved_models/checkpoint_epoch_*.pt`)
* Save final model → `models/saved_models/model_final.pt`
* Save training logs → `training/logs/` (loss, accuracy, F1-score).

## Training (training/train.py)

Paths:

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
SAVED_MODEL_DIR = "models/saved_models/"
LOG_DIR = "training/logs/"


Variables:

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5


Outputs:

checkpoint_epoch_{n}.pt

model_final.pt

TensorBoard logs, CSV logs











---

### **Step 6: Evaluation**

**File:** `06_evaluate.py` → `training/evaluate.py`
**Tasks:**

* Load test set + trained model.
* Compute metrics: Accuracy, Confusion Matrix, Macro F1 score.
* Save evaluation report & plots → `data/logs/`.



## Evaluation (training/evaluate.py)

Paths:

TEST_CSV = "data/processed/test.csv"
MODEL_PATH = "models/saved_models/model_final.pt"
EVAL_LOG_DIR = "data/logs/"


Outputs:

Accuracy, F1 score, confusion matrix

Evaluation report → CSV/PNG







---

### **Step 7: Inference**

**File:**  `server/predict.py`
**Tasks:**

* Load model.
* Accept image(s) + optional text.
* Return predictions:

```python
predicted_label_text, predicted_label_num, confidence_score
```

* Handle multiple images in batch.

---

## Inference (server/predict.py)

Paths:

MODEL_PATH = "models/saved_models/model_final.pt"
IMAGE_UPLOAD_DIR = "data/images/tmp_uploads/"


Variables:

BATCH_SIZE = 16


Outputs:

{
  "predicted_label_text": "Neutral",
  "predicted_label_num": 0,
  "confidence_score": 0.85
}









### **Step 8: Backend Server**

**File:** `08_server.py` → `server/app.py`
**Tasks:**

* Flask/FastAPI backend.
* Endpoint `/predict` → accepts image upload → returns JSON predictions.
* Handles:

  * Multiple images
  * Logging errors
* Connects frontend → backend.

## Backend Server (server/app.py)

Paths & Variables:

MODEL_PATH = "models/saved_models/model_final.pt"
UPLOAD_FOLDER = "data/images/tmp_uploads/"
ALLOWED_EXTENSIONS = ["jpg", "png"]
API_ENDPOINT = "/predict"


Outputs:

JSON with predictions

Logs errors in data/logs/









---

### **Step 9: Frontend**

**Folder:** `web/`
**Tasks:**

* HTML page to upload images (`index.html`).
* Calls backend API (`flask`).
* Displays:

  * Image thumbnail
  * OCR extracted text
  * Predicted sentiment + confidence
* Optional: Batch summary & charts.




## Frontend (web/)

Variables / Paths:

API_URL = "http://localhost:8000/predict"


Outputs:

Displays image thumbnail, OCR text, predicted sentiment + confidence

Optional batch summary & charts





---

### **Step 10: Utilities**

**File:** `utils.py` → `preprocessing/utils.py` + `server/utils.py`
**Tasks:**

* OCR extraction
* Deduplication of images/text
* Tokenization helpers
* Logging & CSV handling
* Image augmentation helpers

## Utilities (preprocessing/utils.py & server/utils.py)

Functions & Paths:

save_csv(df, path) → CSV file

log_error(message, path="data/logs/errors.log")

deduplicate_images(text_list, image_list)

tokenize_text(text, tokenizer)

augment_image(image)





---

### **Data Flow Summary**

```text
Raw images → OCR + Label → CSVs (train/val/test)
CSV + Images → Preprocessing → PyTorch Dataset
Dataset → Model → Training → Saved Model
Saved Model → Evaluation → Metrics
Saved Model → Inference → Predictions → Backend API
Frontend → API → Display predictions
```

---

### **Production-Ready Details**

* **Logging:** All scripts log to `data/logs/`.
* **Error Handling:** Corrupted images, missing OCR, invalid labels handled gracefully.
* **Config Management:** `configs/default.yaml` for paths, splits, hyperparameters, preprocessing.
* **Reproducibility:** Fixed random seed for splits & label assignment.
* **Versioning:** Model version + config saved in `models/`.
* **Dependencies:** `requirements.txt` includes PyTorch, Transformers, OCR, Flask/FastAPI, etc.

---

✅ This now **fully maps your step-by-step modules to the production folder structure**, making it clear where each module belongs and what it does.

Next, I can **generate all the starter `.py` files with empty classes/functions and comments**, ready for you to fill in logic, keeping this structure intact.

Do you want me to do that?
