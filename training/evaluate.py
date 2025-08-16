# training/evaluate.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN
import json
from utils.path import TEST_CSV, SAVED_MODEL_PATH, MODEL_CONFIG, LOG_DIR

# ------------------ PATHS ------------------
TEST_CSV = TEST_CSV 
MODEL_PATH = SAVED_MODEL_PATH 
MODEL_CONFIG = MODEL_CONFIG 
EVAL_LOG_DIR = LOG_DIR 

os.makedirs(EVAL_LOG_DIR, exist_ok=True)

# ------------------ HYPERPARAMS ------------------
BATCH_SIZE = 1  # 32 increase when add data
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ LOAD CONFIG ------------------
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

# ------------------ LOAD DATASET ------------------
test_dataset = CustomDataset(TEST_CSV)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  #4 increase when add data

# ------------------ LOAD MODEL ------------------
model = FG_MFN(cfg).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()





print("Device:", DEVICE)
print("Number of test samples:", len(test_dataset))
sample = next(iter(test_loader))
print("Batch keys:", sample.keys())
print("Sample shapes:", {k: v.shape for k, v in sample.items()})
# ------------------ EVALUATION LOOP ------------------
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        images = batch["visual"].to(DEVICE)
        texts = batch["text"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(images, texts)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# ------------------ METRICS ------------------
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# ------------------ SAVE EVALUATION REPORT ------------------
report_path = os.path.join(EVAL_LOG_DIR, "evaluation_report.csv")
report_df = pd.DataFrame({
    "metric": ["accuracy", "macro_f1"],
    "value": [accuracy, f1]
})
report_df.to_csv(report_path, index=False)

# ------------------ PLOT CONFUSION MATRIX ------------------
# cm_fig, ax = plt.subplots(figsize=(6,6))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Positive"])
# disp.plot(cmap=plt.cm.Blues, ax=ax)
# plt.title("Confusion Matrix")
# plt.savefig(os.path.join(EVAL_LOG_DIR, "confusion_matrix.png"))
# plt.close()

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
cm_fig, ax = plt.subplots(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neutral", "Positive"])
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(EVAL_LOG_DIR, "confusion_matrix.png"))
plt.close()