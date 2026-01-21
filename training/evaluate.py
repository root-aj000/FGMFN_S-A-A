# training/evaluate.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Determine mode
legacy_mode = test_dataset.legacy_mode if hasattr(test_dataset, 'legacy_mode') else True

print(f"Evaluation mode: {'legacy (single label)' if legacy_mode else 'multi-attribute'}")

# ------------------ LOAD MODEL ------------------
model = FG_MFN(cfg).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Device:", DEVICE)
print("Number of test samples:", len(test_dataset))

# ------------------ EVALUATION LOOP ------------------
all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}

with torch.no_grad():
    for batch in test_loader:
        images = batch["visual"].to(DEVICE)
        texts = batch["text"].to(DEVICE)
        masks = batch["attention_mask"].to(DEVICE)

        outputs = model(images, texts, attention_mask=masks)
        
        if legacy_mode:
            # Backwards compatibility
            labels = batch["label"]
            if "sentiment" in outputs:
                preds = torch.argmax(outputs["sentiment"], dim=1).cpu().numpy()
            else:
                first_key = list(outputs.keys())[0]
                preds = torch.argmax(outputs[first_key], dim=1).cpu().numpy()
            all_preds["sentiment"].extend(preds)
            all_labels["sentiment"].extend(labels.cpu().numpy())
        else:
            # Multi-attribute evaluation
            for attr in ATTRIBUTE_NAMES:
                if attr in outputs and attr in batch:
                    preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                    all_preds[attr].extend(preds)
                    all_labels[attr].extend(batch[attr].cpu().numpy())

# ------------------ METRICS ------------------
report_data = []

for attr in ATTRIBUTE_NAMES:
    if all_preds[attr]:
        accuracy = accuracy_score(all_labels[attr], all_preds[attr])
        f1 = f1_score(all_labels[attr], all_preds[attr], average='macro', zero_division=0)
        
        print(f"\n{attr.upper()}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {f1:.4f}")
        
        report_data.append({
            "attribute": attr,
            "accuracy": accuracy,
            "macro_f1": f1
        })
        
        # Get label names for confusion matrix
        if "ATTRIBUTES" in cfg and attr in cfg["ATTRIBUTES"]:
            label_names = cfg["ATTRIBUTES"][attr]["labels"]
            num_classes = cfg["ATTRIBUTES"][attr]["num_classes"]
        else:
            label_names = None
            num_classes = len(set(all_labels[attr]))
        
        # Compute and save confusion matrix
        cm = confusion_matrix(all_labels[attr], all_preds[attr], labels=range(num_classes))
        
        cm_fig, ax = plt.subplots(figsize=(8, 8))
        if label_names:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f"Confusion Matrix - {attr}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_LOG_DIR, f"confusion_matrix_{attr}.png"))
        plt.close()

# ------------------ SAVE EVALUATION REPORT ------------------
report_path = os.path.join(EVAL_LOG_DIR, "evaluation_report.csv")
report_df = pd.DataFrame(report_data)
report_df.to_csv(report_path, index=False)
print(f"\nEvaluation report saved to: {report_path}")

# Summary
if report_data:
    avg_accuracy = np.mean([r["accuracy"] for r in report_data])
    avg_f1 = np.mean([r["macro_f1"] for r in report_data])
    print(f"\n=== OVERALL ===")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Macro F1: {avg_f1:.4f}")