# training/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json

from preprocessing.dataset import CustomDataset
from models.fg_mfn import FG_MFN
from training.logger import Logger
from utils.path import TRAIN_CSV, VAL_CSV, SAVED_MODEL_DIR, MODEL_CONFIG


# ------------------ PATHS ------------------
TRAIN_CSV = TRAIN_CSV
VAL_CSV = VAL_CSV
SAVED_MODEL_DIR = SAVED_MODEL_DIR
MODEL_CONFIG = MODEL_CONFIG

# ------------------ HYPERPARAMS ------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# ------------------ LOAD CONFIG ------------------
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

# ------------------ DATASETS ------------------
train_dataset = CustomDataset(TRAIN_CSV, max_len=MAX_TEXT_LEN)
val_dataset = CustomDataset(VAL_CSV, max_len=MAX_TEXT_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------ MODEL ------------------
model = FG_MFN(cfg).to(DEVICE)

# ------------------ LOSS & OPTIMIZER ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# ------------------ LOGGER ------------------
logger = Logger()

# ------------------ EARLY STOPPING ------------------
best_val_loss = np.inf
patience_counter = 0

# ------------------ TRAINING LOOP ------------------
for epoch in range(1, EPOCHS+1):
    # ----- TRAIN -----
    model.train()
    train_losses = []
    all_preds, all_labels = [], []

    for batch in train_loader:
        images = batch["visual"].to(DEVICE)
        texts = batch["text"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_train_loss = np.mean(train_losses)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    # ----- VALIDATION -----
    model.eval()
    val_losses = []
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["visual"].to(DEVICE)
            texts = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = np.mean(val_losses)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    # ----- LOGGING -----
    logger.log_metrics({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_f1": train_f1,
        "val_f1": val_f1
    }, epoch)

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} "
          f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    # ----- CHECKPOINT -----
    checkpoint_path = os.path.join(SAVED_MODEL_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # ----- EARLY STOPPING -----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, "model_final.pt"))
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

logger.close()
