# training/train.py
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from models.fg_mfn import FG_MFN
from training.logger import Logger
from utils.path import TRAIN_CSV, VAL_CSV, SAVED_MODEL_DIR, MODEL_CONFIG
from preprocessing.dataset import CustomDataset

# ------------------ CONFIG ------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)

os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# ------------------ LOAD MODEL CONFIG ------------------
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

# ------------------ DATA ------------------
train_dataset = CustomDataset(TRAIN_CSV)
val_dataset = CustomDataset(VAL_CSV)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------ MODEL ------------------
model = FG_MFN(cfg).to(DEVICE)

# ------------------ LOSS & OPTIMIZER ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ------------------ LOGGER ------------------
LOG_DIR = os.path.join(SAVED_MODEL_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = Logger(LOG_DIR)

# ------------------ EARLY STOPPING ------------------
best_val_loss = np.inf
patience_counter = 0

# ------------------ TRAINING FUNCTION ------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    train_losses = []
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["visual"].to(device)
        texts = batch["text"].to(device)
        labels = (batch["label"] - 1).to(device)


        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(train_losses)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

# ------------------ VALIDATION FUNCTION ------------------
def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_losses = []
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["visual"].to(device)
            texts = batch["text"].to(device)
            labels = (batch["label"] - 1).to(device)

            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(val_losses)
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')
    return avg_loss, acc, f1

# ------------------ MAIN TRAIN LOOP ------------------
def main():
    global best_val_loss, patience_counter
    scaler = torch.cuda.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE)

        # Log metrics
        logger.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "val_f1": val_f1
        }, epoch)

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # Scheduler step
        scheduler.step(val_loss)

        # Save last checkpoint
        torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, "model_last.pt"))

        # Early stopping and best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVED_MODEL_DIR, "model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    logger.close()

if __name__ == "__main__":
    main()
