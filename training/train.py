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

from models.fg_mfn import FG_MFN, ATTRIBUTE_NAMES
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

# Determine which attributes are available for training
available_attributes = train_dataset.available_attributes if hasattr(train_dataset, 'available_attributes') else []
legacy_mode = train_dataset.legacy_mode if hasattr(train_dataset, 'legacy_mode') else True

print(f"Training mode: {'legacy (single label)' if legacy_mode else 'multi-attribute'}")
if not legacy_mode:
    print(f"Available attributes: {available_attributes}")

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
    all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
    all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["visual"].to(device)
        texts = batch["text"].to(device)
        masks = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            outputs = model(images, texts, attention_mask=masks)
            
            # Compute multi-task loss
            total_loss = 0
            num_attrs = 0
            
            if legacy_mode:
                # Backwards compatibility: single sentiment label
                labels = (batch["label"] - 1).to(device)
                if "sentiment" in outputs:
                    loss = criterion(outputs["sentiment"], labels)
                else:
                    # Use first available head
                    first_key = list(outputs.keys())[0]
                    loss = criterion(outputs[first_key], labels)
                total_loss = loss
            else:
                # Multi-attribute loss
                for attr in ATTRIBUTE_NAMES:
                    if attr in outputs and attr in batch:
                        labels = batch[attr].to(device)
                        loss = criterion(outputs[attr], labels)
                        total_loss += loss
                        num_attrs += 1
                        
                        # Track predictions
                        preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                        all_preds[attr].extend(preds)
                        all_labels[attr].extend(labels.cpu().numpy())
                
                if num_attrs > 0:
                    total_loss = total_loss / num_attrs  # Average loss

        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        train_losses.append(total_loss.item())

    avg_loss = np.mean(train_losses)
    
    # Compute per-attribute metrics
    metrics = {"loss": avg_loss}
    for attr in ATTRIBUTE_NAMES:
        if all_preds[attr]:
            metrics[f"{attr}_acc"] = accuracy_score(all_labels[attr], all_preds[attr])
            metrics[f"{attr}_f1"] = f1_score(all_labels[attr], all_preds[attr], average='weighted', zero_division=0)
    
    return metrics

# ------------------ VALIDATION FUNCTION ------------------
def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_losses = []
    all_preds = {attr: [] for attr in ATTRIBUTE_NAMES}
    all_labels = {attr: [] for attr in ATTRIBUTE_NAMES}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["visual"].to(device)
            texts = batch["text"].to(device)
            masks = batch["attention_mask"].to(device)

            outputs = model(images, texts, attention_mask=masks)
            
            # Compute multi-task loss
            total_loss = 0
            num_attrs = 0
            
            if legacy_mode:
                labels = (batch["label"] - 1).to(device)
                if "sentiment" in outputs:
                    loss = criterion(outputs["sentiment"], labels)
                else:
                    first_key = list(outputs.keys())[0]
                    loss = criterion(outputs[first_key], labels)
                total_loss = loss
            else:
                for attr in ATTRIBUTE_NAMES:
                    if attr in outputs and attr in batch:
                        labels = batch[attr].to(device)
                        loss = criterion(outputs[attr], labels)
                        total_loss += loss
                        num_attrs += 1
                        
                        preds = torch.argmax(outputs[attr], dim=1).cpu().numpy()
                        all_preds[attr].extend(preds)
                        all_labels[attr].extend(labels.cpu().numpy())
                
                if num_attrs > 0:
                    total_loss = total_loss / num_attrs

            val_losses.append(total_loss.item())

    avg_loss = np.mean(val_losses)
    
    metrics = {"loss": avg_loss}
    for attr in ATTRIBUTE_NAMES:
        if all_preds[attr]:
            metrics[f"{attr}_acc"] = accuracy_score(all_labels[attr], all_preds[attr])
            metrics[f"{attr}_f1"] = f1_score(all_labels[attr], all_preds[attr], average='weighted', zero_division=0)
    
    return metrics

# ------------------ MAIN TRAIN LOOP ------------------
def main():
    global best_val_loss, patience_counter
    scaler = torch.cuda.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE)

        # Log metrics
        log_data = {}
        for key, value in train_metrics.items():
            log_data[f"train_{key}"] = value
        for key, value in val_metrics.items():
            log_data[f"val_{key}"] = value
        logger.log_metrics(log_data, epoch)

        # Print summary
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Print per-attribute accuracies
        for attr in ATTRIBUTE_NAMES:
            if f"{attr}_acc" in train_metrics:
                print(f"  {attr}: Train Acc={train_metrics[f'{attr}_acc']:.3f}, Val Acc={val_metrics.get(f'{attr}_acc', 0):.3f}")

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
