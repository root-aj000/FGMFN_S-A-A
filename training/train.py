import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.fg_mfn import FGMFN
from utils.dataset import AdvertisementDataset
from losses.matching_loss import MatchingLoss
from losses.mutual_info_loss import MutualInfoLoss
from utils.metrics import compute_accuracy


def train(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Dataset & loader
    train_dataset = AdvertisementDataset(data_path, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = FGMFN().to(device)

    # Losses
    matching_loss_fn = MatchingLoss()
    mi_loss_fn = MutualInfoLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0

        for batch in train_loader:
            visuals, texts, labels = (
                batch["visual"].to(device),
                batch["text"].to(device),
                batch["label"].to(device),
            )

            optimizer.zero_grad()

            # Forward pass
            visual_embeds, text_embeds, logits = model(visuals, texts)

            # Losses
            loss_match = matching_loss_fn(visual_embeds, text_embeds)
            loss_mi = mi_loss_fn(visual_embeds, text_embeds)
            loss_ce = nn.CrossEntropyLoss()(logits, labels)

            loss = loss_match + loss_mi + loss_ce
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, labels)

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Acc: {avg_acc:.2f}%")

    # Save model after training
    torch.save(model.state_dict(), "fgmfn_sentiment.pth")
    print("Model saved as fgmfn_sentiment.pth")


if __name__ == "__main__":
    train(data_path="data/ytb_ads")