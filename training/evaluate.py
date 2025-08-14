import torch
from torch.utils.data import DataLoader

from models.fg_mfn import FGMFN
from utils.dataset import AdvertisementDataset
from utils.metrics import compute_accuracy


def evaluate(
    checkpoint_path: str,
    data_path: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Dataset & loader
    val_dataset = AdvertisementDataset(data_path, split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = FGMFN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            visuals, texts, labels = (
                batch["visual"].to(device),
                batch["text"].to(device),
                batch["label"].to(device),
            )

            _, _, logits = model(visuals, texts)
            total_acc += compute_accuracy(logits, labels)

    avg_acc = total_acc / len(val_loader)
    print(f"Validation Accuracy: {avg_acc:.2f}%")


if __name__ == "__main__":
    evaluate(
        checkpoint_path="checkpoints/fgmfn_best.pth",
        data_path="data/ytb_ads"
    )