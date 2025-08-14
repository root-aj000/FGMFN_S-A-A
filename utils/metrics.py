import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return (correct / labels.size(0)) * 100.0


def retrieval_recall(sim_matrix: torch.Tensor, k: int = 5) -> float:
    """
    Compute Recall@K for retrieval tasks.
    sim_matrix: similarity scores (N x N)
    """
    topk = sim_matrix.topk(k, dim=1)[1]
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device).view(-1, 1)
    hits = (topk == labels).sum().item()
    return hits / sim_matrix.size(0)