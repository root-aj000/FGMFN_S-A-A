import torch
import torch.nn as nn
import torch.nn.functional as F


class MutualInfoLoss(nn.Module):
    """
    Mutual Information Maximization Loss
    Based on InfoNCE objective.
    """

    def __init__(self, temperature: float = 0.07):
        super(MutualInfoLoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_embeds, text_embeds):
        """
        Args:
            visual_embeds: (batch, dim)
            text_embeds: (batch, dim)
        """
        batch_size = visual_embeds.size(0)

        # Normalize embeddings
        visual_embeds = F.normalize(visual_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)

        # Similarity matrix
        logits = torch.mm(visual_embeds, text_embeds.t()) / self.temperature

        labels = torch.arange(batch_size, device=visual_embeds.device)

        # Cross-entropy for both directions
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.t(), labels)

        return (loss_v2t + loss_t2v) / 2