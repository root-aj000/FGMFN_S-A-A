import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingLoss(nn.Module):
    """
    Matching loss for aligning visual and textual embeddings.
    Can be used for image-text matching in cross-modal learning.
    """

    def __init__(self, margin: float = 0.2):
        super(MatchingLoss, self).__init__()
        self.margin = margin

    def forward(self, visual_embeds, text_embeds):
        """
        Args:
            visual_embeds: Tensor of shape (batch_size, embed_dim)
            text_embeds: Tensor of shape (batch_size, embed_dim)
        Returns:
            loss: scalar tensor
        """
        # Normalize embeddings
        visual_embeds = F.normalize(visual_embeds, dim=1)
        text_embeds = F.normalize(text_embeds, dim=1)

        # Similarity matrix
        sim_matrix = visual_embeds @ text_embeds.t()

        # Get diagonal (positive pairs)
        pos = sim_matrix.diag().view(-1, 1)

        # Compute ranking loss
        cost_text = (self.margin + sim_matrix - pos).clamp(min=0)
        cost_visual = (self.margin + sim_matrix.t() - pos).clamp(min=0)

        # Zero out diagonal
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        cost_text = cost_text.masked_fill(mask, 0)
        cost_visual = cost_visual.masked_fill(mask, 0)

        # Total loss
        loss = cost_text.sum() + cost_visual.sum()
        return loss / visual_embeds.size(0)