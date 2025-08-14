from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_module import VisualMultiScale
from .text_module import TextGuidedEncoder


@dataclass
class FGMFNConfig:
    # Visual
    vis_backbone: str = "resnet50"
    vis_pretrained: bool = True
    vis_embed_dim: int = 512
    vis_scales: tuple = (1, 2, 3)

    # Text
    text_model_name: str = "bert-base-uncased"
    text_visual_guidance: bool = True
    text_num_attn_layers: int = 1
    text_num_heads: int = 8
    text_dropout: float = 0.1
    text_freeze_encoder: bool = False

    # Fusion / classifier
    num_classes: int = 3
    dropout: float = 0.1


class ProjectionHead(nn.Module):
    """
    Projects embeddings to a shared space for matching / MI losses.
    """
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FGMFN(nn.Module):
    """
    Fine-Grained Multiscale Cross-Modal Sentiment Model.

    Outputs:
      - logits: [B, C]
      - vis_proj: [B, P]    (projected visual embedding for matching/MI)
      - txt_proj: [B, P]    (projected text embedding for matching/MI)
      - aux: dict with intermediate tensors
    """
    def __init__(self, cfg: Optional[FGMFNConfig] = None):
        super().__init__()
        cfg = cfg or FGMFNConfig()

        # Visual encoder
        self.visual = VisualMultiScale(
            backbone=cfg.vis_backbone,
            pretrained=cfg.vis_pretrained,
            embed_dim=cfg.vis_embed_dim,
            scales=list(cfg.vis_scales),
            add_positional_encoding=True,
        )

        # Text encoder with visual guidance
        self.text = TextGuidedEncoder(
            model_name=cfg.text_model_name,
            visual_guidance=cfg.text_visual_guidance,
            num_attn_layers=cfg.text_num_attn_layers,
            num_heads=cfg.text_num_heads,
            dropout=cfg.text_dropout,
            freeze_text_encoder=cfg.text_freeze_encoder,
        )

        hidden_size = self.text.hidden_size

        # Align visual global to text hidden size (if needed)
        self.vis_to_hidden = nn.Sequential(
            nn.Linear(cfg.vis_embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Classification head on guided text representation
        self.cls = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, cfg.num_classes),
        )

        # Projection heads for contrastive / matching / MI objectives
        proj_dim = 256
        self.txt_proj = ProjectionHead(hidden_size, proj_dim)
        self.vis_proj = ProjectionHead(hidden_size, proj_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            images: [B, 3, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            logits, vis_p, txt_p, aux
        """
        # Visual path
        vis_tokens, vis_global_raw = self.visual(images)           # [B,T,Dv], [B,Dv]
        vis_global = self.vis_to_hidden(vis_global_raw)            # [B, Dh]

        # Text path with visual guidance
        txt_seq, txt_pooled, txt_guided = self.text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_tokens=vis_tokens,           # keys/values for cross-attn
            vis_global=vis_global,           # global vector for gated fusion
        )

        # Classifier on guided representation
        logits = self.cls(txt_guided)                                # [B, C]

        # Projections for auxiliary losses
        txt_p = self.txt_proj(txt_guided)                            # [B, P]
        vis_p = self.vis_proj(vis_global)                            # [B, P]

        aux = {
            "txt_seq": txt_seq,                     # [B, L, Dh]
            "txt_pooled": txt_pooled,               # [B, Dh]
            "txt_guided": txt_guided,               # [B, Dh]
            "vis_tokens": vis_tokens,               # [B, T, Dv]
            "vis_global_raw": vis_global_raw,       # [B, Dv]
            "vis_global": vis_global,               # [B, Dh]
        }
        return logits, vis_p, txt_p, aux