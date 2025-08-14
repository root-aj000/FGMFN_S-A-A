from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class GatedFusion(nn.Module):
    """
    Gated fusion of text and a single visual vector:
        h = tanh(Wt*t + Wv*v)
        g = sigmoid(Wg*[t; v])
        out = g * h + (1 - g) * t
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.wt = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wg = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.wt(t) + self.wv(v))
        g = torch.sigmoid(self.wg(torch.cat([t, v], dim=-1)))
        return g * h + (1.0 - g) * t


class CrossModalAttention(nn.Module):
    """
    Cross-attention from text queries to visual tokens (keys/values).
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, txt_seq: torch.Tensor, vis_tokens: torch.Tensor) -> torch.Tensor:
        # txt_seq: [B, L, D], vis_tokens: [B, T, D]
        x = self.ln(txt_seq)
        y, _ = self.attn(query=x, key=vis_tokens, value=vis_tokens, need_weights=False)
        x = x + y
        z = self.ln2(x + self.ffn(self.ln(x)))
        return z


class TextGuidedEncoder(nn.Module):
    """
    Text encoder (HF Transformers) + visual guidance (cross-attn + gated fusion).
    Returns sequence embeddings, pooled [CLS]-style, and a guided pooled vector.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        visual_guidance: bool = True,
        num_attn_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = getattr(self.encoder.config, "hidden_size", 768)
        self.visual_guidance = visual_guidance

        self.cross_layers = nn.ModuleList([
            CrossModalAttention(self.hidden_size, num_heads=num_heads, dropout=dropout)
            for _ in range(max(1, num_attn_layers)) if visual_guidance
        ])

        self.gated_fusion = GatedFusion(self.hidden_size) if visual_guidance else None

        if freeze_text_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vis_tokens: Optional[torch.Tensor] = None,
        vis_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            vis_tokens: [B, T, D]   (optional if visual_guidance=False)
            vis_global: [B, D]      (optional if visual_guidance=False)
        Returns:
            txt_seq: [B, L, D]
            txt_pooled: [B, D]         (CLS pooled from backbone)
            txt_guided: [B, D]         (guided pooled vector)
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        txt_seq = out.last_hidden_state                  # [B, L, D]
        txt_pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None \
            else txt_seq[:, 0]                           # [B, D]

        if not self.visual_guidance or vis_tokens is None or vis_global is None:
            return txt_seq, txt_pooled, txt_pooled

        x = txt_seq
        for layer in self.cross_layers:
            x = layer(x, vis_tokens)                    # [B, L, D]

        # Pool the cross-attended sequence (simple masked mean)
        mask = attention_mask.unsqueeze(-1).float()     # [B, L, 1]
        masked = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        x_pooled = masked.sum(dim=1) / denom            # [B, D]

        guided = self.gated_fusion(x_pooled, vis_global)
        return x, txt_pooled, guided