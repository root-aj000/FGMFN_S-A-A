import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv


def _remove_classification_head(m: nn.Module) -> nn.Sequential:
    # Works for common torchvision CNNs (ResNet, EfficientNet, etc.)
    if isinstance(m, tv.ResNet):
        return nn.Sequential(*list(m.children())[:-2])  # up to C5 feature map
    if hasattr(m, "features"):  # EfficientNet, MobileNet
        return m.features
    # Fallback: drop the last linear layers if present
    modules = list(m.children())
    while modules and isinstance(modules[-1], (nn.Linear, nn.Dropout, nn.AdaptiveAvgPool2d, nn.Flatten)):
        modules.pop()
    return nn.Sequential(*modules)


def build_visual_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Returns a conv feature extractor (no classifier) and its output channels.
    """
    name = name.lower()
    if name == "resnet50":
        net = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat = _remove_classification_head(net)
        out_ch = 2048
    elif name == "resnet18":
        net = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = _remove_classification_head(net)
        out_ch = 512
    elif name == "efficientnet_b0":
        net = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = _remove_classification_head(net)
        out_ch = 1280
    else:
        raise ValueError(f"Unsupported backbone '{name}'. Try: resnet50, resnet18, efficientnet_b0.")
    return feat, out_ch


class MultiScalePool(nn.Module):
    """
    Pools a feature map to multiple grid sizes, projects each to a common dim,
    flattens to a sequence of tokens, and concatenates all scales.
    """
    def __init__(self, in_ch: int, embed_dim: int, scales: List[int]):
        super().__init__()
        self.scales = scales
        self.proj = nn.ModuleList([
            nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False) for _ in scales
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in scales
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, C, H, W]
        Returns:
            tokens: [B, N_tokens, D] where N_tokens = sum(s*s for s in scales)
        """
        batch_tokens = []
        for s, proj, ln in zip(self.scales, self.proj, self.norm):
            pooled = F.adaptive_avg_pool2d(feat, (s, s))      # [B,C,s,s]
            x = proj(pooled)                                  # [B,D,s,s]
            x = x.flatten(2).transpose(1, 2)                  # [B, s*s, D]
            x = ln(x)
            batch_tokens.append(x)
        return torch.cat(batch_tokens, dim=1)                 # [B, sum(s*s), D]


class VisualMultiScale(nn.Module):
    """
    Visual encoder:
      backbone -> feature map -> multiscale pooling -> visual tokens + global vector
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 512,
        scales: Optional[List[int]] = None,
        add_positional_encoding: bool = True,
    ):
        super().__init__()
        if scales is None:
            scales = [1, 2, 3]  # 1x1, 2x2, 3x3 -> 14 tokens total
        self.backbone, out_ch = build_visual_backbone(backbone, pretrained)
        self.msp = MultiScalePool(in_ch=out_ch, embed_dim=embed_dim, scales=scales)
        self.add_pos = add_positional_encoding
        if add_positional_encoding:
            n_tokens = sum(s * s for s in scales)
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)

        # global vector (GAP + projection)
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            vis_tokens: [B, T, D]  (T depends on scales)
            vis_global: [B, D]
        """
        feat = self.backbone(images)                   # [B, C, H', W']
        tokens = self.msp(feat)                        # [B, T, D]
        if self.add_pos:
            tokens = tokens + self.pos
        g = self.global_proj(feat)                     # [B, D]
        return tokens, g