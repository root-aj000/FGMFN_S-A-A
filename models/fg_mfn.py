# models/fg_mfn.py
import json
import torch
import torch.nn as nn
from models.visual_module import VisualModule
from models.text_module import TextModule
from utils.path import MODEL_CONFIG

# Load config
MODEL_CONFIG = MODEL_CONFIG
with open(MODEL_CONFIG, "r") as f:
    cfg = json.load(f)

class FG_MFN(nn.Module):
    def __init__(self, cfg):
        super(FG_MFN, self).__init__()
        self.visual_module = VisualModule(backbone=cfg["IMAGE_BACKBONE"], out_features=cfg["HIDDEN_DIM"])
        self.text_module = TextModule(encoder_name=cfg["TEXT_ENCODER"], out_features=cfg["HIDDEN_DIM"])
        self.fusion_type = cfg.get("FUSION_TYPE", "concat")
        
        fusion_dim = cfg["HIDDEN_DIM"] * 2 if self.fusion_type == "concat" else cfg["HIDDEN_DIM"]
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg["HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Dropout(cfg["DROPOUT"]),
            nn.Linear(cfg["HIDDEN_DIM"], cfg["NUM_CLASSES"])
        )

    def forward(self, image_tensor, text_tensor, attention_mask=None):
        visual_feat = self.visual_module(image_tensor)       # [batch, HIDDEN_DIM]
        text_feat = self.text_module(text_tensor, attention_mask)  # [batch, HIDDEN_DIM]

        if self.fusion_type == "concat":
            fused = torch.cat([visual_feat, text_feat], dim=1)
        else:
            fused = visual_feat + text_feat  # simple addition as example

        logits = self.classifier(fused)
        return logits  # [batch, NUM_CLASSES]

# Example usage
if __name__ == "__main__":
    model = FG_MFN(cfg)
    img = torch.randn(2, 3, 224, 224)          # dummy image batch
    text = torch.randint(0, 1000, (2, 128))    # dummy token ids
    logits = model(img, text)
    print("Logits shape:", logits.shape)
