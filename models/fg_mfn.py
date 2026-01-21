# models/fg_mfn.py
import json
import torch
import torch.nn as nn
from models.visual_module import VisualModule
from models.text_module import TextModule
from utils.path import MODEL_CONFIG

# Attribute names for multi-head classification
ATTRIBUTE_NAMES = [
    "theme", "sentiment", "emotion", "dominant_colour", "attention_score",
    "trust_safety", "target_audience", "predicted_ctr", "likelihood_shares"
]

class FG_MFN(nn.Module):
    def __init__(self, cfg):
        super(FG_MFN, self).__init__()
        freeze = cfg.get("FREEZE_BACKBONE", False)
        
        self.visual_module = VisualModule(backbone=cfg["IMAGE_BACKBONE"], out_features=cfg["HIDDEN_DIM"], freeze=freeze)
        self.text_module = TextModule(encoder_name=cfg["TEXT_ENCODER"], out_features=cfg["HIDDEN_DIM"], freeze=freeze)
        self.fusion_type = cfg.get("FUSION_TYPE", "concat")
        
        fusion_dim = cfg["HIDDEN_DIM"] * 2 if self.fusion_type == "concat" else cfg["HIDDEN_DIM"]
        
        # Shared hidden layer
        self.shared_fc = nn.Sequential(
            nn.Linear(fusion_dim, cfg["HIDDEN_DIM"]),
            nn.ReLU(),
            nn.Dropout(cfg["DROPOUT"])
        )
        
        # Multi-head classifiers for each attribute
        self.attribute_heads = nn.ModuleDict()
        self.cfg = cfg
        
        if "ATTRIBUTES" in cfg:
            # New multi-attribute config
            for attr_name in ATTRIBUTE_NAMES:
                if attr_name in cfg["ATTRIBUTES"]:
                    num_classes = cfg["ATTRIBUTES"][attr_name]["num_classes"]
                    self.attribute_heads[attr_name] = nn.Linear(cfg["HIDDEN_DIM"], num_classes)
        else:
            # Backwards compatibility: single sentiment classifier
            self.attribute_heads["sentiment"] = nn.Linear(cfg["HIDDEN_DIM"], cfg.get("NUM_CLASSES", 2))

    def forward(self, image_tensor, text_tensor, attention_mask=None):
        visual_feat = self.visual_module(image_tensor)       # [batch, HIDDEN_DIM]
        text_feat = self.text_module(text_tensor, attention_mask=attention_mask)  # [batch, HIDDEN_DIM]

        if self.fusion_type == "concat":
            fused = torch.cat([visual_feat, text_feat], dim=1)
        else:
            fused = visual_feat + text_feat  # simple addition as example

        # Pass through shared layer
        shared_out = self.shared_fc(fused)
        
        # Get logits from each head
        outputs = {}
        for attr_name, head in self.attribute_heads.items():
            outputs[attr_name] = head(shared_out)
        
        return outputs  # dict of {attr_name: [batch, num_classes]}
    
    def get_label_names(self, attr_name):
        """Get the label names for a given attribute."""
        if "ATTRIBUTES" in self.cfg and attr_name in self.cfg["ATTRIBUTES"]:
            return self.cfg["ATTRIBUTES"][attr_name]["labels"]
        return None

# Example usage
if __name__ == "__main__":
    with open(MODEL_CONFIG, "r") as f:
        cfg = json.load(f)
    model = FG_MFN(cfg)
    img = torch.randn(2, 3, 224, 224)          # dummy image batch
    text = torch.randint(0, 1000, (2, 128))    # dummy token ids
    outputs = model(img, text)
    print("Output keys:", outputs.keys())
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
