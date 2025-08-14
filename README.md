
# FGMFN: Fine-Grained Multiscale Cross-Modal Sentiment Analysis

This project predicts sentiment in advertisements by analyzing both **images** and **text**.

---

## 📂 Folder & File Overview

Below is a description of each file and folder, along with example usage.

---

### 1. `data/` — Datasets
- **ytb_ads/** — YouTube Ads dataset  
- **twitter2015/** — Twitter dataset (2015)  
- **twitter2017/** — Twitter dataset (2017)  

**Example dataset format (`train.txt`):**

images/ad1.jpg	This product is amazing!	2 images/ad2.jpg	Not worth the money.	0

- First column: image path  
- Second column: ad text  
- Third column: sentiment label (0 = negative, 1 = neutral, 2 = positive)  

---

### 2. `models/` — Model Components

#### `fg_mfn.py` — Core FGMFN model combining image & text features
```python
from models.fg_mfn import FGMFN
model = FGMFN(embed_dim=256)

visual_module.py — Multi-scale image feature extractor
from models.visual_module import VisualFeatureExtractor
visual_model = VisualFeatureExtractor()

text_module.py — Text encoder with visual guidance
from models.text_module import TextEncoder
text_model = TextEncoder(model_name="bert-base-uncased")

3. losses/ — Custom Loss Functions
matching_loss.py — Matches image & text features if they belong together
from losses.matching_loss import MatchingLoss
loss_fn = MatchingLoss(margin=0.2)

mutual_info_loss.py — Encourages information sharing between image & text
from losses.mutual_info_loss import MutualInfoLoss
loss_fn = MutualInfoLoss()

4. training/ — Training & Evaluation
train.py — Trains the FGMFN model
python training/train.py --config configs/default.yaml

evaluate.py — Tests the trained model
python training/evaluate.py --config configs/default.yaml

5. utils/ — Helper Functions
dataset.py — Loads image + text dataset into PyTorch
from utils.dataset import AdvertisementDataset
dataset = AdvertisementDataset("data/ytb_ads", split="train")

preprocessing.py — Cleans and preprocesses text
from utils.preprocessing import clean_text
print(clean_text("This Product!!! is Awesome..."))  # 'this product is awesome'

metrics.py — Calculates evaluation metrics
from utils.metrics import compute_accuracy
acc = compute_accuracy(pred_logits, true_labels)

6. configs/ — Project Configurations
default.yaml — Controls dataset paths, training settings, and model parameters
training:
  epochs: 10
  batch_size: 32
model:
  embed_dim: 256

7. Root Files
 * requirements.txt — Python dependencies
 * README.md — Documentation (this file)
🚀 How to Run the Project
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python training/train.py --config configs/default.yaml

3️⃣ Evaluate the model
python training/evaluate.py --config configs/default.yaml

If this isn't what you meant, could you clarify what you mean by "don't appear neatly in md"?
