---

FGMFN: Fine-Grained Multiscale Cross-Modal Sentiment Analysis

This project analyzes advertisements using both images and text to predict sentiment.


---

📂 Folder & File Guide with Examples


---

1. data/ – Your datasets live here

ytb_ads/ — YouTube ads dataset

twitter2015/ — Twitter dataset (2015)

twitter2017/ — Twitter dataset (2017)


Example dataset file (train.txt):

images/ad1.jpg	This product is amazing!	2
images/ad2.jpg	Not worth the money.	0

Here:

images/ad1.jpg → image path

This product is amazing! → text

2 → label (e.g., 0 = negative, 1 = neutral, 2 = positive)



---

2. models/ – All model parts

File	Purpose	Example Use

fg_mfn.py	Main FGMFN model combining image & text features	


from models.fg_mfn import FGMFN  
model = FGMFN(embed_dim=256)
``` |
| `visual_module.py` | Extracts multi-scale image features |  
```python
from models.visual_module import VisualFeatureExtractor  
visual_model = VisualFeatureExtractor()
``` |
| `text_module.py` | Encodes text using a BERT-like model |  
```python
from models.text_module import TextEncoder  
text_model = TextEncoder(model_name="bert-base-uncased")
``` |

---

### **3. `losses/`** – Custom loss functions  

| File | Purpose | Example Use |
|------|---------|-------------|
| `matching_loss.py` | Makes image & text features match if they belong together |  
```python
from losses.matching_loss import MatchingLoss  
loss_fn = MatchingLoss(margin=0.2)
``` |
| `mutual_info_loss.py` | Increases information sharing between image & text |  
```python
from losses.mutual_info_loss import MutualInfoLoss  
loss_fn = MutualInfoLoss()
``` |

---

### **4. `training/`** – Train & evaluate the model  

| File | Purpose | Example Use |
|------|---------|-------------|
| `train.py` | Trains the FGMFN model |  
```bash
python training/train.py --config configs/default.yaml
``` |
| `evaluate.py` | Tests the trained model on a dataset |  
```bash
python training/evaluate.py --config configs/default.yaml
``` |

---

### **5. `utils/`** – Helper tools  

| File | Purpose | Example Use |
|------|---------|-------------|
| `dataset.py` | Loads image + text dataset into PyTorch |  
```python
from utils.dataset import AdvertisementDataset  
dataset = AdvertisementDataset("data/ytb_ads", split="train")
``` |
| `preprocessing.py` | Cleans and preprocesses text |  
```python
from utils.preprocessing import clean_text  
print(clean_text("This Product!!! is Awesome..."))  # "this product is awesome"
``` |
| `metrics.py` | Calculates accuracy and recall |  
```python
from utils.metrics import compute_accuracy  
acc = compute_accuracy(pred_logits, true_labels)
``` |

---

### **6. `configs/`** – Settings for the project  

| File | Purpose | Example Use |
|------|---------|-------------|
| `default.yaml` | Controls dataset paths, training settings, and model parameters |  
```yaml
training:
  epochs: 10
  batch_size: 32


---

7. Root Files

File	Purpose

requirements.txt	Python packages needed
README.md	Documentation (this file)



---

🚀 How to Run

1️⃣ Install packages

pip install -r requirements.txt

2️⃣ Train the model

python training/train.py --config configs/default.yaml

3️⃣ Evaluate on test set

python training/evaluate.py --config configs/default.yaml


---

Do you want me to now add code snippets for every file in this README so that someone could copy-paste and run the example immediately without looking inside the file? That would make it a true “docs + usage” guide.







 & L:/college_project/FGMFN_S-A-A/.venv/Scripts/Activate.ps1