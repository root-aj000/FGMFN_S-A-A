

FGMFN: Fine-Grained Multiscale Cross-Modal Sentiment Analysis

This project predicts sentiment in advertisements by analyzing both images and text.



📂 Folder & File Overview

Below is a description of each file and folder, along with example usage.


---

1. data/ — Datasets

ytb_ads/ — YouTube Ads dataset

twitter2015/ — Twitter dataset (2015)

twitter2017/ — Twitter dataset (2017)


Example dataset format (train.txt):

images/ad1.jpg	This product is amazing!	2
images/ad2.jpg	Not worth the money.	0

First column: image path

Second column: ad text

Third column: sentiment label (0 = negative, 1 = neutral, 2 = positive)



---

2. models/ — Model Components

File	Purpose	Example Usage

fg_mfn.py	Core FGMFN model combining image & text features	python\nfrom models.fg_mfn import FGMFN\nmodel = FGMFN(embed_dim=256)\n
visual_module.py	Multi-scale image feature extractor	python\nfrom models.visual_module import VisualFeatureExtractor\nvisual_model = VisualFeatureExtractor()\n
text_module.py	Text encoder with visual guidance	python\nfrom models.text_module import TextEncoder\ntext_model = TextEncoder(model_name="bert-base-uncased")\n



---

3. losses/ — Custom Loss Functions

File	Purpose	Example Usage

matching_loss.py	Matches image & text features if they belong together	python\nfrom losses.matching_loss import MatchingLoss\nloss_fn = MatchingLoss(margin=0.2)\n
mutual_info_loss.py	Encourages information sharing between image & text	python\nfrom losses.mutual_info_loss import MutualInfoLoss\nloss_fn = MutualInfoLoss()\n



---

4. training/ — Training & Evaluation

File	Purpose	Example Usage

train.py	Trains the FGMFN model	bash\npython training/train.py --config configs/default.yaml\n
evaluate.py	Tests the trained model	bash\npython training/evaluate.py --config configs/default.yaml\n



---

5. utils/ — Helper Functions

File	Purpose	Example Usage

dataset.py	Loads image + text dataset into PyTorch	python\nfrom utils.dataset import AdvertisementDataset\ndataset = AdvertisementDataset(\"data/ytb_ads\", split=\"train\")\n
preprocessing.py	Cleans and preprocesses text	python\nfrom utils.preprocessing import clean_text\nprint(clean_text(\"This Product!!! is Awesome...\"))  # 'this product is awesome'\n
metrics.py	Calculates evaluation metrics	python\nfrom utils.metrics import compute_accuracy\nacc = compute_accuracy(pred_logits, true_labels)\n



---

6. configs/ — Project Configurations

File	Purpose	Example

default.yaml	Controls dataset paths, training settings, and model parameters	yaml\ntraining:\n  epochs: 10\n  batch_size: 32\nmodel:\n  embed_dim: 256\n



---

7. Root Files

requirements.txt — Python dependencies

README.md — Documentation (this file)



---

🚀 How to Run the Project

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Train the model

python training/train.py --config configs/default.yaml

3️⃣ Evaluate the model

python training/evaluate.py --config configs/default.yaml


---

