# dataset_generator\gen_script.py
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
IMAGE_OUTPUT_DIR = "data/images/"
LOG_DIR = "data/logs/"

# preprocessing\dataset.py
IMAGE_DIR = "data/processed/images/" 
TRAIN_CSV = "data/processed/train.csv" 

# models/fg_mfn.py
MODEL_CONFIG = "models/configs/model_config.json" 

# preprocessing\dataset_preprocessing.py
    # TRAIN_CSV = "data/processed/train.csv"
PROCESSED_IMAGE_DIR = "data/images/"
VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"

# training/train.py
    # TRAIN_CSV = "data/processed/train.csv"
    # VAL_CSV = "data/processed/val.csv"
#    MODEL_CONFIG = "models/configs/model_config.json"
SAVED_MODEL_DIR = "models/saved_models/"

# training/logger.py
    # LOG_DIR = "data/logs/"

# training/evaluate.py
    # TEST_CSV = "data/processed/test.csv"
    # MODEL_CONFIG = "models/configs/model_config.json"
    # LOG_DIR = "data/logs/"
SAVED_MODEL_PATH = "models/saved_models/model_best.pt"

# server\predict.py
    # SAVED_MODEL_PATH = "models/saved_models/model_final.pt"
    # MODEL_CONFIG = "models/configs/model_config.json"
IMAGE_UPLOAD_DIR = "data/images/tmp_uploads/"


# server/app.py
    # SAVED_MODEL_PATH = "models/saved_models/model_final.pt"
    # UPLOAD_FOLDER = IMAGE_UPLOAD_DIR
    # LOG_DIR = "data/logs/"

