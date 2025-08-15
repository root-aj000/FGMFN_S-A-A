import os
import logging
from PIL import Image
import pytesseract
from utils.path import LOG_DIR

# ------------------ PATHS ------------------
LOG_DIR = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------ LOGGING ------------------
def log_error(message: str, path: str = os.path.join(LOG_DIR, "server_errors.log")):
    logging.basicConfig(
        filename=path,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.error(message)

# ------------------ OCR ------------------
def extract_text(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        log_error(f"OCR failed: {str(e)}")
        return ""
