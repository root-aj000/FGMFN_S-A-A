# preprocessing/image_preprocessing.py
import cv2
import numpy as np
import torch
from augmentation import augment_image

# Config
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def resize_image(image, size=IMAGE_SIZE):
    """Resize image to fixed size."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """Normalize image to [0,1] and apply mean/std for pretrained CNNs."""
    image = image.astype(np.float32) / 255.0
    # Convert HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    # Normalize
    image = (image - np.array(NORMALIZE_MEAN)[:, None, None]) / np.array(NORMALIZE_STD)[:, None, None]
    return torch.tensor(image, dtype=torch.float)


