# preprocessing/augmentation.py
import cv2
import numpy as np
import random

def augment_image(image):
    """Apply random augmentation: flip, rotate, crop."""
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random crop (up to 10% of image)
    crop_ratio = random.uniform(0, 0.1)
    if crop_ratio > 0:
        h_crop = int(h * crop_ratio)
        w_crop = int(w * crop_ratio)
        image = image[h_crop:h-h_crop, w_crop:w-w_crop]
        image = cv2.resize(image, (w, h))

    return image
