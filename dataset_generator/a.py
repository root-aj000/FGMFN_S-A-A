
from paddlex import create_pipeline

ocr = create_pipeline(pipeline="ocr")
img_path = "data/raw/images.png"
result = list(ocr.predict(img_path))  # turn generator into list

print("\n===== RAW OCR OUTPUT =====")
print(result)
print("==========================")
