from fastapi import FastAPI
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image
from predict import predict_ad_sentiment  # your previously created predict_ad_sentiment function
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class AdInput(BaseModel):
    image_url: str

# API endpoint
@app.post("/predict")
def get_sentiment(ad: AdInput):
    try:
        response = requests.get(ad.image_url)
        img = Image.open(BytesIO(response.content))
        img.save("temp.jpg")  # temp save
        sentiment, confidence, text = predict_ad_sentiment("temp.jpg")
        return {"sentiment": sentiment, "confidence": confidence, "extracted_text": text}
    except Exception as e:
        return {"error": str(e)}