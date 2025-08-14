from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from PIL import Image
import requests
from io import BytesIO
from predict.predict import predict_ad_sentiment

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    return path, filename

def save_url_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = url.split("/")[-1].split("?")[0] or "temp.jpg"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(path)
        return path, filename
    except:
        return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    if "results" not in session:
        session["results"] = []

    results = session["results"]

    if request.method == "POST":
        # Handle uploaded files
        files = request.files.getlist("images")
        for file in files:
            if file and allowed_file(file.filename):
                path, filename = save_uploaded_file(file)
                sentiment, confidence, text = predict_ad_sentiment(path)
                results.append({
                    "filename": filename,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "text": text
                })

        # Handle URLs
        urls = request.form.get("image_urls", "").splitlines()
        for url in urls:
            url = url.strip()
            if url:
                path, filename = save_url_image(url)
                if path:
                    sentiment, confidence, text = predict_ad_sentiment(path)
                    results.append({
                        "filename": filename,
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "text": text
                    })

        session["results"] = results  # save back to session
        return redirect(url_for("index"))

    return render_template("index.html", results=results)

@app.route("/clear", methods=["POST"])
def clear():
    session["results"] = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)