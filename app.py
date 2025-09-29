import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import io

# ===== CONFIG =====
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ===== FLASK APP =====
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Needed for flash messages

# Load your trained model
MODEL_PATH = r"C:\Users\User\Documents\tinos\Breast\vgg16_breast.keras"
model = load_model(MODEL_PATH)

# Image size (same as used in training)
IMG_SIZE = (224, 224)

# Class labels
CLASS_NAMES = ["benign", "malignant", "normal"]

# ===== Helper Functions =====
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(filepath):
    # Load image in grayscale
    img = image.load_img(filepath, target_size=IMG_SIZE, color_mode="grayscale")
    
    # Convert to RGB by replicating the single channel across all three channels
    img_array = image.img_to_array(img)
    
    # If image is grayscale (1 channel), convert to RGB (3 channels)
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def is_grayscale(img_path):
    """Check if an image is grayscale"""
    try:
        img = Image.open(img_path)
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image data
        pixels = list(img.getdata())
        
        # Check if all pixels have equal R, G, B values
        for pixel in pixels:
            r, g, b = pixel
            if r != g or r != b or g != b:
                return False
        return True
    except Exception as e:
        print(f"Error checking if image is grayscale: {e}")
        return False

# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded", "error")
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file", "error")
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            
            # Check if the image is grayscale
            if not is_grayscale(filepath):
                os.remove(filepath)  # Remove the uploaded file
                flash("Color images are not supported. Please upload a grayscale image.", "error")
                return redirect(request.url)
            
            return redirect(url_for("predict", filename=filename))
    
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict/<filename>")
def predict(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(filepath):
        flash(f"File {filename} not found", "error")
        return redirect(url_for("upload_file"))

    img_array = prepare_image(filepath)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))

    benign_confidence = round(float(preds[0][0]) * 100, 2)
    malignant_confidence = round(float(preds[0][1]) * 100, 2)
    normal_confidence = round(float(preds[0][2]) * 100, 2)

    upload_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    result = {
        "filename": filename,
        "prediction": CLASS_NAMES[class_idx],
        "confidence": round(confidence * 100, 2)
    }

    return render_template("result.html",
                           result=result,
                           benign_confidence=benign_confidence,
                           malignant_confidence=malignant_confidence,
                           normal_confidence=normal_confidence,
                           upload_time=upload_time)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)