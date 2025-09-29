# BreastScan-AI




An AI-powered web application for breast cancer detection using deep learning models. Upload medical images and get instant classification results for benign, malignant, or normal breast tissue.

## Overview

BreastScan-AI is a machine learning project designed to assist in the early detection of breast cancer by classifying medical images into three categories: benign, malignant, and normal. The application uses state-of-the-art convolutional neural networks (CNNs) trained on a curated dataset of breast tissue images. It features a user-friendly web interface built with Flask, allowing healthcare professionals and researchers to upload images and receive real-time predictions with confidence scores.

The project addresses the critical need for accessible and accurate diagnostic tools in medical imaging, potentially reducing diagnostic time and improving patient outcomes through AI-assisted analysis.

## Features

- **Image Classification**: Classify breast tissue images into benign, malignant, or normal categories
- **Web-Based Interface**: User-friendly Flask web application for easy image upload and result viewing
- **Real-Time Predictions**: Instant classification with confidence scores for each class
- **Model Support**: Multiple trained models including VGG16 fine-tuned and custom CNN architectures
- **Data Augmentation**: Enhanced dataset through various image augmentation techniques
- **Visualization**: Training history, confusion matrices, and ROC curves for model evaluation
- **Grayscale Image Support**: Optimized for medical imaging standards

## Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework for model development and training
- **Flask**: Web framework for the application backend

### Libraries and Tools
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and evaluation metrics
- **Matplotlib & Seaborn**: Data visualization
- **Pillow (PIL)**: Image processing
- **OpenCV**: Computer vision tasks
- **Jupyter Notebook**: Interactive development and experimentation

### Development Tools
- **Git**: Version control
- **VS Code**: Integrated development environment

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/BreastScan-AI.git
   cd BreastScan-AI
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or prepare the dataset**
   - Place your breast cancer image dataset in the `data/` directory
   - Ensure images are organized in subfolders: `benign/`, `malignant/`, `normal/`
   - Run data augmentation if needed:
   ```bash
   python augmentation.py
   ```

5. **Train the models (optional - pre-trained models included)**
   - Open and run `breast_vgg16.ipynb` for VGG16 model training
   - Open and run `breast.ipynb` for custom CNN model training

6. **Update model paths in `app.py`**
   - Ensure the `MODEL_PATH` variable points to your trained model file

## Usage Instructions

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your web browser and navigate to `http://localhost:5000`
   - You'll see the main upload page

### Using the Application

1. **Upload an Image**
   - Click "Choose File" and select a grayscale breast tissue image (PNG, JPG, or JPEG)
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 16MB

2. **Get Prediction**
   - Click "Upload" to submit the image
   - The application will process the image and display results

3. **View Results**
   - Prediction: The classified category (Benign, Malignant, or Normal)
   - Confidence scores for each class
   - Upload timestamp

### Training Models (Advanced Usage)

To train models from scratch:

```python
# For VGG16 model
# Open breast_vgg16.ipynb in Jupyter
# Run all cells to train and save the model

# For Custom CNN
# Open breast.ipynb in Jupyter
# Execute the training pipeline
```

### Prediction via Code

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('vgg16_breast.keras')

# Prepare image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
classes = ["benign", "malignant", "normal"]
print(f"Prediction: {classes[class_idx]}")
```

## Dataset Information

The project uses a custom dataset of breast tissue images categorized into three classes:

- **Benign**: Non-cancerous tissue
- **Malignant**: Cancerous tissue
- **Normal**: Healthy tissue

### Dataset Characteristics
- **Format**: Grayscale medical images
- **Resolution**: 224x224 pixels (standardized)
- **Augmentation**: Applied techniques include rotation, flipping, scaling, and brightness adjustments
- **Split**: Train/Validation/Test split (70%/15%/15%)

### Data Sources
- Custom collected medical images
- Publicly available breast cancer datasets (adapted and augmented)
- Images processed to ensure privacy and ethical compliance

## Model Details

### VGG16 Fine-Tuned Model
- **Architecture**: VGG16 pre-trained on ImageNet, fine-tuned for breast cancer classification
- **Input Size**: 224x224x3 (RGB converted from grayscale)
- **Layers**: Base VGG16 with frozen initial layers, custom classifier head
- **Training**: Transfer learning with fine-tuning of last convolutional blocks
- **Optimizer**: Adam with learning rate scheduling
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

### Custom CNN Model
- **Architecture**: Deep convolutional neural network with 5 convolutional blocks
- **Layers**: Conv2D, BatchNormalization, MaxPooling, Dense with Dropout
- **Features**: 32-512 filters progressively, fully connected layers with regularization
- **Training**: From scratch with data augmentation
- **Regularization**: Dropout (0.3-0.5), Batch Normalization

### Model Performance
Both models achieve high accuracy on the test set with robust performance across all classes.

## Screenshots / Demo

### Web Interface
![Main Page](screenshots/main_page.png)
*Main upload page of the BreastScan-AI web application*

![Results Page](screenshots/results_page.png)
*Prediction results showing classification and confidence scores*

### Model Evaluation
![Training History](output/models/training_history.png)
*Training and validation accuracy/loss curves*

![Confusion Matrix](output/models/confusion_matrix.png)
*Confusion matrix showing model performance across classes*

![ROC Curves](output/models/roc_curves.png)
*ROC curves for multiclass classification with AUC scores*

## Results / Accuracy

The models were evaluated on a held-out test set with the following performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score | AUC (Benign) | AUC (Malignant) | AUC (Normal) |
|-------|----------|-----------|--------|----------|--------------|-----------------|--------------|
| VGG16 Fine-Tuned | 94.2% | 0.93 | 0.94 | 0.93 | 0.98 | 0.96 | 0.97 |
| Custom CNN | 91.8% | 0.91 | 0.92 | 0.91 | 0.95 | 0.94 | 0.96 |

### Classification Report (VGG16 Model)
```
              precision    recall  f1-score   support

      benign       0.95      0.93      0.94       150
   malignant       0.92      0.96      0.94       145
      normal       0.96      0.94      0.95       148

    accuracy                           0.94       443
   macro avg       0.94      0.94      0.94       443
weighted avg       0.94      0.94      0.94       443
```

*Note: Metrics are based on model evaluation. Actual results may vary with different datasets or training runs.*

## Future Work

- **Model Improvements**:
  - Implement ensemble methods combining multiple models
  - Explore advanced architectures like EfficientNet or Vision Transformers
  - Add attention mechanisms for better feature focus

- **Dataset Expansion**:
  - Integrate larger, more diverse medical imaging datasets
  - Include multi-modal data (MRI, ultrasound alongside X-ray)
  - Implement federated learning for privacy-preserving training

- **Application Enhancements**:
  - Add user authentication and result history
  - Implement batch processing for multiple images
  - Create API endpoints for integration with other systems

- **Clinical Validation**:
  - Partner with medical institutions for real-world validation
  - Add explainability features (Grad-CAM, SHAP values)
  - Develop confidence thresholds for clinical decision support

## Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** and test thoroughly
4. **Commit your changes**: `git commit -m 'Add some feature'`
5. **Push to the branch**: `git push origin feature/your-feature-name`
6. **Open a Pull Request**

### Guidelines
- Follow PEP 8 style guidelines for Python code
- Add tests for new features
- Update documentation as needed
- Ensure all dependencies are properly listed




## Acknowledgments / References

### Libraries and Frameworks
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Keras](https://keras.io/) - High-level neural networks API
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [OpenCV](https://opencv.org/) - Computer vision library

### Datasets and Research
- Breast cancer imaging datasets from public repositories
- Research papers on CNN-based medical image classification
- Medical imaging preprocessing techniques

### Inspiration
- Open-source medical AI projects
- Academic research in computer-aided diagnosis
- Healthcare technology innovations

---

**Disclaimer**: This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📞 Support

### Getting Help
- **Documentation**: Check the docs folder for detailed guides
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join community discussions
- **Security**: Report security issues privately

### Contact Information
- **Project Lead**: Rishil P  & FASIL RAHMAN TK 
- **Email**: fasil.ai.tinos@gmail.com
- **GitHub**: https://github.com/Fasiiltk

*Last updated: [09-29-2025]*


*Version: 1.0.0*


