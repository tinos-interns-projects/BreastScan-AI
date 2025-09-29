# BreastScan AI

BreastScan AI is a deep learning–based system for the automated classification of *breast ultrasound images* into three categories: *Benign, Malignant, and Normal*.
The project integrates a *VGG16-based Convolutional Neural Network (CNN)* with a *Flask web application*, allowing users to upload grayscale ultrasound images, view prediction results with confidence levels, and download structured PDF reports.

*Disclaimer:* This project is not medically certified and is intended for research and educational purposes only.

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [System Architecture](#system-architecture)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Model Training](#model-training)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Discussion and Insights](#discussion-and-insights)
* [Future Scope](#future-scope)
* [Contributors](#contributors)

---

## Overview

Breast cancer remains a major global health concern, with millions of new cases each year. While ultrasound imaging is a widely used diagnostic method, manual interpretation can be time-intensive, error-prone, and inconsistent.

BreastScan AI addresses these challenges by providing an AI-powered second-opinion tool. Through transfer learning and fine-tuning of the VGG16 architecture, the system achieves expert-level accuracy and offers a user-friendly platform for real-time analysis of breast ultrasound images.

---

## Features

* Convolutional Neural Network (CNN) model based on VGG16
* Web-based interface with Flask, HTML, TailwindCSS, and JavaScript
* Intuitive confidence visualization for predictions
* Automated PDF report generation with results and metrics
* Data augmentation pipeline for enhanced generalization
* Local deployment for testing and demonstration

---

## System Architecture

1. *Frontend*

   * Built using HTML, TailwindCSS, and JavaScript
   * Key pages: index.html (upload), about.html (project details), result.html (predictions and reports)

2. *Backend*

   * Implemented using Flask (Python)
   * Handles file upload, preprocessing, model inference, and report generation

3. *Model*

   * Transfer learning with VGG16 (ImageNet weights)
   * Architecture: Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(3, Softmax)
   * Outputs: Benign, Malignant, Normal

---

## Dataset and Preprocessing

* *Dataset*: Grayscale ultrasound images categorized into three classes (Benign, Malignant, Normal)
* *Preprocessing steps*:

  * Resize to 224×224
  * Convert grayscale to RGB (3 channels)
  * Normalize pixel values to range [0,1]
* *Data Augmentation*: Horizontal/vertical flips, rotations, brightness/contrast adjustment, cropping, zooming

---

## Model Training

* Framework: TensorFlow & Keras
* Training strategy:

  * Phase 1: Train classifier head with frozen convolutional base
  * Phase 2: Fine-tune last convolutional blocks of VGG16
* Loss function: Categorical Crossentropy
* Optimizer: Adam with learning rate scheduling
* Regularization: Dropout and augmentation
* Callbacks: Early stopping, model checkpointing

---

## Installation

Clone the repository and install dependencies:

bash
git clone https://github.com/yourusername/BreastScan-AI.git
cd BreastScan-AI
pip install -r requirements.txt


---

## Usage

Run the Flask application:

bash
python app.py


Access the application in a browser:


http://127.0.0.1:5000/


Upload an ultrasound image, view the prediction with confidence metrics, and download the generated PDF report.

---

## Results

* *Accuracy*: 98.69%
* *Precision*: ~0.99
* *Recall*: ~0.99
* *F1-Score*: ~0.99

Confusion matrix analysis confirmed reliable classification across all categories, with high accuracy in differentiating between benign and malignant cases and robust detection of normal tissues.

---

## Discussion and Insights

### Strengths

* High accuracy and balanced performance metrics
* Effective transfer learning and fine-tuning strategy
* User-friendly interface with clear visualization
* Clinically relevant performance in ultrasound classification

### Challenges

* High computational requirements due to VGG16
* Large model size (56 MB)
* Dependence on high-quality labeled medical datasets

---

## Future Scope

* Explore alternative architectures such as EfficientNet and ResNet
* Integrate patient metadata for comprehensive diagnostic support
* Optimize for real-time deployment in clinical workflows
* Implement explainability methods (e.g., Grad-CAM)
* Conduct prospective clinical validation studies
* Develop cloud-based deployment with API integration

---

## Contributors

* *Rishil P*
* *Fasil Rahman TK*
* *Mentor:* Fayiz

---
