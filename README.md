# 🌿 Plant Disease Prediction CNN - Deep Learning Project

A deep learning-based web application that detects plant diseases from leaf images and provides treatment recommendations. Built with **TensorFlow**, **Keras**, and **Streamlit**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Model Architecture](#-model-architecture)
- [Dataset Information](#-dataset-information)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Your Own Model](#-training-your-own-model)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Disease Detection** | Identifies 23 different plant diseases across 5 crop types |
| 💊 **Treatment Suggestions** | Provides personalized treatment recommendations |
| 📊 **Confidence Score** | Shows prediction confidence percentage |
|  **Web Interface** | User-friendly Streamlit web application |
| 📱 **Multi-Format Support** | Accepts JPG, JPEG, and PNG images |
| ⚡ **Fast Prediction** | Real-time inference (< 1 second) |
| 💾 **Lightweight Model** | Only ~516 KB (easy to download and deploy) |

---
## 🎬 Demo
![Demo](assets/demo.gif)

## Model Architecture

This project uses a **Convolutional Neural Network (CNN)** with **GlobalAveragePooling2D** for efficient feature extraction and reduced model size.

### **Architecture Diagram**

```
                            Input (224×224×3)
                                    ↓
                            Conv2D (32 filters, 3×3) + ReLU
                                    ↓
                            MaxPooling2D (2×2)
                                    ↓
                            Dropout (0.25)
                                    ↓
                            Conv2D (64 filters, 3×3) + ReLU
                                    ↓
                            MaxPooling2D (2×2)
                                    ↓
                            Dropout (0.25)
                                    ↓
                            Conv2D (128 filters, 3×3) + ReLU
                                    ↓
                            MaxPooling2D (2×2)
                                    ↓
                            Dropout (0.25)
                                    ↓
                            GlobalAveragePooling2D ⭐
                                    ↓
                            Dense (256 units) + ReLU
                                    ↓
                            Dropout (0.5)
                                    ↓
                            Dense (23 units) + Softmax
                                    ↓
                            Output (23 Classes)
```

### **Model Specifications**

| Parameter | Value |
|-----------|-------|
| **Architecture** | CNN with GlobalAveragePooling2D |
| **Total Parameters** | 132,183 |
| **Model Size** | ~516 KB |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Activation** | ReLU (hidden), Softmax (output) |

### **Why GlobalAveragePooling2D?**

| Traditional Flatten | GlobalAveragePooling2D |
|--------------------|----------------------|
| ~180 MB model size | ~516 KB model size ✅ |
| More parameters | Fewer parameters ✅ |
| Prone to overfitting | Better generalization ✅ |

---

## 📊 Dataset Information

### **Dataset Statistics**

| Metric | Value |
|--------|-------|
| **Total Images** | 35,725 |
| **Training Images** | 28,589 (80%) |
| **Validation Images** | 7,136 (20%) |
| **Number of Classes** | 23 |
| **Image Size** | 224 × 224 pixels |
| **Format** | RGB (3 channels) |

### **Supported Plant Diseases**

| Plant | Diseases | Classes |
|-------|----------|---------|
| 🍎 **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy | 4 |
| 🌽 **Corn/Maize** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| 🫑 **Pepper Bell** | Bacterial Spot, Healthy | 2 |
| 🥔 **Potato** | Early Blight, Late Blight, Healthy | 3 |
| 🍅 **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |

### **Data Augmentation**

The following augmentation techniques were applied during training:

- ✅ Rotation (±40°)
- ✅ Width Shift (20%)
- ✅ Height Shift (20%)
- ✅ Shear (20%)
- ✅ Zoom (20%)
- ✅ Horizontal Flip
- ✅ Rescaling (1/255)

---

## 📁 Project Structure

```
plant-disease-prediction-cnn-deep-leanring-project/
├── app/
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5    # Pre-trained model (~516 KB)
│   ├── class_indices.json                        # Class mapping
│   ├── treatments.json                           # Treatment recommendations
│   ├── main.py                                   # Streamlit app
│   ├── config.toml                               # Streamlit config
│   
├── test_images/                                  # Sample images for testing
│   ├── Apple___Apple_scab.JPG
│   ├── Tomato_healthy1.JPG
│   └── ... (60+ test images)
|
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

### **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.12+ |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB | 5 GB+ |
| **CPU** | Any | Multi-core |
| **GPU** | Optional | NVIDIA GPU (for training) |

### **Python Dependencies**

All dependencies are listed in `requirements.txt`:

```txt
tensorflow>=2.12.0
streamlit>=1.28.0
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
scipy>=1.11.0
```

---

## 🚀 Installation

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/yasirfareeddev/Plant_disease_classifier.git
cd plant-disease-prediction-cnn-deep-leanring-project
```

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Verify Installation**

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import streamlit as st; print(st.__version__)"
```

---

## 💻 Usage

### **Option 1: Use Pre-trained Model (Recommended)** ⚡

**No training required!** The pre-trained model is already included.

```bash
# Navigate to app directory
cd app

# Run the Streamlit application through the workspace virtual environment
..\..\.venv\Scripts\python.exe -m streamlit run main.py
```

The app will open in your browser at: **`http://localhost:8501`**

On Windows, you can also run the bundled launcher from `app/`:

```bash
run_app.bat
```

### **Step-by-Step Usage**

1. **Upload an Image**: Click "Upload a leaf image..." and select a plant leaf photo
2. **Click Classify**: Press the "🔍 Classify Disease" button
3. **View Results**: See the prediction, confidence score, and treatment recommendations
4. **Test with Sample Images**: Use images from the `test_images/` folder

### **Option 2: Train Your Own Model** 🏋️

If you want to retrain with custom data:

```bash
# 1. Download the dataset (556 MB)
# Option A: Manual download
# Download from: https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection
# Extract to: project/Dataset/

# Option B: Using Kaggle API
# Place kaggle.json in app/ folder
# Run the Jupyter notebook

# 2. Open Jupyter Notebook
jupyter notebook model_training_notebook/Plant_Disease_Prediction_CNN_Image_Classifier.ipynb

# 3. Run all cells to train the model
# Training time: ~161 minutes on CPU, ~15-30 minutes on GPU

# 4. Model will be saved to: app/trained_model/plant_disease_prediction_model.h5
```

---

## 🎓 Training Your Own Model

### **Training Configuration**

| Parameter | Value |
|-----------|-------|
| **Epochs** | 10 |
| **Batch Size** | 32 |
| **Learning Rate** | Adam (default) |
| **Validation Split** | 20% |
| **Image Size** | 224 × 224 |
| **Training Time** | ~161 minutes (CPU) / ~15-30 minutes (GPU) |

### **Training Process**

1. Open the Jupyter notebook
2. Ensure dataset is in `Dataset/` folder
3. Run all cells sequentially
4. Model and class indices will be saved automatically

### **Expected Performance**

| Epoch | Training Accuracy | Validation Accuracy |
|-------|------------------|---------------------|
| 1 | 34.37% | 60.82% |
| 5 | 75.49% | 84.04% |
| 10 | 83.53% | **90.01%** ✅ |

---

## 📈 Results

### **Model Performance**

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | 90.01% |
| **Final Training Accuracy** | 83.53% |
| **Final Validation Loss** | 0.3214 |
| **Final Training Loss** | 0.5012 |
| **Model Size** | 516.34 KB |
| **Inference Time** | < 1 second |

### **Accuracy Graph**

```
Epoch 1:  ████████░░░░░░░░░░░░ 60.82%
Epoch 5:  ████████████████░░░░ 84.04%
Epoch 10: ████████████████████ 90.01%
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### **Areas for Improvement**

- [ ] Add more plant species
- [ ] Improve model accuracy (>95%)
- [ ] Add mobile app support
- [ ] Implement real-time camera detection
- [ ] Add multi-language support
- [ ] Create API endpoint for integration

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Built by:** Yasir Fareed

**Project Link:** https://github.com/yasirfareeddev/Plant_disease_classifier


---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| ⭐ **Model Accuracy** | 90.01% |
| 📦 **Model Size** | 516 KB |
| 🌿 **Supported Plants** | 5 (Apple, Corn, Pepper, Potato, Tomato) |
| 🦠 **Detectable Diseases** | 23 |
| ⏱️ **Training Time** | 161 minutes (CPU) |
| 🚀 **Inference Time** | < 1 second |
| 📸 **Test Images Included** | 60+ |

---

<div align="center">

**Made by Yasir Fareed using Streamlit & TensorFlow**

⭐ **Star this repository if you found it helpful!**

</div>