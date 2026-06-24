# Rice Disease Classifier using CNN

A Deep Learning based Rice Leaf Disease Classification System built using TensorFlow, Keras, and Streamlit. This project identifies common rice leaf diseases from uploaded images and predicts the disease class using a Convolutional Neural Network (CNN).

---

## Project Overview

Rice crops are highly vulnerable to various leaf diseases that can significantly reduce crop yield and quality. Early detection of these diseases helps farmers take preventive actions and minimize losses.

This project uses a Custom Convolutional Neural Network (CNN) trained on rice leaf images to automatically classify diseased leaves into one of the following categories:

- Bacterial Blight
- Blast
- Brown Spot
- Tungro

The trained model is deployed through a Streamlit web application for real-time disease prediction.

---

## Features

- Rice leaf disease detection using Deep Learning
- User-friendly Streamlit interface
- Upload leaf images for prediction
- Real-time classification results
- Data augmentation for better generalization
- Automatic model checkpoint saving
- Class index mapping using JSON
- Lightweight deployment-ready model

---

## Supported Diseases

| Disease | Description |
|----------|-------------|
| Bacterial Blight | Bacterial infection causing leaf wilting and yellowing |
| Blast | Fungal disease producing diamond-shaped lesions |
| Brown Spot | Disease causing brown circular spots on leaves |
| Tungro | Viral disease causing stunted growth and yellow-orange discoloration |

---

## Project Structure

```text
Rice_disease_classifier/
│
├── app/
│   ├── trained_model/
│   │   ├── best_model.keras
│   │   └── plant_disease_prediction_model.h5
│   │
│   ├── class_indices.json
│   ├── config.toml
│   ├── main.py
│   └── run_app.bat
│
├── Dataset/
│   ├── Bacterialblight/
│   ├── Blast/
│   ├── Brownspot/
│   └── Tungro/
│
├── model_training_notebook/
│   └── Rice_Disease_Prediction_CNN_Image_Classifier.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Architecture

The model is a Custom CNN built from scratch using TensorFlow/Keras.

### Architecture

```text
Input Layer (224x224x3)

↓ Rescaling (1/255)

↓ Data Augmentation
    - Random Flip
    - Random Rotation
    - Random Zoom
    - Random Contrast

↓ Conv2D (32) + BatchNormalization
↓ MaxPooling2D

↓ Conv2D (64) + BatchNormalization
↓ MaxPooling2D
↓ Dropout (0.2)

↓ Conv2D (128) + BatchNormalization
↓ MaxPooling2D
↓ Dropout (0.25)

↓ Conv2D (128) + BatchNormalization

↓ GlobalAveragePooling2D

↓ Dense (128, ReLU)

↓ Dropout (0.4)

↓ Dense (4, Softmax)

Output Layer
```

---

## Technologies Used

### Programming Language

- Python 3.x

### Libraries & Frameworks

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Streamlit
- JSON
- OS

### Deep Learning Techniques

- Convolutional Neural Networks (CNN)
- Data Augmentation
- Batch Normalization
- Dropout Regularization
- Transfer Learning Concepts
- Early Stopping
- Learning Rate Scheduling

---

## Data Preprocessing

The dataset is automatically loaded using:

```python
tf.keras.utils.image_dataset_from_directory()
```

### Preprocessing Steps

- Image resizing to 224×224
- Pixel normalization using Rescaling(1./255)
- Data augmentation
- Dataset shuffling
- Prefetching for optimized training

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Categorical Crossentropy |
| Metrics | Accuracy |
| Epochs | 15 |
| Validation Split | 20% |

---

## Callbacks Used

### EarlyStopping

Stops training when validation accuracy stops improving.

```python
EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)
```

### ReduceLROnPlateau

Reduces learning rate when validation loss stagnates.

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2
)
```

### ModelCheckpoint

Automatically saves the best-performing model.

```python
ModelCheckpoint(
    save_best_only=True
)
```

---

## Saved Files

### Trained Models

```text
best_model.keras
plant_disease_prediction_model.h5
```

### Class Mapping

```json
{
  "0": "Bacterialblight",
  "1": "Blast",
  "2": "Brownspot",
  "3": "Tungro"
}
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Rice_disease_classifier.git
cd Rice_disease_classifier
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

Navigate to app folder:

```bash
cd app
```

Run:

```bash
streamlit run main.py
```

Or on Windows:

```bash
run_app.bat
```

Application opens at:

```text
http://localhost:8501
```

---

## Workflow

```text
Rice Leaf Image
        ↓
Image Preprocessing
        ↓
CNN Feature Extraction
        ↓
Disease Classification
        ↓
Prediction Result
```

---

## Learning Outcomes

This project demonstrates:

- Deep Learning Fundamentals
- CNN Architecture Design
- Image Classification
- Data Augmentation
- Model Optimization
- TensorFlow/Keras Development
- Streamlit Deployment
- End-to-End Machine Learning Pipeline

---

## Future Improvements

- Increase dataset size
- Add more rice disease classes
- Mobile application deployment
- Real-time camera prediction
- Disease severity estimation
- Treatment recommendation system
- Cloud deployment

---

## Author

**Usman Talib**

BS Computer Science

Deep Learning & AI Projects

---

## If you found this project useful

Give this repository a star and support the project.
