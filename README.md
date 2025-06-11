# Real-Time Shape Classification System

A machine learning project developed at UC San Diego as part of the ENLACE program for real-time geometric shape recognition using Support Vector Machine (SVM) classification and computer vision.

## Overview

This system trains an SVM classifier to recognize geometric shapes (circles, squares, triangles, stars) from 16x16 pixel images and provides real-time classification through webcam capture. The project demonstrates practical machine learning applications in computer vision with optimized integer weights for potential hardware implementation.

## Features

- **Multi-shape Support**: Classifies circles, squares, triangles, and stars
- **Real-time Classification**: Webcam integration for live shape detection
- **Optimized Implementation**: Integer weight quantization for hardware efficiency
- **High Accuracy**: Achieves 98%+ classification accuracy
- **Interactive Interface**: Google Colab compatible with webcam capture

## Requirements

### Python Libraries
```python
numpy
opencv-python (cv2)
scikit-learn
matplotlib
PIL (Pillow)
pathlib
```

### Google Colab Specific
- IPython.display
- google.colab (for drive mounting and file upload)
- base64 (for webcam functionality)

## Project Structure

```
project/
├── shapes/                 # Training data directory
│   ├── circle/            # Circle images (*.png)
│   ├── square/            # Square images (*.png)
│   ├── triangle/          # Triangle images (*.png)
│   └── star/              # Star images (*.png)
├── vec_train.npy          # Training feature vectors
├── cat_train.npy          # Training labels
├── vec_test.npy           # Test feature vectors
├── cat_test.npy           # Test labels
└── photo.jpg              # Captured webcam image
```

## Installation & Setup

### 1. Clone and Setup Environment
```bash
# In Google Colab or local Jupyter environment
pip install opencv-python scikit-learn matplotlib pillow numpy
```

### 2. Data Preparation
- Create a `shapes/` directory with subdirectories for each shape class
- Place PNG images (16x16 recommended) in respective shape folders
- Or upload pre-processed `.npy` files directly

### 3. Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

### 1. Data Loading and Preprocessing
The system automatically:
- Loads images from the `shapes/` directory
- Resizes images to 16x16 pixels
- Converts to grayscale
- Flattens to 256-dimensional feature vectors

### 2. Model Training
```python
# Select target shape
target_shape = 'square'  # Options: 'circle', 'star', 'square', 'triangle'

# Train SVM classifier
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(vec_train, cat_train_bin)
```

### 3. Real-time Classification
```python
# Capture image from webcam
filename = take_photo()

# Process and classify
result = classify_image(filename, target_shape)
```

## Model Architecture

### Feature Extraction
- **Input**: RGB images of any size
- **Preprocessing**: Resize to 16x16, convert to grayscale, binary thresholding
- **Features**: 256-dimensional flattened pixel vectors

### Classification
- **Algorithm**: Support Vector Machine (SVM) with linear kernel
- **Binary Classification**: One-vs-all approach for each shape
- **Weight Quantization**: Coefficients scaled and clipped to 4-bit integers (-8 to 7)

### Performance Optimization
- Integer weights for hardware implementation
- Reduced precision maintains 98%+ accuracy
- Efficient dot product computation

## Configuration Options

### Shape Selection
```python
target_shape = 'circle'    # Detect circles
target_shape = 'star'      # Detect stars  
target_shape = 'square'    # Detect squares
target_shape = 'triangle'  # Detect triangles
```

### SVM Parameters
```python
# Linear kernel (recommended)
clf = SVC(kernel='linear')

# Alternative kernels
clf = SVC(kernel='rbf')
clf = SVC(random_state=0, tol=1e-5)
```

### Image Processing
```python
# Binary threshold (adjustable)
thres = 100
img_bin[img_gray > thres] = 255
img_bin[img_gray < thres] = 0
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure `shapes/` directory exists with proper structure
2. **Webcam Access**: Grant camera permissions in browser
3. **Import Errors**: Install required packages via pip
4. **Low Accuracy**: Check image quality and threshold settings

### Data Requirements

- **Minimum Images**: 1000+ per shape class recommended
- **Image Format**: PNG files preferred
- **Image Quality**: Clear, well-lit shapes on contrasting backgrounds
- **Balanced Dataset**: Equal representation of all shape classes

## Educational Objectives

This project demonstrates:
- **Machine Learning Pipeline**: Data loading, preprocessing, training, evaluation
- **Computer Vision**: Image processing and feature extraction
- **Real-world Application**: Webcam integration and live classification
- **Optimization Techniques**: Model quantization for deployment
- **Performance Analysis**: Accuracy metrics and validation
