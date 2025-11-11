# Brain Tumor Detection using Deep Learning (VGG16)
# Project Overview

The Brain Tumor Detection System is a deep learning-based project that classifies MRI brain images into four categories:

Glioma

Meningioma

Pituitary

No Tumor

This project leverages transfer learning using the VGG16 Convolutional Neural Network (CNN) architecture to detect tumors automatically from MRI scans. It assists doctors and radiologists in identifying tumors quickly and accurately, potentially improving early diagnosis and treatment planning.

#  About the Project

This project demonstrates how deep learning can be applied to the medical imaging domain. The complete workflow includes:

1. Data Preprocessing – Loading and resizing MRI images to 128×128 pixels

2. Data Augmentation – Applying random transformations to improve model generalization

3.Feature Extraction – Using VGG16 pre-trained on ImageNet

4.Model Training – Fine-tuning top layers for brain tumor classification

5. Performance Evaluation – Generating accuracy, loss plots, confusion matrix, and classification report

# Model Architecture

| Layer Type        | Description                                             |
| ----------------- | ------------------------------------------------------- |
| **Input Layer**   | 128×128×3 MRI images                                    |
| **Base Model**    | VGG16 (pre-trained on ImageNet, last 3 layers unfrozen) |
| **Flatten Layer** | Converts feature maps into 1D                           |
| **Dense Layer 1** | 128 neurons with ReLU activation                        |
| **Dropout**       | Dropout rate of 0.3                                     |
| **Dense Layer 2** | 64 neurons with ReLU activation                         |
| **Dropout**       | Dropout rate of 0.2                                     |
| **Output Layer**  | 4 neurons with Softmax activation (4 tumor classes)     |

Optimizer: Adam (learning_rate=0.0001)
Loss Function: Sparse Categorical Crossentropy
Metric: Sparse Categorical Accuracy

# 🩺 Dataset Description

Dataset: Brain Tumor Detection
Source: Kaggle 

| Folder      | Description                                   |
| ----------- | --------------------------------------------- |
| `Training/` | Contains training images for each tumor class |
| `Testing/`  | Contains testing images for each tumor class  |

# Libraries & Frameworks:

1. TensorFlow / Keras – Model Building (VGG16, Sequential, Layers)

2. NumPy – Numerical Computations

3. Matplotlib, Seaborn – Data Visualization

4. PIL (Pillow) – Image Loading & Augmentation

5. scikit-learn – Label Encoding, Shuffling, and Evaluation Metrics

# Features

1. Classifies MRI brain images into 4 tumor categories
2. Uses transfer learning with VGG16 for high accuracy
3. Includes data augmentation (flipping, brightness adjustment)
4. Visualizes training history, confusion matrix, and sample predictions
5. Easy to retrain with new MRI data

# Model Performance
Training Summary
<img width="950" height="225" alt="image" src="https://github.com/user-attachments/assets/2eb2c5c5-5740-447c-8aca-549bb2eebbe6" />

# Classification Report
<img width="595" height="266" alt="image" src="https://github.com/user-attachments/assets/8a83dbf0-cba3-449d-8b3a-79ebba5854b0" />

