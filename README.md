# CNN Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) for image classification.  
The model is trained and evaluated on multiple datasets including MNIST, ORL, and CIFAR.  
The goal is to compare performance across datasets and study how model changes affect accuracy.

---

## Project Structure

- `main.py` → Entry point of the project
- `dataset.py` → Handles dataset loading and preprocessing
- `method.py` → Defines the CNN model architecture
- `setting.py` → Training loop (training, testing, evaluation)
- `evaluate.py` → Evaluation metrics (accuracy, etc.)
- `result.py` → Saves training results and plots
- `script_data_loader.py` → Loads datasets and initializes pipeline

---

## Datasets Used

- **MNIST**: Handwritten digit dataset (10 classes)
- **ORL**: Face dataset (40 classes)
- **CIFAR**: Color image dataset (10 classes)

Each dataset is split into:
- Training set
- Testing set

---

## Model Architecture

The CNN model consists of:

- Convolution Layer → Feature extraction
- ReLU Activation → Non-linearity
- Max Pooling → Downsampling
- Convolution Layer → Deeper feature extraction
- ReLU Activation
- Max Pooling
- Fully Connected Layer → Classification output

---

## Training Setup

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Learning Rate: 0.001
- Epochs: 20

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy (main metric)
- Precision
- Recall
- F1 Score

---

## Results

The model achieves good performance across datasets:

- MNIST: High accuracy with fast convergence
- ORL: Moderate accuracy due to face complexity
- CIFAR: More challenging, lower accuracy compared to MNIST


## How to Run

```bash
python3 main.py
