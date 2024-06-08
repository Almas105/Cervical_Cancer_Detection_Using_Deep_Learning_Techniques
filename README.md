# CNN-Based Cervical Cancer Detection and Classification System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Introduction
The **CNN-Based Cervical Cancer Detection and Classification System** leverages Convolutional Neural Networks (CNNs) to provide a robust solution for detecting and classifying cervical cancer from medical images. Due to the challenge of obtaining large datasets of medical images, this system employs transfer learning and fine-tuning techniques, utilizing pretrained CNN models to enhance performance with limited data.

The system investigates three CNN models: a shallow CNN architecture and two deep architectures (VGG-16 and CaffeNet). These models are fine-tuned on a targeted medical image database and integrated with classifiers for accurate detection and classification.

## Features
- **Shallow CNN Model**: Consists of two convolutional layers and two max-pooling layers, followed by two fully connected layers and a softmax output layer.
- **Deep CNN Models**: Utilizes VGG-16 and CaffeNet architectures, pretrained on large image datasets and fine-tuned for cervical cancer detection.
- **Transfer Learning and Fine-Tuning**: Applies transfer learning to adapt pretrained models to the specific medical image database.
- **ELM-Based Classifier**: Uses Extreme Learning Machines (ELMs) for fast learning and classification.
- **AE-Based Classifier**: Employs Autoencoders (AEs) for noise removal and feature extraction.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Almas105/Cervical_Cancer_Detection_Using_Deep_Learning_Techniques.git

    ```
2. **Install dependencies**:
   

### Dependencies
Ensure you have the following libraries installed:

- `tensorflow`
- `numpy`
- `scikit-learn`

You can install them using:
```bash
pip install tensorflow numpy scikit-learn
 ```
## Usage
To use the cervical cancer detection system, follow these steps:
- **Prepare your data:**:
   - The dataset used here is Herlev Dataset
   - Available in https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset
   - Organize your dataset into train and test directories with subdirectories for each class label.
## Configuration
- **Model Selection**: Choose between shallow CNN, VGG-16, and CaffeNet models.
- **Classifier Choice**: Select between ELM-based and AE-based classifiers.
- **Hyperparameters:**:Adjust learning rate, batch size, and epoch size

## Note
Please be aware that the efficiency of this system may not be optimal. This project has been implemented according to the base paper, but the results may not match the outcomes reported in the original paper.

