# Optimizing Image Classification Model: Exploring with CNNs, Transfer Learning, Hyperparameter Tuning, and K-Fold Cross-Validation

This project focuses on optimizing an image classification model using Convolutional Neural Networks (CNNs), transfer learning, hyperparameter tuning, and K-Fold Cross-Validation. The goal is to explore various techniques to improve model performance on the CIFAR-10 dataset.

## Disclaimer

This project is for exploration and testing purposes only and is not intended for production use.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)

## Introduction

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The project aims to build and optimize a CNN model to classify these images accurately.

## Dataset

The CIFAR-10 dataset is loaded and preprocessed using the following steps:
- Normalization of pixel values to be between 0 and 1.
- Conversion of labels to one-hot encoding.

## Methodology

### Data Augmentation

Enhanced data augmentation techniques are applied to improve model generalization. These include:
- Width and height shifts.
- Horizontal flips.
- Rotations.
- Zooming.
- Shearing.

### Model Architecture

A slightly more complex CNN model is built with the following layers:
- Convolutional layers with batch normalization.
- MaxPooling layers.
- Dense layers with dropout for regularization.

### Hyperparameter Tuning

Hyperparameters are tuned using Optuna, an automatic hyperparameter optimization framework. The following hyperparameters are optimized:
- Learning rate.
- Batch size.

### Transfer Learning

Transfer learning is applied using pre-trained models such as MobileNetV2 and VGG16. The base models are frozen, and custom top layers are added for classification.

### K-Fold Cross-Validation

K-Fold Cross-Validation is used to evaluate the model's performance more robustly. The dataset is split into 2 folds, and the model is trained and validated on each fold.

## Results

### Training and Validation Curves

The training and validation curves show the model's performance over epochs. The plots indicate the accuracy and loss trends.

### Hyperparameter Optimization

The best trial from Optuna optimization yielded the following parameters:
- Learning rate: 0.0008750803832793538
- Batch size: 32

### Transfer Learning Evaluation

Transfer learning models (MobileNetV2 and VGG16) were evaluated on a smaller subset of the dataset. The results are as follows:

- **MobileNetV2:**
  - Test Accuracy: 0.5699999928474426
  - Precision: 0.5869781144781144
  - Recall: 0.57
  - F1-Score: 0.5679252766473905
  - ROC-AUC: 0.8973518931972926

- **VGG16:**
  - Test Accuracy: 0.46000000834465027
  - Precision: 0.48210802742149483
  - Recall: 0.46
  - F1-Score: 0.45696542882981744
  - ROC-AUC: 0.8721693513561938

## Evaluation

Confusion matrices are generated to visualize the model's performance. The matrices show the true vs. predicted classes for both MobileNetV2 and VGG16.

## Conclusion

This project demonstrates the effectiveness of various techniques in optimizing image classification models. Transfer learning with MobileNetV2 and VGG16, combined with hyperparameter tuning and K-Fold Cross-Validation, provides insights into improving model performance on the CIFAR-10 dataset.