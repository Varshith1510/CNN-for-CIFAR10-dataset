# CNN for CIFAR-10 Image Classification

This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. This model aims to achieve a high classification accuracy by applying CNN techniques.

## Project Overview

The goal of this project is to train a CNN model to classify images from the CIFAR-10 dataset into one of 10 categories. The model architecture consists of several convolutional layers followed by pooling layers, and a final fully connected layer for classification. The performance of the model is evaluated using accuracy and loss metrics.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 pixel images across 10 categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture

The CNN model used in this project consists of:
- **Convolutional layers**: Used for feature extraction from images.
- **Activation functions**: ReLU activation function is applied after each convolutional layer.
- **Pooling layers**: MaxPooling is used to reduce the spatial dimensions of the feature maps.
- **Fully connected layers**: After flattening the pooled feature maps, fully connected layers are used to classify the images.

## Evaluation

The model is evaluated based on:
- **Accuracy**: The percentage of correct classifications.
- **Loss**: The model’s error during training, which is minimized using backpropagation and gradient descent.

## Kaggle Notebook

You can find the complete code implementation and analysis in my Kaggle notebook:  
[Kaggle Notebook Link](https://www.kaggle.com/code/varshithpsingh/cnn-for-cifar-10)
