# Emotion-Aware Doodle Classifier

A deep learning project that classifies emotions (Angry, Fear, Happy, Sad) from hand-drawn doodles using ResNet18.

## Features
- Image preprocessing with CLAHE
- Data augmentation
- Transfer learning (ResNet18)
- Evaluation with confusion matrix and metrics

## Dataset
Two datasets combined:
- archive/data
- NewArts2

## Model
- ResNet18 (pretrained on ImageNet)
- Fine-tuned final layers

## Results
- Accuracy: 68.47%
- Best class: Happy (F1: 0.88)
- Confusion mainly between Fear and Sad

## How to Run

### 1. Preprocess