# Emotion-Aware Doodle Classifier

A deep learning system that classifies emotions (**Angry, Fear, Happy, Sad**) from hand-drawn doodles using a pretrained ResNet18 model.

---

## Overview

This project explores whether emotional states can be inferred from visual drawing patterns.  
People, especially children, often express emotions through drawings rather than words. This system analyzes such doodles to predict the underlying emotional state.

---

## Problem Statement

Traditional emotion recognition systems rely on facial expressions, speech, or text.  
However, these methods may not work effectively for individuals who struggle with verbal or facial expression.

This project provides an alternative approach by analyzing **hand-drawn doodles**.

---

## Features

- Image preprocessing using CLAHE
- Data augmentation (rotation, flipping, affine transformations)
- Transfer learning using ResNet18 (pretrained on ImageNet)
- Class imbalance handling using weighted loss
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

---

## Dataset

Combined dataset from:
- archive/data
- NewArts2

Classes:
- Angry
- Fear
- Happy
- Sad

Note: Dataset is not included in this repository due to size constraints.

---

## Model

- ResNet18 (pretrained)
- Final layers fine-tuned
- Dropout used for regularization

---

## Training Details

- Optimizer: Adam
- Learning Rate: 0.0001
- Scheduler: Cosine Annealing
- Loss Function: Weighted CrossEntropyLoss
- Batch Size: 16
- Early Stopping applied

---

## Results

- Accuracy: 68.47%
- Best class: Happy (F1-score: 0.88)
- Confusion mainly between Fear and Sad

---

## Project Pipeline

1. Image preprocessing
2. Data augmentation
3. Feature extraction using ResNet18
4. Model training
5. Evaluation on test data

---

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Preprocess data
python preprocessing.py

### 3. Train model
python train.py

### 4. Evaluate model
python evaluate.py

---

## Future Improvements

- Add stroke-based features (speed, pressure, pauses)
- Build real-time drawing interface
- Improve dataset size and diversity
- Extend to more complex emotional states

---

## Disclaimer

This project is for research and educational purposes only.  
It is not intended for medical or psychological diagnosis.

---

## License

MIT License
