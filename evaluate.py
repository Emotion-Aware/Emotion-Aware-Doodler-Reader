import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import pickle
import json

print("="*70)
print("EMOTION DOODLE CLASSIFIER - EVALUATION")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

PROCESSED_DIR = "processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# ============================
# LOAD DATA
# ============================
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print(f"\nTest shape: {X_test.shape}")
print(f"Classes: {encoder.classes_}")

# Convert to tensor
X_test_tensor = torch.from_numpy(X_test).float()
X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)
X_test_tensor = X_test_tensor.repeat(1, 3, 1, 1)

# ============================
# LOAD MODEL (ResNet18 SAME AS TRAINING)
# ============================
num_classes = len(encoder.classes_)

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

model = model.to(device)

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
model.eval()

print("\nModel loaded successfully")

# ============================
# PREDICTIONS
# ============================
all_preds = []
all_probs = []

with torch.no_grad():
    for i in range(0, len(X_test_tensor), 32):
        batch = X_test_tensor[i:i+32].to(device)

        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

print(f"\nPredictions done for {len(y_pred)} samples")

# ============================
# METRICS
# ============================
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
report = classification_report(y_test, y_pred, target_names=encoder.classes_)
print(report)

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()

# ============================
# CONFIDENCE ANALYSIS
# ============================
max_probs = np.max(y_prob, axis=1)

print("\nConfidence Analysis:")
print(f"Average confidence: {np.mean(max_probs)*100:.2f}%")
print(f"Min confidence: {np.min(max_probs)*100:.2f}%")
print(f"Max confidence: {np.max(max_probs)*100:.2f}%")

# ============================
# SAVE RESULTS
# ============================
results = {
    "accuracy": float(accuracy),
    "confusion_matrix": cm.tolist(),
}

with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nEvaluation complete")
print("="*70)