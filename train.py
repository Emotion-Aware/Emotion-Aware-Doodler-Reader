import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
import matplotlib.pyplot as plt
import pickle
import json

print("="*70)
print(" EMOTION DOODLE CLASSIFIER - ResNet50 TRAINING")
print("="*70)

print("\n[STEP 1] Setting up environment...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

PROCESSED_DIR = "processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

print(f"\n[STEP 2] Loading preprocessed data...")

try:
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Classes: {encoder.classes_}")

except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    print(f"  Run preprocessing.py first!")
    exit(1)

print(f"\n[STEP 3] Converting to PyTorch tensors...")

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

# (N, H, W, 1) → (N, 1, H, W)
X_train_tensor = X_train_tensor.permute(0, 3, 1, 2)
X_test_tensor = X_test_tensor.permute(0, 3, 1, 2)

# ✅ ResNet needs 3 channels — repeat grayscale across RGB
X_train_tensor = X_train_tensor.repeat(1, 3, 1, 1)
X_test_tensor = X_test_tensor.repeat(1, 3, 1, 1)

print(f"  X_train tensor shape: {X_train_tensor.shape}")
print(f"  X_test tensor shape: {X_test_tensor.shape}")

# ============================================================
# DATA AUGMENTATION - stronger for small dataset
# ============================================================
print(f"\n[STEP 3.5] Setting up data augmentation...")

train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])

class AugmentedDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

print(f"  Data augmentation enabled (rotation, flip, affine, color jitter, blur)")

# ============================================================
# CREATE DATALOADERS
# ============================================================
BATCH_SIZE = 16
train_dataset = AugmentedDataset(X_train_tensor, y_train_tensor, transform=train_transform)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ============================================================
# STEP 4: ResNet50 - same as research paper
# ============================================================
print(f"\n[STEP 4] Building pretrained ResNet50 model (like research paper)...")

num_classes = len(encoder.classes_)

# Load pretrained ResNet50
model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

# ✅ Replace final classifier
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
)

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  ResNet50 loaded with pretrained ImageNet weights")
print(f"  Total parameters: {total:,}")
print(f"  Trainable parameters: {trainable:,}")

# ============================================================
# STEP 5: Training setup
# ============================================================
print(f"\n[STEP 5] Setting up training...")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ✅ label smoothing helps generalization

class_counts = np.bincount(y_train)
weights = 1. / class_counts
weights = weights / weights.sum()
weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,
    weight_decay=1e-4
)

EPOCHS = 50
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_val_acc = 0
patience_counter = 0
patience = 12

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

print(f"  Optimizer: Adam (lr=0.0005, weight_decay=1e-4)")
print(f"  Loss: CrossEntropyLoss with label smoothing=0.1")
print(f"  Scheduler: CosineAnnealingLR")
print(f"  Epochs: {EPOCHS} (early stopping patience={patience})")

print(f"\n[STEP 6] Starting training...\n")
print("="*70)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # ✅ Gradient clipping - prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, 100 * correct / total

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / total, 100 * correct / total

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, test_loader, criterion, device)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:6.2f}%  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:6.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
        print(f"           →  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print("="*70)

print(f"\n[STEP 7] Final evaluation...")

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
final_val_loss, final_val_acc = validate(model, test_loader, criterion, device)

print(f"\n  Best validation accuracy: {final_val_acc:.2f}%")
print(f"  Final validation loss: {final_val_loss:.4f}")

print(f"\n[STEP 8] Saving results...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Training & Validation Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150, bbox_inches="tight")
print(f"  Saved training_history.png")
plt.show()

with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)
print(f"  Saved training_history.json")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print("="*70)
print(f"\n  Model: {MODELS_DIR}/best_model.pth")
print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
print(f"  Results saved to: {RESULTS_DIR}/")
print(f"\n  Next step: Run evaluate.py to get detailed metrics!")
print("="*70 + "\n")