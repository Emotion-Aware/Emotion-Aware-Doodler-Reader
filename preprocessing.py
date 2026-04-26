import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json

print("="*60)
print("EMOTION DOODLE DATASET PREPROCESSING")
print("="*60)

print("\n[STEP 1] Setting up configuration...")

DATASET_DIRS = ["archive/data", "archive/NewArts2"]
OUTPUT_DIR = "processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

for directory in [OUTPUT_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

EMOTIONS = ["Angry", "Fear", "Happy", "Sad"]
IMG_SIZE = 128
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

print(f"Target emotions: {EMOTIONS}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")

print("\n[STEP 2] Loading images...")

X = []
y = []
emotion_counts = {emotion: 0 for emotion in EMOTIONS}
skipped = 0

for DATASET_DIR in DATASET_DIRS:
    if not os.path.exists(DATASET_DIR):
        print(f"Warning: '{DATASET_DIR}' not found, skipping...")
        continue

    print(f"\nLoading from: {DATASET_DIR}")

    for emotion in EMOTIONS:
        emotion_path = os.path.join(DATASET_DIR, emotion)

        if not os.path.exists(emotion_path):
            print(f"Skipping missing folder: {emotion}")
            continue

        files = [f for f in os.listdir(emotion_path)
                 if f.lower().endswith(VALID_EXTENSIONS)]

        loaded_count = 0

        for file in files:
            try:
                filepath = os.path.join(emotion_path, file)

                # Read image
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    skipped += 1
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)

                img = img.astype(np.float32) / 255.0

                X.append(img)
                y.append(emotion)
                emotion_counts[emotion] += 1
                loaded_count += 1

            except Exception:
                skipped += 1

        print(f"{emotion}: {loaded_count} images")

print(f"\nSkipped files: {skipped}")

if len(X) == 0:
    print("Error: No images loaded")
    exit(1)

X = np.array(X)

print(f"\nDataset shape: {X.shape}")

print("\n[STEP 3] Encoding labels...")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print(f"Classes: {encoder.classes_}")

print("\n[STEP 4] Train/Test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Train: {len(X_train)}")
print(f"Test: {len(X_test)}")

print("\n[STEP 5] Reshaping...")

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

print("\n[STEP 6] Saving...")

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

metadata = {
    "total_images": len(X),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "image_size": IMG_SIZE,
    "emotions": list(encoder.classes_),
    "emotion_distribution": emotion_counts,
    "skipped_files": skipped,
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nPreprocessing complete")