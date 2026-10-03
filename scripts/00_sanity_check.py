"""
scripts/00_sanity_check.py

Phase 0 smoke test: confirms the recognition model builds, trains, and
predicts correctly. Uses the small HuggingFace ABRobOcular_Attacks subset
(10 subjects, already-attacked images) purely as a fast plumbing check --
this is NOT the real VISOB baseline, which needs the full dataset.
"""
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.recognition_model_phase0 import build_recognition_model, predict_labels

print("Loading dataset...")
ds = load_dataset("BharathK333/ABRobOcular_Attacks")["train"]

images = np.stack([np.array(img.convert("RGB").resize((256, 256))) for img in ds["image"]])
images = tf.keras.applications.xception.preprocess_input(images.astype("float32"))

le = LabelEncoder()
labels = le.fit_transform(ds["label"])
num_classes = len(le.classes_)
print(f"Loaded {len(images)} images, {num_classes} classes.")

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)
y_train_oh = to_categorical(y_train, num_classes)
y_test_oh = to_categorical(y_test, num_classes)

print("Building model...")
training_model, embedding_model = build_recognition_model(num_classes)
training_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss="categorical_crossentropy", metrics=["accuracy"])

print("Training briefly (smoke test only -- few epochs, tiny data)...")
training_model.fit([X_train, y_train_oh], y_train_oh, epochs=5, batch_size=8, verbose=1)

print("Predicting on held-out test images...")
preds = predict_labels(training_model, X_test, num_classes)
acc = (preds == y_test).mean()
print(f"Smoke-test accuracy: {acc:.2%}  (10-class tiny subset -- not a real result)")
print("Sample predictions:", preds[:10])
print("Sample true labels:", y_test[:10])