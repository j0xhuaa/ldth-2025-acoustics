import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import librosa

from data_loader import pad_or_truncate, load_data_to_waveform, waveform_to_logmel

#CHECKPOINT: Needs rethinking from scratch, currently there are issues with data matching.

"""
Tests the trained model on unseen data, to see how it performs in the wild:

1. Load the trained model from disk.
2. Load the unseen data and process it.
3. Run inference
4. Output performance metrics
"""


# config
MODEL_PATH = "./models/model_logmel_cnn_v1.1.h5"
TEST_DIR = "/Users/joshua/Devspace/projects/ldth-2025-acoustics/training/data/raw/test"
SAMPLE_RATE = 16000
N_MELS = 64
DURATION = 2.0

model = load_model(MODEL_PATH)

# === Load and preprocess unseen data ===
X_raw, y_labels, _ = load_data_to_waveform(TEST_DIR)

log_mels = [waveform_to_logmel(w, SAMPLE_RATE) for w in X_raw]
X_unseen = np.array(log_mels)[..., np.newaxis]  # shape: (samples, 128, time_steps, 1)

# Normalize (global)
X_unseen = (X_unseen - np.mean(X_unseen)) / np.std(X_unseen)

# # encode labels
# le = LabelEncoder()
# y_encoded = le.fit_transform(y_labels)
# y_true = to_categorical(y_encoded, num_classes=len(le.classes_))

# Use labels directly
y_encoded = np.array(y_labels)
y_true = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

# For reporting purposes
unique_classes = sorted(np.unique(y_encoded))
target_names = [str(c) for c in unique_classes]


# compile model so we can also tack on ability to measure loss and
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# predict
y_probs = model.predict(X_unseen)
y_pred = np.argmax(y_probs, axis=1)

from sklearn.utils.multiclass import unique_labels

# # Identify which labels are present in either true or predicted
# present_labels = unique_labels(y_encoded, y_pred)
# filtered_target_names = [le.classes_[i] for i in present_labels]

# print("le.classes_:", le.classes_)
# print("Type of elements in le.classes_:", type(le.classes_[0]))

# print("Sample y_labels:", y_labels[:5])
# print("Type of first y_label:", type(y_labels[0]))


print("\nClassification Report on Unseen Data")
print(classification_report(
    y_encoded,
    y_pred,
    target_names=target_names
))

# evaluate unseen data for accuracy and cross entropy loss
loss, accuracy = model.evaluate(X_unseen, y_true, verbose=0)
print(f"\nUnseen Test Accuracy: {accuracy:.4f}")
print(f"Unseen Test Loss: {loss:.4f}")

print("Unseen shape:", X_unseen.shape)