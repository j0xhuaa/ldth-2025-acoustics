import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# envs
MODEL_PATH = './models/model_logmel_cnn_v1.1.h5'

# pull in the trained model
model = load_model(MODEL_PATH)

# not strictly necessary as model is not getting retrained but suppresses warning
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# load the test data from disk for evaluation
X_test = np.load("./test_data/X_test_logmel_v1.1.npy")
y_test = np.load("./test_data/y_test_logmel_v1.1.npy")

# with open("labels_logmel.json", "r") as f:
#     labels = json.load(f)

# evaluate the model on the remaining test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# compute probabilities for a classification report

# predict class probabilities
y_pred_probs = model.predict(X_test)

# convert one-hot to class indices
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# use classification_report function from sklearn to compute precision, recall, and f1 score
report = classification_report(y_true, y_pred)
print("The classification report:")
print(report)
