import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

MODEL_PATH = r"E:\Workspace\ai\fruit\fruit_model.h5"
DATASET_PATH = r"E:\Workspace\ai\fruit\dataset\fruits"

class_names = ["apple", "banana", "mango"]

X = []
y = []

for idx, class_name in enumerate(class_names):
    class_folder = os.path.join(DATASET_PATH, class_name)
    for file in os.listdir(class_folder):
        file_path = os.path.join(class_folder, file)
        img = cv2.imread(file_path)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            X.append(img)
            y.append(idx)

X = np.array(X, dtype="float32") / 255.0
y = to_categorical(y, num_classes=len(class_names))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=1)
model.save(MODEL_PATH)

print("✅ Saved model")