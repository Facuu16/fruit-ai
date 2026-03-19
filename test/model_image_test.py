import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH = r"E:\Workspace\ai\fruit\fruit_model.h5"
DATASET_DIR = r"E:\Workspace\ai\fruit\dataset\fruits"
TEST_DIR = r"E:\Workspace\ai\fruit\dataset\test"

model = load_model(MODEL_PATH)
class_names = [name for name in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, name))]

print("Detected classes:", class_names)

image_paths = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".jfif")):
            image_paths.append(os.path.join(root, file))

if not image_paths:
    print("⚠ No images found in the test folder")
    exit()

current_idx = 0

def predict_image(path):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (100, 100))
    img_norm = img_resized.astype("float32") / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)
    pred = model.predict(img_expanded, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    return class_names[pred_class], confidence

while True:
    path = image_paths[current_idx]
    label, conf = predict_image(path)
    
    img = cv2.imread(path)
    text = f"{label} ({conf*100:.1f}%)"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Test Viewer", img)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break
    elif key == ord('d'):
        current_idx = (current_idx + 1) % len(image_paths)
    elif key == ord('a'):
        current_idx = (current_idx - 1) % len(image_paths)

cv2.destroyAllWindows()