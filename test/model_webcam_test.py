import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = r"E:\Workspace\ai\fruit\fruit_model.h5"
DATASET_DIR = r"E:\Workspace\ai\fruit\dataset\fruits"

model = load_model(MODEL_PATH)
class_names = [name for name in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, name))]

print("Detected classes:", class_names)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("The camera could not be opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (100,100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    clase = class_names[np.argmax(pred)]
    confianza = np.max(pred) * 100

    cv2.putText(frame, f"{clase} ({confianza:.1f}%)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    cv2.imshow("Fruit Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()