import os
import cv2
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "model/fruit_model.pth"
DATASET_DIR = "dataset/fruits"

IMG_SIZE = 100


class FruitClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(FruitClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 23 * 23, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class_names = [name for name in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, name))]
print("Detected classes:", class_names)

model = FruitClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("The camera could not be opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img.astype("float32") / 255.0
    img_expanded = np.transpose(img_norm, (2, 0, 1))
    img_tensor = torch.from_numpy(img_expanded).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    label = class_names[pred_class.item()]
    conf = confidence.item() * 100

    cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Fruit Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
