import cv2
import numpy as np
import os
import torch
import torch.nn as nn

MODEL_PATH = "model/fruit_model.pth"
DATASET_DIR = "dataset/fruits"
TEST_DIR = "dataset/test"

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

image_paths = []
for root, _, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".jfif")):
            image_paths.append(os.path.join(root, file))

if not image_paths:
    print("No images found in the test folder")
    exit()

current_idx = 0


def predict_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_expanded = np.transpose(img_norm, (2, 0, 1))
    img_tensor = torch.from_numpy(img_expanded).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    return class_names[pred_class.item()], confidence.item()


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
