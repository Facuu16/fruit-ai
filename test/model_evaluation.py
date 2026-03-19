import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "model/fruit_model.pth"
TEST_DIR = "dataset/test"

CLASS_NAMES = ["apple", "banana", "mango"]
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


def load_test_images(folder):
    images = []
    true_labels = []
    filename_labels = {"Apple": 0, "Banana": 1, "Mango": 2}

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
            for class_name, label in filename_labels.items():
                if class_name in file:
                    img = cv2.imread(os.path.join(folder, file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        true_labels.append(label)
                    break
    return np.array(images), np.array(true_labels)


def main():
    print("Loading PyTorch model...")
    model = FruitClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"Model loaded from: {MODEL_PATH}\n")

    print("Loading test dataset...")
    X_test, y_test = load_test_images(TEST_DIR)
    print(f"Test samples: {len(X_test)}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(y_test == i)
        print(f"  {name}: {count}")

    X_test = X_test.astype("float32") / 255.0
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    X_tensor = torch.from_numpy(X_test)
    y_tensor = torch.from_numpy(y_test)

    print("\nEvaluating model...")
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predictions = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1).numpy()

    y_pred = predictions.numpy()
    y_pred_classes = y_pred

    y_one_hot = np.zeros((len(y_test), len(CLASS_NAMES)))
    y_one_hot[np.arange(len(y_test)), y_test] = 1

    loss = -np.sum(y_one_hot * np.log(probs + 1e-7)) / len(y_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_test, y_pred_classes, target_names=CLASS_NAMES, digits=4)
    print(report)

    cm = confusion_matrix(y_test, y_pred_classes)
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"             Predicted")
    print(f"             apple  banana  mango")
    print(f"Actual apple   {cm[0][0]:<5}  {cm[0][1]:<6}  {cm[0][2]}")
    print(f"       banana  {cm[1][0]:<5}  {cm[1][1]:<6}  {cm[1][2]}")
    print(f"       mango   {cm[2][0]:<5}  {cm[2][1]:<6}  {cm[2][2]}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Fruit Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to: confusion_matrix.png")

    print("\n" + "=" * 60)
    print("README METRICS SUMMARY")
    print("=" * 60)
    print(f"- **Accuracy:** {accuracy*100:.2f}%")
    print(f"- **Loss:** {loss:.4f}")

    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"- **{name.capitalize()}:** Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
