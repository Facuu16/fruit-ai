import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

MODEL_PATH = "model/fruit_model.pth"
DATASET_PATH = "dataset/fruits"

IMG_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

class_names = ["apple", "banana", "mango"]
num_classes = len(class_names)


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


def load_dataset():
    X = []
    y = []

    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Folder not found: {class_folder}")
            continue
        for file in os.listdir(class_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
                file_path = os.path.join(class_folder, file)
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(idx)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)
    return X, y


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Total samples: {len(X)}")
    for i, name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {name}: {count}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FruitClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch [{epoch+1:2d}/{EPOCHS}] "
            f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    main()
