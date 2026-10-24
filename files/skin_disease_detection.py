"""Skin Disease Detection System (ResNet18 transfer learning on a SYNTHETIC dataset).

Educational demo only - NOT a medical tool.
Usage:  python skin_disease_detection.py
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_PATH = "dataset"
CLASSES = ["melanoma", "nevus", "eczema", "psoriasis"]
BASE_COLORS = {
    "melanoma": [120, 50, 50],
    "nevus": [180, 120, 90],
    "eczema": [200, 120, 120],
    "psoriasis": [230, 230, 230],
}
MODEL_PATH = "skin_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def generate_image(label):
    """Solid base colour per class + random noise."""
    img = np.zeros((224, 224, 3), dtype=np.int32)
    img[:] = BASE_COLORS[label]
    noise = np.random.randint(0, 50, (224, 224, 3))
    img = np.clip(img + noise, 0, 255)
    return Image.fromarray(img.astype("uint8"))


def create_dataset(n_train=50, n_val=10):
    for split in ["train", "val"]:
        for cls in CLASSES:
            os.makedirs(f"{BASE_PATH}/{split}/{cls}", exist_ok=True)
    for cls in CLASSES:
        for i in range(n_train):
            generate_image(cls).save(f"{BASE_PATH}/train/{cls}/{cls}_{i}.jpg")
        for i in range(n_val):
            generate_image(cls).save(f"{BASE_PATH}/val/{cls}/{cls}_{i}.jpg")
    print("Dataset created")


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False          # freeze backbone
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head (trainable)
    return model.to(device)


def train_model(model, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch + 1}/{epochs} -> Accuracy: {correct / total:.4f}")


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print("Validation Accuracy:", acc)
    return acc


def predict_image(model, class_names, path):
    model.eval()
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
        conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item()


def confidence_level(conf):
    if conf < 0.6:
        return "Low confidence - consult a doctor"
    if conf < 0.8:
        return "Medium confidence"
    return "High confidence prediction"


def main():
    create_dataset()
    train_data = datasets.ImageFolder(f"{BASE_PATH}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{BASE_PATH}/val", transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    class_names = train_data.classes
    print("Using device:", device)
    print("Classes:", class_names)

    model = build_model(len(class_names))
    train_model(model, train_loader, epochs=10)
    evaluate(model, val_loader)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to", MODEL_PATH)

    test_img = f"{BASE_PATH}/val/melanoma/melanoma_1.jpg"
    label, conf = predict_image(model, class_names, test_img)
    print("Prediction:", label)
    print("Confidence:", conf)
    print(confidence_level(conf))


if __name__ == "__main__":
    main()
