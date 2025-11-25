import os
import argparse
import random
from copy import deepcopy
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from azureml.core import Run

# --- Pipeline for PyTorch ---
class Pipeline(Dataset):
    def __init__(self, folder, transform):
        super(Pipeline, self).__init__()
        self.transform = transform
        self.data = []

        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            # Extract label from folder structure: foldername = label
            label = os.path.basename(folder)
            self.data.append((path, label))

        # Map string labels to integers
        self.labels_map = {l: i for i, l in enumerate(sorted(set(l for _, l in self.data)))}
        self.data = [(img, self.labels_map[label]) for img, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# --- ResNet wrapper ---
class ResNet(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNet, self).__init__()
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return nn.functional.softmax(self.model(x), dim=1)

# --- Main function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, required=True)
    parser.add_argument('--val_folder', type=str, required=True)
    parser.add_argument('--test_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    run = Run.get_context()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = Pipeline(args.train_folder, transform)
    val_ds = Pipeline(args.val_folder, transform)
    test_ds = Pipeline(args.test_folder, transform)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    num_classes = len(train_ds.labels_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = ResNet(base_model, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    best_model = deepcopy(model)
    best_acc = 0
    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    # --- Training loop ---
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_acc, total_samples = 0, 0, 0

        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (out.argmax(1) == target).sum().item()
            total_samples += data.size(0)

        train_loss.append(total_loss / total_samples)
        train_acc.append(total_acc / total_samples)

        model.eval()
        val_total_loss, val_total_acc, val_total_samples = 0, 0, 0
        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss = criterion(out, target)
                val_total_loss += loss.item()
                val_total_acc += (out.argmax(1) == target).sum().item()
                val_total_samples += data.size(0)

        val_loss.append(val_total_loss / val_total_samples)
        val_acc.append(val_total_acc / val_total_samples)

        if val_acc[-1] >= best_acc:
            best_acc = val_acc[-1]
            best_model = deepcopy(model)

        scheduler.step()
        print(f"Epoch {epoch}: train_loss={train_loss[-1]:.4f}, train_acc={train_acc[-1]:.4f}, "
              f"val_loss={val_loss[-1]:.4f}, val_acc={val_acc[-1]:.4f}")

        run.log("train_loss", train_loss[-1])
        run.log("train_acc", train_acc[-1])
        run.log("val_loss", val_loss[-1])
        run.log("val_acc", val_acc[-1])

    # --- Test ---
    best_model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)
            out = best_model(data)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    cf_matrix = confusion_matrix(all_targets, all_preds)
    print("Test Confusion Matrix:\n", cf_matrix)
    print(classification_report(all_targets, all_preds))

    os.makedirs(args.output_folder, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(args.output_folder, "best_model.pth"))
    np.save(os.path.join(args.output_folder, "confusion_matrix.npy"), cf_matrix)
    print(f"Training complete. Model saved at {args.output_folder}/best_model.pth")


if __name__ == "__main__":
    main()