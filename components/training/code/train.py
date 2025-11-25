import os
import argparse
import random
from copy import deepcopy
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from azureml.core import Run

# --- Pipeline for PyTorch ---
class Pipeline(Dataset):
    def __init__(self, folder, transform=None, valid_extensions=(".png", ".jpg", ".jpeg", ".bmp")):
        super(Pipeline, self).__init__()
        self.transform = transform
        self.valid_extensions = valid_extensions
        self.data = []

        # Iterate over subfolders (each subfolder = label)
        for label_name in os.listdir(folder):
            label_folder = os.path.join(folder, label_name)
            if not os.path.isdir(label_folder):
                continue

            label_images = []
            for fname in os.listdir(label_folder):
                if not fname.lower().endswith(self.valid_extensions):
                    continue
                path = os.path.join(label_folder, fname)
                if os.path.isfile(path):
                    label_images.append(path)

            if label_images:
                self.data.extend([(img_path, label_name) for img_path in label_images])
                print(f"[Pipeline] Found {len(label_images)} images for label '{label_name}'", flush=True)

        if not self.data:
            print("[Pipeline] Warning: No images found in the dataset!", flush=True)

        # Map string labels to integers
        unique_labels = sorted(set(label for _, label in self.data))
        self.labels_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.data = [(img, self.labels_map[label]) for img, label in self.data]

        print(f"[Pipeline] Total images: {len(self.data)}", flush=True)
        print(f"[Pipeline] Labels map: {self.labels_map}", flush=True)

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
        # No softmax here, CrossEntropyLoss expects raw logits
        return self.model(x)

# --- Main function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', type=str, required=True)
    parser.add_argument('--val_folder', type=str, required=True)
    parser.add_argument('--test_folder', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--model_name', type=str, default="facial_expression_model")
    args = parser.parse_args()

    run = Run.get_context()

    print("Checking folders:", flush=True)
    print("Train folder exists:", os.path.exists(args.train_folder), flush=True)
    print("Val folder exists:", os.path.exists(args.val_folder), flush=True)
    print("Test folder exists:", os.path.exists(args.test_folder), flush=True)
    print("Train folder content:", os.listdir(args.train_folder) if os.path.exists(args.train_folder) else "MISSING", flush=True)
    print("Val folder content:", os.listdir(args.val_folder) if os.path.exists(args.val_folder) else "MISSING", flush=True)
    print("Test folder content:", os.listdir(args.test_folder) if os.path.exists(args.test_folder) else "MISSING", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = Pipeline(args.train_folder, transform)
    val_ds = Pipeline(args.val_folder, transform)
    test_ds = Pipeline(args.test_folder, transform)

    print(f"Train samples: {len(train_ds)}", flush=True)
    print(f"Val samples: {len(val_ds)}", flush=True)
    print(f"Test samples: {len(test_ds)}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_ds.labels_map)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)
    model = ResNet(base_model, num_classes).to(device)
    print(f"print: using device: {device}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    best_model = deepcopy(model)
    best_acc = 0
    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    # --- Training loop ---
    print("print: Starting training...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_acc, total_samples = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_dl, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (out.argmax(1) == target).sum().item()
            total_samples += data.size(0)

            if batch_idx % 50 == 0 or batch_idx == len(train_dl):
                print(f"[Epoch {epoch}] Batch {batch_idx}/{len(train_dl)} - Loss: {loss.item():.4f}, Acc: {(out.argmax(1) == target).float().mean():.4f}", flush=True)

        train_loss.append(total_loss / total_samples)
        train_acc.append(total_acc / total_samples)

        # --- Validation ---
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
        print(f"Print: Epoch {epoch} - train_loss={train_loss[-1]:.4f}, train_acc={train_acc[-1]:.4f}, "
              f"val_loss={val_loss[-1]:.4f}, val_acc={val_acc[-1]:.4f}", flush=True)
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
    print("Test Confusion Matrix:\n", cf_matrix, flush=True)
    print(classification_report(all_targets, all_preds), flush=True)

    os.makedirs(args.output_folder, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(args.output_folder, f"{args.model_name}.pth"))
    np.save(os.path.join(args.output_folder, "confusion_matrix.npy"), cf_matrix)
    print(f"Training complete. Model saved at {args.output_folder}/{args.model_name}.pth", flush=True)


if __name__ == "__main__":
    main()