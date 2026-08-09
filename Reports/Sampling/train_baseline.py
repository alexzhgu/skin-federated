"""
Short ResNet-18 training run on the sampled manifest.

Loads ./eda_outputs/sample_manifest.csv, does a stratified 70/15/15 split,
trains, evaluates on test set.

Outputs (./runs/<timestamp>/):
    config.json, train_log.csv, best.pt, test_results.json
"""

from pathlib import Path
from datetime import datetime
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ===== Config =====
MANIFEST = Path("./eda_outputs/sample_manifest.csv")
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0           # set >0 on Linux/Colab; 0 is safest on Windows
EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
PATIENCE = 4

RUN_DIR = Path("./runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SkinDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), self.class_to_idx[row["class"]]


def build_transforms():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / total, correct / total, f1, preds_all, labels_all


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE} | Run dir: {RUN_DIR}")

    df = pd.read_csv(MANIFEST)[["filepath", "class"]]
    print(f"Loaded manifest: {len(df):,} images, {df['class'].nunique()} classes")

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["class"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["class"], random_state=SEED)
    print(f"  train/val/test = {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")

    classes = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_tf, eval_tf = build_transforms()
    train_loader = DataLoader(SkinDataset(train_df, class_to_idx, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(SkinDataset(val_df, class_to_idx, eval_tf),
                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(SkinDataset(test_df, class_to_idx, eval_tf),
                             batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)

    counts = train_df["class"].value_counts()
    weights = torch.tensor([1.0 / counts[c] for c in classes], dtype=torch.float32)
    weights = (weights / weights.sum() * len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    json.dump({
        "seed": SEED, "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "lr": LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE, "num_classes": len(classes),
        "device": str(DEVICE), "classes": classes,
    }, open(RUN_DIR / "config.json", "w"), indent=2)

    log = []
    best_acc, no_improve = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")

        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        log.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
                    "elapsed_s": time.time() - t0})
        pd.DataFrame(log).to_csv(RUN_DIR / "train_log.csv", index=False)
        print(f"  train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} f1 {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, RUN_DIR / "best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop (no improvement for {PATIENCE} epochs)")
                break

    ckpt = torch.load(RUN_DIR / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion)
    print(f"\nTest: loss {test_loss:.4f} acc {test_acc:.4f} macro_f1 {test_f1:.4f}")

    report = classification_report(labels, preds, target_names=classes,
                                    output_dict=True, zero_division=0)
    json.dump({"test_loss": test_loss, "test_acc": test_acc, "test_macro_f1": test_f1,
               "best_val_acc": best_acc, "best_epoch": ckpt["epoch"],
               "per_class": report}, open(RUN_DIR / "test_results.json", "w"), indent=2)

    print(f"Saved to {RUN_DIR.resolve()}")


if __name__ == "__main__":
    main()