"""
Centralised baseline for skin disease classification.

Reads paths, labels, and the train/val/test assignment from a manifest CSV
(columns: filepath, class, split).

Usage:
    python train_baseline.py --manifest sub_manifest.csv
    python train_baseline.py --manifest runs/20260707_163655/split_manifest.csv

Outputs (./runs/<timestamp>/):
    config.json, train_log.csv, best.pt, last.pt, test_results.json
"""

from pathlib import Path
from datetime import datetime
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# Data
class ManifestDataset(Dataset):
    """Reads (path, integer label) rows. Labels must already be 0..C-1 ints."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_transforms(img_size):
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def load_manifest(path, path_col="filepath", label_col="class"):
    df = pd.read_csv(path)
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths = df[path_col].astype(str).to_numpy()
    labels = np.array([class_to_idx[l] for l in df[label_col]], dtype=np.int64)
    splits = df["split"].to_numpy()
    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    return paths, labels, classes, train_idx, val_idx, test_idx


# Model
def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def warmup_cosine_lr(epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    preds_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16,
                      enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(y.cpu().tolist())
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / total, correct / total, f1, preds_all, labels_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="sub_manifest.csv")
    p.add_argument("--path-col", default="filepath")
    p.add_argument("--label-col", default="class")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device} | run dir: {run_dir}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    paths, labels, classes, train_idx, val_idx, test_idx = load_manifest(
        args.manifest, args.path_col, args.label_col
    )
    num_classes = len(classes)
    print(f"manifest: {args.manifest}")
    print(f"classes: {num_classes} | train {len(train_idx):,} "
          f"val {len(val_idx):,} test {len(test_idx):,}")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_loader = DataLoader(
        ManifestDataset(paths[train_idx], labels[train_idx], train_tf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, prefetch_factor=4,
    )
    val_loader = DataLoader(
        ManifestDataset(paths[val_idx], labels[val_idx], eval_tf),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        ManifestDataset(paths[test_idx], labels[test_idx], eval_tf),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = build_model(num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    cfg = vars(args).copy()
    cfg.update({"num_classes": num_classes, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx), "classes": list(classes)})
    json.dump(cfg, open(run_dir / "config.json", "w"), indent=2)

    log = []
    best_acc, best_epoch, no_improve = 0.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        lr = warmup_cosine_lr(epoch - 1, args.epochs, args.warmup_epochs, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16,
                          enabled=(device.type == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(lr=f"{lr:.4f}", loss=f"{loss_sum/total:.4f}",
                             acc=f"{correct/total:.4f}")

        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        log.append({"epoch": epoch, "lr": lr,
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "val_macro_f1": val_f1, "elapsed_s": time.time() - t0})
        pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False)
        print(f"  epoch {epoch}: train {train_loss:.4f}/{train_acc:.4f} | "
              f"val {val_loss:.4f}/{val_acc:.4f} f1 {val_f1:.4f} | "
              f"elapsed {time.time()-t0:.0f}s")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_acc": val_acc}, run_dir / "last.pt")

        if val_acc > best_acc:
            best_acc, best_epoch, no_improve = val_acc, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": val_acc}, run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop (no improvement for {args.patience} epochs)")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, test_f1, preds, labels_all = evaluate(model, test_loader, device)
    print(f"\nTEST acc {test_acc:.4f} | macro_f1 {test_f1:.4f} "
          f"(best val {best_acc:.4f} @ epoch {best_epoch})")

    report = classification_report(labels_all, preds,
                                   target_names=[str(c) for c in classes],
                                   output_dict=True, zero_division=0)
    json.dump({"manifest": args.manifest,
               "test_loss": test_loss, "test_acc": test_acc,
               "test_macro_f1": test_f1, "best_val_acc": best_acc,
               "best_epoch": best_epoch, "per_class": report},
              open(run_dir / "test_results.json", "w"), indent=2)

    print(f"saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
