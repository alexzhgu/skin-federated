"""
Federated learning simulation for skin disease classification.
Implements FedAvg and FedProx over simulated clients, with IID and Dirichlet
non-IID partitioning.
Run examples:
    python train_federated.py --algo fedavg  --partition iid
    python train_federated.py --algo fedavg  --partition dirichlet --alpha 0.5
    python train_federated.py --algo fedprox --partition dirichlet --alpha 0.1 --mu 0.1
"""
import argparse
import copy
import csv
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, f1_score
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
# Partitioning
def partition_iid(labels, num_clients, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return [list(shard) for shard in np.array_split(idx, num_clients)]
def partition_dirichlet(labels, num_clients, alpha, seed):
    """For each class, draw client proportions ~ Dir(alpha) and split that
    class's samples accordingly. Lower alpha => more skew."""
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    client_idx = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * num_clients)
        # cut points along this class's samples
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for client, shard in enumerate(np.split(c_idx, cuts)):
            client_idx[client].extend(shard.tolist())
    for k in range(num_clients):
        rng.shuffle(client_idx[k])
    return client_idx
# Model
def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
# Local training (FedProx when mu > 0, otherwise plain FedAvg local step)
def local_train(model, loader, device, epochs, lr, momentum, mu, scaler):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    global_params = None
    if mu > 0.0:
        global_params = [p.detach().clone() for p in model.parameters()]
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16,
                          enabled=(device.type == "cuda")):
                loss = F.cross_entropy(model(x), y)
            if mu > 0.0:
                prox = 0.0
                for p, g in zip(model.parameters(), global_params):
                    prox = prox + ((p - g) ** 2).sum()
                loss = loss + (mu / 2.0) * prox
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
# Aggregation (weighted by local sample count)
def aggregate(client_states, client_counts):
    total = float(sum(client_counts))
    agg = copy.deepcopy(client_states[0])
    for key in agg.keys():
        if agg[key].dtype.is_floating_point:
            stacked = torch.stack(
                [client_states[i][key].float() * (client_counts[i] / total)
                 for i in range(len(client_states))],
                dim=0,
            )
            agg[key] = stacked.sum(dim=0).to(client_states[0][key].dtype)
        else:
            # Integer buffers (e.g. BatchNorm num_batches_tracked) are not
            # averaged
            agg[key] = client_states[0][key].clone()
    return agg
# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        with autocast(device_type=device.type, dtype=torch.float16,
                      enabled=(device.type == "cuda")):
            out = model(x.to(device, non_blocking=True))
        preds.extend(out.argmax(1).cpu().numpy().tolist())
        targets.extend(y.numpy().tolist())
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, macro_f1, targets, preds
# Federated loop
def run_federated(args, train_idx, val_idx, test_idx, paths, labels, classes,
                  device):
    train_tf, eval_tf = build_transforms(args.img_size)
    num_classes = len(classes)
    # partition the training indices only
    train_labels = labels[train_idx]
    if args.partition == "iid":
        local_parts = partition_iid(train_labels, args.clients, args.seed)
    else:
        local_parts = partition_dirichlet(
            train_labels, args.clients, args.alpha, args.seed
        )
    # map back to absolute indices into paths/labels
    client_indices = [train_idx[np.array(part, dtype=int)] for part in local_parts]
    # skip clients too small to form a usable batch
    client_indices = [c for c in client_indices if len(c) >= 2]
    num_clients = len(client_indices)
    client_sizes = [len(c) for c in client_indices]
    print(f"client sizes: {client_sizes}")

    client_loaders = []
    for idxs in client_indices:
        # drop a size-1 final batch when there is more than one batch
        drop_last = len(idxs) > args.batch_size
        client_loaders.append(DataLoader(
            ManifestDataset(paths[idxs], labels[idxs], train_tf),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=drop_last,
            persistent_workers=args.workers > 0,
        ))
    # central val / test loaders
    val_loader = DataLoader(
        ManifestDataset(paths[val_idx], labels[val_idx], eval_tf),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ManifestDataset(paths[test_idx], labels[test_idx], eval_tf),
        batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    global_model = build_model(num_classes, pretrained=args.pretrained).to(device)
    local_model = build_model(num_classes, pretrained=False).to(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    # communication accounting
    state_bytes = sum(v.numel() * v.element_size()
                      for v in global_model.state_dict().values())
    cumulative_gb = 0.0
    round_to_target = None
    best_val_acc = 0.0
    best_state = copy.deepcopy(global_model.state_dict())
    best_round = 0
    rng = np.random.default_rng(args.seed)
    log_rows = []
    mu = args.mu if args.algo == "fedprox" else 0.0
    for rnd in range(1, args.rounds + 1):
        m = max(1, int(args.frac * num_clients))
        selected = rng.choice(num_clients, size=m, replace=False).tolist()
        client_states, client_counts = [], []
        for cid in selected:
            local_model.load_state_dict(global_model.state_dict())
            state = local_train(
                local_model, client_loaders[cid], device, args.local_epochs,
                args.lr, args.momentum, mu, scaler,
            )
            client_states.append(state)
            client_counts.append(client_sizes[cid])
        if not client_states:
            continue
        new_state = aggregate(client_states, client_counts)
        global_model.load_state_dict(new_state)
        # comms this round: each participant downloads + uploads one model
        cumulative_gb += len(client_states) * 2 * state_bytes / 1e9
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            acc, macro_f1, _, _ = evaluate(global_model, val_loader, device)
            if round_to_target is None and acc >= args.target_acc:
                round_to_target = rnd
            if acc > best_val_acc:
                best_val_acc = acc
                best_state = copy.deepcopy(global_model.state_dict())
                best_round = rnd
            print(f"round {rnd:3d} | val acc {acc:.4f} | macro_f1 {macro_f1:.4f} "
                  f"| comms {cumulative_gb:.2f} GB")
            log_rows.append({
                "round": rnd, "val_acc": acc, "val_macro_f1": macro_f1,
                "cumulative_gb": cumulative_gb,
            })
    # final test
    global_model.load_state_dict(best_state)
    test_acc, test_f1, targets, preds = evaluate(global_model, test_loader, device)
    print(f"\nTEST acc {test_acc:.4f} | macro_f1 {test_f1:.4f} "
          f"(best val {best_val_acc:.4f} @ round {best_round})")
    print(f"rounds to target acc {args.target_acc}: {round_to_target}")
    report = classification_report(
        targets, preds, target_names=[str(c) for c in classes],
        zero_division=0, digits=4,
    )
    return {
        "test_acc": test_acc, "test_macro_f1": test_f1,
        "best_val_acc": best_val_acc, "best_round": best_round,
        "round_to_target": round_to_target,
        "total_comms_gb": cumulative_gb, "report": report,
        "log_rows": log_rows, "global_model": global_model,
    }
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="sub_manifest.csv")
    p.add_argument("--path-col", default="filepath")
    p.add_argument("--label-col", default="class")
    p.add_argument("--algo", choices=["fedavg", "fedprox"], default="fedavg")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=0.1)
    # federated hyperparameters
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=5)
    # optimisation
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    # split / eval
    p.add_argument("--target-acc", type=float, default=0.90)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="fed_runs")
    return p.parse_args()
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | algo: {args.algo} | partition: {args.partition}"
          + (f" alpha={args.alpha}" if args.partition == "dirichlet" else "")
          + (f" mu={args.mu}" if args.algo == "fedprox" else ""))
    paths, labels, classes, train_idx, val_idx, test_idx = load_manifest(
        args.manifest, args.path_col, args.label_col
    )
    print(f"classes: {len(classes)} | train {len(train_idx)} "
          f"val {len(val_idx)} test {len(test_idx)}")
    t0 = time.time()
    result = run_federated(
        args, train_idx, val_idx, test_idx, paths, labels, classes, device
    )
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.algo}_{args.partition}"
    if args.partition == "dirichlet":
        tag += f"_a{args.alpha}"
    if args.algo == "fedprox":
        tag += f"_mu{args.mu}"
    if not args.pretrained:
        tag += "_scratch"
    tag += f"_c{args.clients}"
    tag += f"_seed{args.seed}"
    with open(os.path.join(args.out_dir, f"{tag}_log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "val_acc", "val_macro_f1",
                                          "cumulative_gb"])
        w.writeheader()
        w.writerows(result["log_rows"])
    with open(os.path.join(args.out_dir, f"{tag}_report.txt"), "w") as f:
        f.write(f"test_acc {result['test_acc']:.4f}\n")
        f.write(f"test_macro_f1 {result['test_macro_f1']:.4f}\n")
        f.write(f"round_to_target({args.target_acc}) {result['round_to_target']}\n")
        f.write(f"total_comms_gb {result['total_comms_gb']:.3f}\n\n")
        f.write(result["report"])
    torch.save(result["global_model"].state_dict(),
               os.path.join(args.out_dir, f"{tag}_model.pt"))
    print(f"saved outputs to {args.out_dir}/{tag}_*")
if __name__ == "__main__":
    main()
