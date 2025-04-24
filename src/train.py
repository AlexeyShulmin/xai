from __future__ import annotations

"""Training script upgraded for the *CSV + image‑dir* pipeline.

Usage example:

```bash
python -m src.train \
    --csv data/train.csv \
    --img-dir data/train_images \
    --epochs 12 \
    --batch-size 32 \
    --weights models/best_resnet50.pt
```
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import get_loaders
from .model import get_model

# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0

    for imgs, labels in tqdm(loader, total=len(loader), desc='Validate'):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    return running_loss / len(loader.dataset), correct / len(loader.dataset)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CNN for DR detection")
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--img-dir", type=str, default="data/train_images", help="Directory with fundus images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights", type=str, default="models/best_resnet50.pt", help="Where to save best checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, classes = get_loaders(
        args.csv,
        args.img_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # ── Model & Optimisation ────────────────────────────────────────────────
    model = get_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ───────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.weights)
            print("✅ Saved new best model (val acc ↑)")


if __name__ == "__main__":
    main()
