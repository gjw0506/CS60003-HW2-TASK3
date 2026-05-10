import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import wandb

from dataset import (
    StanfordBackgroundDataset,
    build_file_lists,
    split_dataset,
)
from model_unet import UNet
from losses import DiceLoss, CEDiceLoss
from metrics import pixel_accuracy, mean_iou


def get_loss(loss_type, num_classes):
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=-1)
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes, )
    elif loss_type == "ce_dice":
        return CEDiceLoss(num_classes=num_classes, ignore_index=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            pred = torch.argmax(logits, dim=1)

            acc = pixel_accuracy(pred, masks)
            miou = mean_iou(pred, masks, num_classes=num_classes)

            total_loss += loss.item()
            total_acc += acc
            total_miou += miou
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches,
        "miou": total_miou / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--loss_type", type=str, choices=["ce", "dice", "ce_dice"], required=True)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="unet-stanford-background")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(
        project=args.project,
        name=args.run_name or args.loss_type,
        config=vars(args),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, label_paths = build_file_lists(args.data_root)
    train_pair, val_pair, test_pair = split_dataset(image_paths, label_paths)

    train_dataset = StanfordBackgroundDataset(
        train_pair[0],
        train_pair[1],
        image_size=(args.image_size, args.image_size),
        augment=True,
    )

    val_dataset = StanfordBackgroundDataset(
        val_pair[0],
        val_pair[1],
        image_size=(args.image_size, args.image_size),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = UNet(
        in_channels=3,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
    ).to(device)

    criterion = get_loss(args.loss_type, args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_miou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        train_miou = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)

            # print("logits shape:", logits.shape)
            # print("mask min/max:", masks.min().item(), masks.max().item())
            # print("mask unique:", torch.unique(masks))

            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits, dim=1)

            acc = pixel_accuracy(pred, masks)
            miou = mean_iou(pred, masks, num_classes=args.num_classes)

            train_loss += loss.item()
            train_acc += acc
            train_miou += miou
            n_batches += 1

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": acc,
                "miou": miou,
            })

        train_log = {
            "train/loss": train_loss / n_batches,
            "train/acc": train_acc / n_batches,
            "train/miou": train_miou / n_batches,
        }

        val_log = evaluate(model, val_loader, criterion, device, args.num_classes)

        log_dict = {
            **train_log,
            "val/loss": val_log["loss"],
            "val/acc": val_log["acc"],
            "val/miou": val_log["miou"],
            "epoch": epoch,
        }

        wandb.log(log_dict)

        print(log_dict)

        if val_log["miou"] > best_miou:
            best_miou = val_log["miou"]
            ckpt_path = f"checkpoints/best_{args.loss_type}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "args": vars(args),
                },
                ckpt_path,
            )
            wandb.save(ckpt_path)

    wandb.finish()


if __name__ == "__main__":
    main()