import argparse
import math
import os
import random

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms.functional as TF

from dataset import build_file_lists, split_dataset
from model_unet import UNet


CLASS_NAMES = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground",
]

COLOR_MAP = np.array([
    [128, 128, 255],   # sky
    [0, 128, 0],       # tree
    [128, 128, 128],   # road
    [0, 255, 0],       # grass
    [0, 0, 255],       # water
    [255, 128, 0],     # building
    [128, 64, 0],      # mountain
    [255, 0, 0],       # foreground
], dtype=np.uint8)


def read_label(path):
    if path.endswith(".txt"):
        mask = np.loadtxt(path, dtype=np.int64)
    else:
        mask = np.array(Image.open(path), dtype=np.int64)

    mask[mask == 255] = -1
    return mask


def resize_mask(mask, image_size):
    mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
    mask_img = TF.resize(mask_img, (image_size, image_size), interpolation=Image.NEAREST)
    mask = np.array(mask_img).astype(np.int64)
    mask[mask == 255] = -1
    return mask


def preprocess_image(img, image_size):
    img_resized = TF.resize(img, (image_size, image_size), interpolation=Image.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor


def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(len(COLOR_MAP)):
        color[mask == cls_id] = COLOR_MAP[cls_id]

    color[mask == -1] = np.array([0, 0, 0], dtype=np.uint8)
    return color


def compute_miou(pred, gt, num_classes=8, ignore_index=-1):
    """
    pred: [H, W], values 0~num_classes-1
    gt: [H, W], values -1, 0~num_classes-1
    """
    valid = gt != ignore_index
    pred = pred[valid]
    gt = gt[valid]

    ious = []

    for cls_id in range(num_classes):
        pred_cls = pred == cls_id
        gt_cls = gt == cls_id

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return float(np.mean(ious))


def load_model(ckpt_path, num_classes, base_ch, device):
    model = UNet(in_channels=3, num_classes=num_classes, base_ch=base_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, img_pil, image_size, device):
    x = preprocess_image(img_pil, image_size).unsqueeze(0).to(device)
    logits = model(x)
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int64)
    return pred


def save_comparison_figure(
    sample_infos,
    preds1,
    preds2,
    preds3,
    miou1,
    miou2,
    miou3,
    diff_scores,
    ckpt_names,
    save_path,
):
    n_rows = len(sample_infos)
    n_cols = 5

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.3 * n_rows),
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Image",
        "GT",
        ckpt_names[0],
        ckpt_names[1],
        ckpt_names[2],
    ]

    for j in range(n_cols):
        axes[0, j].set_title(col_titles[j], fontsize=12)

    for i in range(n_rows):
        image_rgb = sample_infos[i]["image_rgb"]
        gt_mask = sample_infos[i]["gt_mask"]
        gt_color = mask_to_color(gt_mask)

        pred1_color = mask_to_color(preds1[i])
        pred2_color = mask_to_color(preds2[i])
        pred3_color = mask_to_color(preds3[i])

        visuals = [
            image_rgb,
            gt_color,
            pred1_color,
            pred2_color,
            pred3_color,
        ]

        for j in range(n_cols):
            axes[i, j].imshow(visuals[j])
            axes[i, j].axis("off")

        axes[i, 0].set_ylabel(
            f"{sample_infos[i]['name']}\nΔ={diff_scores[i]:.3f}",
            fontsize=9,
        )

        axes[i, 2].set_xlabel(f"mIoU={miou1[i]:.3f}", fontsize=10)
        axes[i, 3].set_xlabel(f"mIoU={miou2[i]:.3f}", fontsize=10)
        axes[i, 4].set_xlabel(f"mIoU={miou3[i]:.3f}", fontsize=10)

    legend_patches = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        color = COLOR_MAP[cls_id] / 255.0
        legend_patches.append(
            mpatches.Patch(color=color, label=f"{cls_id}: {cls_name}")
        )

    legend_patches.append(
        mpatches.Patch(color=(0, 0, 0), label="-1: unknown")
    )

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--checkpoint1", type=str, required=True)
    parser.add_argument("--checkpoint2", type=str, required=True)
    parser.add_argument("--checkpoint3", type=str, required=True)

    parser.add_argument("--ckpt_name1", type=str, default="CE")
    parser.add_argument("--ckpt_name2", type=str, default="Dice")
    parser.add_argument("--ckpt_name3", type=str, default="CE+Dice")

    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=64)

    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--samples_per_figure", type=int, default=5)

    parser.add_argument("--save_dir", type=str, default="vis_compare_topdiff")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, label_paths = build_file_lists(args.data_root)
    train_pair, val_pair, test_pair = split_dataset(image_paths, label_paths, seed=args.seed)

    if args.split == "train":
        selected_images, selected_labels = train_pair
    elif args.split == "val":
        selected_images, selected_labels = val_pair
    else:
        selected_images, selected_labels = test_pair

    paired = list(zip(selected_images, selected_labels))
    random.Random(args.seed).shuffle(paired)

    # 先从候选样本中计算差异，再挑差异最大的样本
    paired = paired[: min(args.num_candidates, len(paired))]

    print("Loading models...")
    model1 = load_model(args.checkpoint1, args.num_classes, args.base_ch, device)
    model2 = load_model(args.checkpoint2, args.num_classes, args.base_ch, device)
    model3 = load_model(args.checkpoint3, args.num_classes, args.base_ch, device)

    all_results = []

    print(f"Running inference on {len(paired)} candidate samples...")
    with torch.no_grad():
        for idx, (img_path, label_path) in enumerate(paired):
            img = Image.open(img_path).convert("RGB")

            img_resized = TF.resize(
                img,
                (args.image_size, args.image_size),
                interpolation=Image.BILINEAR,
            )
            image_rgb = np.array(img_resized).astype(np.uint8)

            gt_mask = read_label(label_path)
            gt_mask = resize_mask(gt_mask, args.image_size)

            pred1 = predict_mask(model1, img, args.image_size, device)
            pred2 = predict_mask(model2, img, args.image_size, device)
            pred3 = predict_mask(model3, img, args.image_size, device)

            m1 = compute_miou(pred1, gt_mask, args.num_classes)
            m2 = compute_miou(pred2, gt_mask, args.num_classes)
            m3 = compute_miou(pred3, gt_mask, args.num_classes)

            diff = max(m1, m2, m3) - min(m1, m2, m3)

            name = os.path.splitext(os.path.basename(img_path))[0]

            all_results.append({
                "name": name,
                "img_path": img_path,
                "label_path": label_path,
                "image_rgb": image_rgb,
                "gt_mask": gt_mask,
                "pred1": pred1,
                "pred2": pred2,
                "pred3": pred3,
                "miou1": m1,
                "miou2": m2,
                "miou3": m3,
                "diff": diff,
            })

            print(
                f"[{idx + 1}/{len(paired)}] {name} | "
                f"{args.ckpt_name1}: {m1:.3f}, "
                f"{args.ckpt_name2}: {m2:.3f}, "
                f"{args.ckpt_name3}: {m3:.3f}, "
                f"diff: {diff:.3f}"
            )

    all_results = sorted(all_results, key=lambda x: x["diff"], reverse=True)
    selected_results = all_results[: args.num_samples]

    ranking_txt = os.path.join(args.save_dir, "top_diff_ranking.txt")
    with open(ranking_txt, "w") as f:
        for rank, item in enumerate(selected_results, start=1):
            f.write(
                f"{rank:03d} | {item['name']} | "
                f"{args.ckpt_name1}: {item['miou1']:.4f} | "
                f"{args.ckpt_name2}: {item['miou2']:.4f} | "
                f"{args.ckpt_name3}: {item['miou3']:.4f} | "
                f"diff: {item['diff']:.4f}\n"
            )

    print(f"\nSaved ranking file: {ranking_txt}")
    print(f"Selected top-{len(selected_results)} samples by mIoU difference.")

    sample_infos = []
    preds1_all, preds2_all, preds3_all = [], [], []
    miou1_all, miou2_all, miou3_all = [], [], []
    diff_all = []

    for item in selected_results:
        sample_infos.append({
            "name": item["name"],
            "image_rgb": item["image_rgb"],
            "gt_mask": item["gt_mask"],
        })

        preds1_all.append(item["pred1"])
        preds2_all.append(item["pred2"])
        preds3_all.append(item["pred3"])

        miou1_all.append(item["miou1"])
        miou2_all.append(item["miou2"])
        miou3_all.append(item["miou3"])

        diff_all.append(item["diff"])

    num_figures = math.ceil(len(sample_infos) / args.samples_per_figure)

    print(f"Saving {num_figures} comparison figure(s) to {args.save_dir} ...")

    for fig_idx in range(num_figures):
        start = fig_idx * args.samples_per_figure
        end = min((fig_idx + 1) * args.samples_per_figure, len(sample_infos))

        save_path = os.path.join(
            args.save_dir,
            f"{args.split}_topdiff_compare_{fig_idx:03d}.png",
        )

        save_comparison_figure(
            sample_infos=sample_infos[start:end],
            preds1=preds1_all[start:end],
            preds2=preds2_all[start:end],
            preds3=preds3_all[start:end],
            miou1=miou1_all[start:end],
            miou2=miou2_all[start:end],
            miou3=miou3_all[start:end],
            diff_scores=diff_all[start:end],
            ckpt_names=[args.ckpt_name1, args.ckpt_name2, args.ckpt_name3],
            save_path=save_path,
        )

        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()