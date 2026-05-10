import torch

@torch.no_grad()
def pixel_accuracy(pred, target, ignore_index=-1):
    valid = target != ignore_index
    correct = (pred[valid] == target[valid]).sum().item()
    total = valid.sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def mean_iou(pred, target, num_classes, ignore_index=-1):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return sum(ious) / max(len(ious), 1)