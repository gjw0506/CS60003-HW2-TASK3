import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1.0, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)

        valid_mask = target != self.ignore_index

        target_safe = target.clone()
        target_safe[~valid_mask] = 0

        target_onehot = F.one_hot(
            target_safe.long(),
            num_classes=self.num_classes
        )
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        dims = (0, 2, 3)

        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)

        dice = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )

        return 1.0 - dice.mean()
    

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)