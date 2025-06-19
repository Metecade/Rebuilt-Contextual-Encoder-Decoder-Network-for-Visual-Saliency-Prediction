import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- KL散度损失包装器 -------------------- #
class KLDivLossWrapper(nn.Module):
    def __init__(self, temperature=1.0, reduction='batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_fn = nn.KLDivLoss(reduction=reduction)

    def forward(self, y_true, y_pred):
        sum_per_image = torch.sum(y_true, dim=(1, 2, 3), keepdim=True)
        y_true = y_true / (1e-7 + sum_per_image)

        sum_per_image = torch.sum(y_pred, dim=(1, 2, 3), keepdim=True)
        y_pred = y_pred / (1e-7 + sum_per_image)

        return self.loss_fn(y_true + 1e-7, y_pred + 1e-7)