import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- KL散度损失包装器 -------------------- #
# class KLDivLossWrapper(nn.Module):
#     def __init__(self, temperature=1.0, reduction='batchmean'):
#         super().__init__()
#         self.temperature = temperature
#         self.reduction = reduction
#         self.loss_fn = nn.KLDivLoss(reduction=reduction)
#
#     def forward(self, y_true, y_pred):
#         sum_per_image = torch.sum(y_true, dim=(1, 2, 3), keepdim=True)
#         y_true = y_true / (1e-7 + sum_per_image)
#
#         sum_per_image = torch.sum(y_pred, dim=(1, 2, 3), keepdim=True)
#         y_pred = y_pred / (1e-7 + sum_per_image)
#
#         return self.loss_fn(y_true + 1e-7, y_pred + 1e-7)

class KLDivLossWrapper(nn.Module):
    def __init__(self, eps=1e-7):
        super(KLDivLossWrapper, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        """
        y_true: (N, C, H, W) ground truth saliency maps (0~255)
        y_pred: (N, C, H, W) predicted saliency maps (0~1)
        """
        eps = self.eps

        # 转 float，避免整数出错
        y_true = y_true.float()
        y_pred = y_pred.float()

        # 归一化为分布
        sum_per_image = y_true.sum(dim=(1, 2, 3), keepdim=True)
        y_true = y_true / (sum_per_image + eps)

        sum_per_image = y_pred.sum(dim=(1, 2, 3), keepdim=True)
        y_pred = y_pred / (sum_per_image + eps)

        # 计算 KLD
        loss = y_true * torch.log((y_true + eps) / (y_pred + eps))
        loss = torch.sum(loss, dim=(1, 2, 3))  # 每张图像求和
        loss = torch.mean(loss)  # batch 求平均

        return loss
