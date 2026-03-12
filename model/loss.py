# loss.py
# ACVNet风格损失函数：带掩码的Charbonnier Loss + 左右一致性损失
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothL1LossWithMask(nn.Module):
    """
    Smooth L1 损失，忽略视差图中无效像素（通常为0或负值）
    参考论文公式(4-10)
    """
    def __init__(self, threshold=0.5):
        super(SmoothL1LossWithMask, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) 预测视差图
        target: (B, H, W) 真实视差图（float）
        """
        # 创建有效像素掩码：视差值 > 0（根据数据集调整）
        mask = (target > 0).float()  # (B, H, W)

        # 计算每个像素的 Smooth L1 损失
        diff = torch.abs(pred.squeeze(1) - target)  # (B, H, W)
        loss = torch.where(diff < self.threshold,
                           0.5 * diff ** 2,
                           diff - 0.5 * self.threshold)
        # 应用掩码并求平均（仅统计有效像素）
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss

class CharbonnierLossWithMask(nn.Module):
    """
    ACVNet使用的Charbonnier Loss（鲁棒的L1损失），忽略无效像素
    公式：L = mean( sqrt(d^2 + eps^2) )
    """

    def __init__(self, eps=1e-3):
        super(CharbonnierLossWithMask, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) 预测视差图
        target: (B, H, W) 真实视差图
        """
        # 有效像素掩码：视差值>0
        mask = (target > 0).float()  # (B, H, W)
        diff = pred.squeeze(1) - target  # (B, H, W)

        # Charbonnier Loss计算
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        # 应用掩码并求平均（仅有效像素）
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss


class LeftRightConsistencyLoss(nn.Module):
    """
    ACVNet左右一致性损失：保证左视差和右视差的一致性
    """

    def __init__(self, eps=1e-3):
        super(LeftRightConsistencyLoss, self).__init__()
        self.eps = eps

    def forward(self, disp_left, disp_right):
        """
        disp_left: (B, 1, H, W) 左视差图
        disp_right: (B, 1, H, W) 右视差图
        """
        # 生成网格用于视差warp
        B, C, H, W = disp_left.shape
        x_coords = torch.arange(W, device=disp_left.device).repeat(B, H, 1)  # (B, H, W)
        y_coords = torch.arange(H, device=disp_left.device).repeat(B, W, 1).transpose(1, 2)  # (B, H, W)

        # 1. 将左视差warp到右图视角，计算与右视差的差异
        # 右图像素对应左图的x坐标 = x - disp_right
        warp_x_left = x_coords - disp_right.squeeze(1)  # (B, H, W)
        warp_x_left = torch.clamp(warp_x_left, 0, W - 1)

        # 双线性采样得到warp后的左视差
        disp_left_warped = F.grid_sample(
            disp_left,
            torch.stack([warp_x_left / W * 2 - 1, y_coords / H * 2 - 1], dim=-1),  # 归一化到[-1,1]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # 2. 有效掩码：视差>0 且 warp后的视差有效
        mask = (disp_left > 0).float() * (disp_left_warped > 0).float()  # (B, 1, H, W)

        # 3. 计算Charbonnier一致性损失
        diff = disp_left - disp_left_warped
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss