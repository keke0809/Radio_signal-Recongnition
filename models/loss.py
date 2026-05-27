# 文件路径: losses/amc_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    针对 RadioML 信号分类的 Focal Loss。
    能够自动降低易分样本（高 SNR）的权重，迫使模型关注难分样本（低 SNR 模糊区间）。
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# def build_loss_fn(loss_type='ce', **kwargs):
#     """
#     Loss 工厂函数，在 train.py 中调用此函数获取 criterion
#     """
#     if loss_type.lower() == 'ce':
#         return nn.CrossEntropyLoss()
#     elif loss_type.lower() == 'focal':
#         return FocalLoss(**kwargs)
#     else:
#         raise ValueError(f"不支持的 Loss 类型: {loss_type}")

def build_loss_fn(loss_type='ce', smoothing=0.1, **kwargs):
    """
    Loss 工厂函数
    smoothing: 标签平滑系数，0.1 代表 90% 留给真实类别，10% 均摊给其余类别
    """
    if loss_type.lower() == 'ce':
        # PyTorch 原生支持的高效标签平滑
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif loss_type.lower() == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"不支持的 Loss 类型: {loss_type}")