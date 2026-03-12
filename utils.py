# utils.py
# 工具函数：指标计算、日志、模型保存等

import torch
import numpy as np
import os
import logging

def setup_logger(log_file):
    """
    配置日志记录器
    """
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 文件handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def compute_epe(pred, target, mask):
    """
    计算端点误差（End Point Error）
    pred: (B, 1, H, W)
    target: (B, H, W)
    mask: (B, H, W) 有效像素掩码
    """
    if isinstance(pred, tuple):
        pred = pred[0]

    diff = torch.abs(pred.squeeze(1) - target)  # (B, H, W)
    epe = (diff * mask).sum() / (mask.sum() + 1e-8)
    return epe.item()

def compute_pixel_error(pred, target, mask, thresholds=[0.5, 1.0, 3.0]):
    """
    计算像素误差率（N像素误差）
    thresholds: 列表，如[0.5, 1.0, 3.0] 对应0.5PER, 1PER, 3PER
    返回对应百分比的列表
    """
    diff = torch.abs(pred.squeeze(1) - target)  # (B, H, W)
    results = []
    for th in thresholds:
        error_mask = (diff > th).float() * mask
        error_rate = error_mask.sum() / (mask.sum() + 1e-8)
        results.append(error_rate.item())
    return results

def save_checkpoint(model, optimizer, epoch, config, is_best=False):
    """
    保存模型检查点
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    filename = os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save(checkpoint, filename)
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best.pth')
        torch.save(checkpoint, best_path)
    return filename

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch