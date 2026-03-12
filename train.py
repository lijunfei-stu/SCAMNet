# train.py
# 主训练脚本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import time

from config import Config
from dataset.scared_dataset import ScaredDataset
from model.NMSCANet import NMSCANet
from model.loss import SmoothL1LossWithMask
from utils import setup_logger, compute_epe, compute_pixel_error, save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    config = Config()
    set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger = setup_logger(config.log_file)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 数据加载
    train_dataset = ScaredDataset(
        datapath=config.datapath,
        list_filename=config.train_list,
        training=True,
        crop_size=(config.crop_width, config.crop_height)
    )
    val_dataset = ScaredDataset(
        datapath=config.datapath,
        list_filename=config.test_list,
        training=False,
        crop_size=(config.crop_width, config.crop_height)  # 验证时也裁剪相同尺寸
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 模型
    model = NMSCANet(max_disp=config.max_disp, in_channels=3, base_channels=config.base_channels)
    model.to(device)

    # 损失函数
    criterion = SmoothL1LossWithMask(threshold=0.5)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 学习率调度器（按epoch步长衰减）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_gamma)

    # 训练循环
    start_epoch = 0
    best_val_epe = float('inf')

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        total_epe = 0.0
        iter_count = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            disparity_gt = batch['disparity'].to(device)  # (B, H, W)

            optimizer.zero_grad()

            # 前向传播
            disparity_pred = model(left, right)  # (B, 1, H, W)

            # 计算损失
            loss = criterion(disparity_pred, disparity_gt)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            with torch.no_grad():
                mask = (disparity_gt > 0).float()
                epe = compute_epe(disparity_pred, disparity_gt, mask)
                total_epe += epe
            iter_count += 1

            if (batch_idx + 1) % config.log_freq == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | "
                            f"Loss: {loss.item():.4f} | EPE: {epe:.4f}")

        avg_loss = total_loss / iter_count
        avg_epe = total_epe / iter_count
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} completed | Time: {epoch_time:.2f}s | "
                    f"Avg Loss: {avg_loss:.4f} | Avg EPE: {avg_epe:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 学习率更新
        scheduler.step()

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:  # 每5个epoch验证一次
            val_epe, val_pixel_errors = validate(model, val_loader, device, config.max_disp)
            logger.info(f"Validation | EPE: {val_epe:.4f} | "
                        f"0.5PER: {val_pixel_errors[0]*100:.2f}% | "
                        f"1PER: {val_pixel_errors[1]*100:.2f}% | "
                        f"3PER: {val_pixel_errors[2]*100:.2f}%")

            # 保存最佳模型
            if val_epe < best_val_epe:
                best_val_epe = val_epe
                save_checkpoint(model, optimizer, epoch+1, config, is_best=True)
                logger.info(f"Best model saved with EPE {val_epe:.4f}")

        # 定期保存检查点
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(model, optimizer, epoch+1, config)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

def validate(model, val_loader, device, max_disp):
    """
    验证函数
    """
    model.eval()
    total_epe = 0.0
    total_pixel_errors = [0.0, 0.0, 0.0]
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            disparity_gt = batch['disparity'].to(device)

            disparity_pred = model(left, right)
            mask = (disparity_gt > 0).float()

            # EPE
            epe = compute_epe(disparity_pred, disparity_gt, mask)
            total_epe += epe * left.size(0)

            # 像素误差率
            pixel_errors = compute_pixel_error(disparity_pred, disparity_gt, mask, thresholds=[0.5, 1.0, 3.0])
            for i in range(3):
                total_pixel_errors[i] += pixel_errors[i] * left.size(0)

            num_samples += left.size(0)

    avg_epe = total_epe / num_samples
    avg_pixel_errors = [e / num_samples for e in total_pixel_errors]
    return avg_epe, avg_pixel_errors

if __name__ == "__main__":
    main()