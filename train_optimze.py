# train.py
# 主训练脚本，用于训练 NMSCANet 模型（适配ACVNet损失）
import torch
import warnings
import sys
import time
from datetime import timedelta

# 精准过滤NumPy重复导入的警告（仅消除目标警告，保留其他重要警告）
warnings.filterwarnings(
    "ignore",
    message="The NumPy module was reloaded (imported a second time). This can in some cases result in small but subtle issues and is discouraged."
)
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")

import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from tqdm import tqdm  # 导入进度条库

# 导入自定义模块
from config import Config
from dataset.scared_dataset import ScaredDataset  # SCARED 数据集类
from model.loss import CharbonnierLossWithMask, LeftRightConsistencyLoss  # ACVNet损失函数
from utils import setup_logger, compute_epe, compute_pixel_error, save_checkpoint  # 工具函数

# from model.NMSCANet import NMSCANet  # NMSCANet 模型
from model.NMSCANet_optimize1 import NMSCANet  # NMSCANet 模型

# 设置随机种子，确保实验可重复
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 格式化秒数为时分秒
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# 主训练函数，包含数据加载、模型构建、训练循环、验证和模型保存
def main():
    # 1. 加载配置
    config = Config()
    # 设置随机种子
    set_seed(config.seed)
    # 创建检查点保存目录（如果不存在）
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    # 初始化日志记录器（同时输出到文件和控制台）
    logger = setup_logger(config.log_file)

    # 2. 确定运行设备（GPU 或 CPU）
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"========== 训练配置 ==========")
    logger.info(f"使用设备: {device}")
    logger.info(f"训练总轮数: {config.epochs}")
    logger.info(f"批次大小: {config.batch_size}")
    logger.info(f"初始学习率: {config.lr}")
    logger.info(f"权重衰减: {config.weight_decay}")
    logger.info(f"最大视差: {config.max_disp}")
    logger.info(f"==============================")

    # 3. 创建训练集和验证集数据集对象
    train_dataset = ScaredDataset(
        datapath=config.datapath,
        list_filename=config.train_list,
        training=True,
        crop_size=(config.crop_width, config.crop_height)  # 训练时随机裁剪
    )
    val_dataset = ScaredDataset(
        datapath=config.datapath,
        list_filename=config.test_list,
        training=False,
        crop_size=(config.crop_width, config.crop_height)  # 验证时固定裁剪（可根据需要改为全图）
    )

    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=config.num_workers,  # 多线程加载数据
        pin_memory=True,  # 锁页内存，加快 GPU 传输
        drop_last=True  # 丢弃最后一个不完整的 batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    logger.info(f"数据集统计 | 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")
    logger.info(f"数据加载器 | 训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")

    # 5. 构建模型并移至设备
    model = NMSCANet(max_disp=config.max_disp, in_channels=3, base_channels=config.base_channels)
    # model = NMSCANetPlus(max_disp=config.max_disp, in_channels=3, base_channels=config.base_channels)
    model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量 | 总参数: {total_params / 1e6:.2f}M | 可训练参数: {trainable_params / 1e6:.2f}M")

    # 6. 定义ACVNet损失函数、优化器和学习率调度器
    criterion_charbonnier = CharbonnierLossWithMask(eps=1e-3)  # ACVNet Charbonnier Loss
    criterion_lr_consistency = LeftRightConsistencyLoss(eps=1e-3)  # 左右一致性损失
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_step,  # 每多少个 epoch 衰减一次
        gamma=config.lr_decay_gamma  # 衰减因子
    )

    # 初始化起始 epoch 和最佳验证 EPE
    start_epoch = 0
    best_val_epe = float('inf')
    best_val_epoch = 0
    lambda_lr = 1.0  # 左右一致性损失权重（ACVNet默认1.0）

    # 7. 训练循环
    logger.info("\n========== 开始训练 ==========")
    for epoch in range(start_epoch, config.epochs):
        model.train()  # 设置为训练模式
        total_charbonnier_loss = 0.0
        total_lr_loss = 0.0
        total_epe = 0.0
        total_pixel_errors = [0.0, 0.0, 0.0]  # 0.5/1.0/3.0 像素误差率
        iter_count = 0
        epoch_start = time.time()

        # 训练进度条（tqdm）
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{config.epochs}] Train",
            leave=True,
            ncols=150,
            file=sys.stdout
        )

        # 遍历训练数据
        for batch_idx, batch in enumerate(pbar):
            # 将数据移至设备
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            disparity_gt = batch['disparity'].to(device)  # 真实视差图 (B, H, W)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播：得到左右视差图 (B, 1, H, W)
            # disp_left, disp_right = model(left, right, return_both=False)

            output = model(left, right, return_both=True)
            disp_left = output[0]
            disp_right = output[1]

            # 计算ACVNet损失
            # 1. Charbonnier损失（左视差 vs 真实视差）
            loss_charbonnier = criterion_charbonnier(disp_left, disparity_gt)
            # 2. 左右一致性损失
            loss_lr = criterion_lr_consistency(disp_left, disp_right)
            # 3. 总损失
            loss = loss_charbonnier + lambda_lr * loss_lr

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

            # 累计统计信息
            total_charbonnier_loss += loss_charbonnier.item()
            total_lr_loss += loss_lr.item()
            iter_count += 1

            # 计算当前batch的评估指标
            with torch.no_grad():
                mask = (disparity_gt > 0).float()
                # EPE（端点误差）
                epe = compute_epe(disp_left, disparity_gt, mask)
                total_epe += epe
                # 像素误差率
                pixel_errors = compute_pixel_error(disp_left, disparity_gt, mask, thresholds=[0.5, 1.0, 3.0])
                for i in range(3):
                    total_pixel_errors[i] += pixel_errors[i] * left.size(0)

            # 更新进度条显示信息
            current_lr = optimizer.param_groups[0]['lr']
            avg_charbonnier = total_charbonnier_loss / iter_count
            avg_lr = total_lr_loss / iter_count
            avg_total_loss = avg_charbonnier + lambda_lr * avg_lr
            avg_epe = total_epe / iter_count

            # 进度条显示格式
            pbar.set_postfix({
                'LR': f"{current_lr:.6f}",
                'Charbonnier Loss': f"{avg_charbonnier:.4f}",
                'LR Loss': f"{avg_lr:.4f}",
                'Total Loss': f"{avg_total_loss:.4f}",
                'EPE': f"{avg_epe:.4f}",
                '0.5PER': f"{total_pixel_errors[0] / (iter_count * config.batch_size):.4f}",
                '1PER': f"{total_pixel_errors[1] / (iter_count * config.batch_size):.4f}"
            })

        # 关闭进度条
        pbar.close()

        # 计算本epoch的平均指标
        epoch_time = time.time() - epoch_start
        avg_charbonnier_loss = total_charbonnier_loss / iter_count
        avg_lr_loss = total_lr_loss / iter_count
        avg_total_loss = avg_charbonnier_loss + lambda_lr * avg_lr_loss
        avg_epe = total_epe / iter_count
        avg_pixel_errors = [e / (iter_count * config.batch_size) for e in total_pixel_errors]
        current_lr = optimizer.param_groups[0]['lr']

        # 打印epoch级训练总结
        logger.info(
            f"\nEpoch [{epoch + 1}/{config.epochs}] 训练总结 | "
            f"用时: {format_time(epoch_time)} | "
            f"学习率: {current_lr:.6f} | "
            f"平均Charbonnier损失: {avg_charbonnier_loss:.4f} | "
            f"平均LR损失: {avg_lr_loss:.4f} | "
            f"平均总损失: {avg_total_loss:.4f} | "
            f"平均EPE: {avg_epe:.4f} | "
            f"0.5PER: {avg_pixel_errors[0] * 100:.2f}% | "
            f"1PER: {avg_pixel_errors[1] * 100:.2f}% | "
            f"3PER: {avg_pixel_errors[2] * 100:.2f}%"
        )

        # 学习率调度器步进
        scheduler.step()

        # 8. 验证：每5个epoch或最后一个epoch执行一次
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            val_start = time.time()
            val_epe, val_pixel_errors = validate(model, val_loader, device, config.max_disp)
            val_time = time.time() - val_start

            # 验证日志
            logger.info(
                f"Epoch [{epoch + 1}/{config.epochs}] 验证结果 | "
                f"用时: {format_time(val_time)} | "
                f"验证EPE: {val_epe:.4f} | "
                f"0.5PER: {val_pixel_errors[0] * 100:.2f}% | "
                f"1PER: {val_pixel_errors[1] * 100:.2f}% | "
                f"3PER: {val_pixel_errors[2] * 100:.2f}% | "
                f"最佳EPE: {best_val_epe:.4f} (Epoch {best_val_epoch})"
            )

            # 保存最佳模型
            if val_epe < best_val_epe:
                best_val_epe = val_epe
                best_val_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch + 1, config, is_best=True)
                logger.info(f"✅ 最佳模型更新 | 新最佳EPE: {best_val_epe:.4f} (Epoch {best_val_epoch})")

        # 9. 定期保存检查点
        if (epoch + 1) % config.save_freq == 0:
            save_checkpoint(model, optimizer, epoch + 1, config)
            logger.info(f"💾 检查点保存 | Epoch {epoch + 1} 模型已保存至 {config.checkpoint_dir}")

    logger.info("\n========== 训练完成 ==========")
    logger.info(f"最佳验证EPE: {best_val_epe:.4f} (Epoch {best_val_epoch})")


def validate(model, val_loader, device, max_disp):
    """
    修正后的验证函数：适配字典格式的batch，统一指标计算逻辑
    """
    model.eval()  # 切换到评估模式
    total_epe = 0.0
    total_pixel_errors = [0.5, 1.0, 3.0]  # 对应0.5/1.0/3.0像素误差
    num_samples = 0  # 累计样本数

    # 关闭梯度计算，节省显存并加速
    with torch.no_grad():
        # 验证进度条
        pbar = tqdm(
            val_loader,
            desc="Validation",
            leave=False,
            ncols=150,
            file=sys.stdout
        )

        for batch in pbar:
            # 从字典中取值（适配ScaredDataset的返回格式）
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            disparity_gt = batch['disparity'].to(device)

            # 模型前向传播（仅预测左视差）
            model_outputs = model(left, right, return_both=False)  # 或return_both=True后取[0]
            disparity_pred = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs

            # 计算有效像素掩码（仅统计视差>0的区域）
            mask = (disparity_gt > 0).float()
            batch_size = left.size(0)

            # 计算EPE（端点误差）
            epe = compute_epe(disparity_pred, disparity_gt, mask)
            total_epe += epe * batch_size

            # 计算像素误差率（0.5/1.0/3.0阈值）
            pixel_errors = compute_pixel_error(disparity_pred, disparity_gt, mask, thresholds=[0.5, 1.0, 3.0])
            for i in range(3):
                total_pixel_errors[i] += pixel_errors[i] * batch_size

            num_samples += batch_size

            # 更新进度条显示
            pbar.set_postfix({
                'Val EPE': f"{total_epe / num_samples:.4f}",
                '0.5PER': f"{total_pixel_errors[0] / num_samples * 100:.2f}%",
                '1PER': f"{total_pixel_errors[1] / num_samples * 100:.2f}%"
            })

        pbar.close()

    # 计算平均指标
    avg_epe = total_epe / num_samples
    avg_pixel_errors = [e / num_samples for e in total_pixel_errors]

    model.train()  # 切回训练模式
    return avg_epe, avg_pixel_errors


if __name__ == "__main__":
    main()