#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张双目图像推理脚本（支持自动中心裁剪至模型输入尺寸 1280×1024）
功能：
- 加载预训练的 LightEndoStereo 模型
- 读取左右目图像和相机重投影矩阵 Q
- 若图像尺寸不是 1280×1024，自动从中心裁剪
- 生成视差图，并转换为深度图
- 可视化并保存视差图、深度图、左右原图
- 生成彩色点云（带RGB）和灰度点云（仅坐标）并保存为PLY文件
- 统计预处理、推理、后处理、点云生成耗时

用法示例：
    python infer_single.py --frame_index 000000
或者直接修改变量后运行
"""

import os
import sys
import json
import time
import yaml
import torch
import numpy as np
import cv2
from os import path as osp

# 添加项目根目录到系统路径
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torchvision.transforms.functional as F
from Dataset.img_reader import rgb_reader
from Models import LightEndoStereo
from tools.visualization import img_rainbow_func
from Evaluators.scared_evaluator import fetch_model, module_remove, disp2depth
from tools.exp_container import ConfigDataContainer

# 模型期望的输入尺寸 (宽, 高)
MODEL_INPUT_SIZE = (1280, 1024)

def load_q_matrix(json_path):
    """从 JSON 文件中读取重投影矩阵 Q (4x4)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    q_matrix = np.array(data['reprojection-matrix'], dtype=np.float32)
    return q_matrix

def preprocess_image(image_path, device):
    """
    读取图像，自动中心裁剪至 MODEL_INPUT_SIZE，并转换为模型输入的张量
    """
    img_np = rgb_reader(image_path)               # numpy array, HWC, RGB
    h, w = img_np.shape[:2]
    target_w, target_h = MODEL_INPUT_SIZE

    # 如果尺寸不匹配，进行中心裁剪
    if (w, h) != (target_w, target_h):
        x = (w - target_w) // 2
        y = (h - target_h) // 2
        img_np = img_np[y:y+target_h, x:x+target_w, :]
        print(f"Image cropped from ({w},{h}) to ({target_w},{target_h}) with offset ({x},{y})")

    img_tensor = F.to_tensor(img_np)               # CHW, [0,1]
    img_tensor = img_tensor.unsqueeze(0)           # 增加 batch 维度
    return img_tensor.to(device)


def generate_pointcloud(disp_np, left_rgb, Q):
    """
    从视差图和左图生成彩色点云
    Args:
        disp_np: (H, W) float32, 视差图
        left_rgb: (H, W, 3) uint8, 左图RGB
        Q: (4,4) float32, 重投影矩阵
    Returns:
        points: (N,3) float32, 世界坐标
        colors: (N,3) uint8, 对应RGB颜色
    """
    # 使用 OpenCV 将视差图转换为 3D 点云 (H, W, 3)
    points_3d = cv2.reprojectImageTo3D(disp_np, Q)   # (H, W, 3) 每个像素的 (X,Y,Z)
    # 有效区域：视差 > 0 且 Z 值有限（排除无效点）
    mask = (disp_np > 0) & np.isfinite(points_3d[..., 2])
    valid_points = points_3d[mask]                    # (N, 3)
    valid_colors = left_rgb[mask]                     # (N, 3)
    return valid_points, valid_colors

def save_ply_color(filepath, points, colors):
    """
    保存彩色点云为 ASCII PLY 文件
    Args:
        filepath: 保存路径
        points: (N,3) float32
        colors: (N,3) uint8
    """
    N = points.shape[0]
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"Saved color PLY: {filepath}")

def save_ply_gray(filepath, points):
    """
    保存灰度点云（仅坐标）为 ASCII PLY 文件
    Args:
        filepath: 保存路径
        points: (N,3) float32
    """
    N = points.shape[0]
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved gray PLY: {filepath}")

def infer_single(left_path, right_path, q_json_path, checkpoint_path, config_path,
                 output_dir='output', frame_index='000000'):
    """主推理函数"""
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config_dict = config['model_config']
    model_config = ConfigDataContainer(**model_config_dict)

    scared_test_config = config.get('scared_test', {})
    maxdisp = scared_test_config.get('maxdisp', 192)

    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. 构建模型并加载权重
    print("Building model...")
    model = fetch_model(model_config, device)
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError:
        state_dict = module_remove(state_dict['model'])
        model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")

    # 4. 读取图像（自动裁剪至模型输入尺寸）
    print("Reading images...")
    left_tensor = preprocess_image(left_path, device)
    right_tensor = preprocess_image(right_path, device)

    # 5. 读取 Q 矩阵
    print("Loading Q matrix...")
    Q = load_q_matrix(q_json_path)

    # 6. 推理计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        disp_est = model(left_tensor, right_tensor)[-1]   # (1, H, W)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time * 1000:.2f} ms")

    # 7. 深度转换计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_post = time.time()
    depth_est = disp2depth(disp_est, Q)                  # (1, H, W)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    post_time = time.time() - start_post
    print(f"Disparity-to-depth time: {post_time * 1000:.2f} ms")
    print(f"Total (inference+post) time: {(inference_time + post_time) * 1000:.2f} ms")

    # 8. 准备可视化图像
    # 生成有效区域掩码（全有效，保持 batch 维度）
    mask = torch.ones_like(disp_est, dtype=torch.bool).cpu()   # (1, H, W)

    # 使用彩虹色映射生成彩色图（输入需包含 batch 维度）
    disp_color = img_rainbow_func.apply(disp_est.cpu(), mask, 'disp')   # (1, 3, H, W)
    depth_color = img_rainbow_func.apply(depth_est.cpu(), mask, 'depth') # (1, 3, H, W)

    # 左/右原图 (转为 uint8)
    left_np = (left_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    right_np = (right_tensor.squeeze().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

    # 9. 保存图像结果
    os.makedirs(output_dir, exist_ok=True)
    save_prefix = osp.join(output_dir, f"frame_{frame_index}")

    # 将彩色张量转为 numpy，去除 batch 维度并转为 (H,W,C)
    disp_color_np = (disp_color.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
    depth_color_np = (depth_color.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)

    # OpenCV 保存（转为 BGR）
    cv2.imwrite(save_prefix + '_disp.png', cv2.cvtColor(disp_color_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_prefix + '_depth.png', cv2.cvtColor(depth_color_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_prefix + '_left.png', cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_prefix + '_right.png', cv2.cvtColor(right_np, cv2.COLOR_RGB2BGR))

    # 保存原始数据
    np.save(save_prefix + '_disp.npy', disp_est.squeeze().cpu().numpy())
    np.save(save_prefix + '_depth.npy', depth_est.squeeze().cpu().numpy())
    print(f"Images saved to {output_dir} with prefix {frame_index}")

    # 10. 点云生成计时
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_pc = time.time()

    # 生成点云
    disp_np = disp_est.squeeze().cpu().numpy()          # (H, W)
    points, colors = generate_pointcloud(disp_np, left_np, Q)

    # 保存彩色点云
    ply_color_path = osp.join(output_dir, f"frame_{frame_index}_color.ply")
    save_ply_color(ply_color_path, points, colors)

    # 保存灰度点云（仅坐标）
    ply_gray_path = osp.join(output_dir, f"frame_{frame_index}_gray.ply")
    save_ply_gray(ply_gray_path, points)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    pc_time = time.time() - start_pc
    print(f"Point cloud generation time: {pc_time * 1000:.2f} ms")
    print(f"Total time (inference+post+pc): {(inference_time + post_time + pc_time) * 1000:.2f} ms")


if __name__ == "__main__":
    # 可根据需要修改为命令行参数解析，这里使用硬编码方便调试

    frame_index = "000000"
    base_path = "/home/ubuntu2404/Desktop/All_datasets/demo/TEST/dataset_1/keyframe_3/data"
    # left_img_path = f"{base_path}/left_finalpass/frame_data{frame_index}.png"
    # right_img_path = f"{base_path}/right_finalpass/frame_data{frame_index}.png"
    # q_matrix_path = f"{base_path}/reprojection_data/frame_data{frame_index}.json"

    left_img_path = f"/home/ubuntu2404/Desktop/All_datasets/20260128_shenzhen1/0128_1628/no_light/left_finalpass/L1.jpg"
    right_img_path = f"/home/ubuntu2404/Desktop/All_datasets/20260128_shenzhen1/0128_1628/no_light/right_finalpass/R1.jpg"
    q_matrix_path = f"/home/ubuntu2404/Desktop/Self-sewing/others/demo_shenzhen/EdgeMed_kidney_datasets_rebuild/camera_parameter/Q_matrix_from_image20260128_inverse.json"

    checkpoint_path = "./TrainLogs/exp1/checkpoint_bestEpe.ckpt"
    config_path = "./configs/lightendostereo_base.yaml"
    output_dir = "./inference_results_scared"

    infer_single(left_img_path, right_img_path, q_matrix_path,
                 checkpoint_path, config_path,
                 output_dir=output_dir, frame_index=frame_index)