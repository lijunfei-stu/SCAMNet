# test_optimize.py
# 扩展版本：从视差图生成深度图和点云（含颜色），并记录各阶段耗时
# 使用训练好的 NMSCANet 模型对双目图像进行视差预测
# 需要重投影矩阵 Q（JSON 格式），用于将视差转换为 3D 点
import torch
from PIL import Image
import numpy as np
import os
import time
import json

# 尝试导入彩色图生成所需的库（不强制，失败则跳过彩色输出）
HAS_MPL = False
HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    HAS_MPL = True
except ImportError:
    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        pass

# 导入模型和预处理工具
from model.NMSCANet_optimize1 import NMSCANet
from dataset.data_io import get_transform


def write_ply(filename, points, colors=None):
    """
    将点云写入 PLY 文件（ASCII 格式）
    Args:
        filename: 输出文件路径
        points:   numpy 数组，形状 (N, 3)，float
        colors:   numpy 数组，形状 (N, 3)，uint8 (0-255)，可选
    """
    assert points.shape[1] == 3
    has_color = colors is not None
    num_points = points.shape[0]

    with open(filename, 'w') as f:
        # 头部
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # 数据
        if has_color:
            for i in range(num_points):
                f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} "
                        f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
        else:
            for i in range(num_points):
                f.write(f"{points[i,0]} {points[i,1]} {points[i,2]}\n")


def main():
    # ==================== 配置参数 ====================
    left_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\left_finalpass\frame_data000000.png"
    right_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\right_finalpass\frame_data000000.png"
    # left_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TRAIN\dataset_1\keyframe_1\data\left_finalpass\frame_data000000.png"
    # right_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TRAIN\dataset_1\keyframe_1\data\right_finalpass\frame_data000000.png"
    q_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\frame_data000000.json"          # 重投影矩阵 Q 的 JSON 文件路径
    checkpoint_path = "./checkpoints1/best.pth"
    output_dir = "./infer_result1"                  # 所有输出存放目录
    os.makedirs(output_dir, exist_ok=True)

    max_disp = 192
    base_channels = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================================================

    print(f"使用设备: {device}")

    # -------------------- 1. 加载模型权重 --------------------
    print("正在加载 checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
        max_disp = getattr(config, 'max_disp', max_disp)
        base_channels = getattr(config, 'base_channels', base_channels)
        print(f"从 checkpoint 读取配置: max_disp={max_disp}, base_channels={base_channels}")
    else:
        print(f"checkpoint 中无配置信息，使用默认参数: max_disp={max_disp}, base_channels={base_channels}")

    # -------------------- 2. 构建模型并加载权重 --------------------
    print("正在构建模型...")
    model = NMSCANet(max_disp=max_disp, in_channels=3, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型加载完成。")

    # -------------------- 3. 加载并预处理图像 --------------------
    print("正在加载图像...")
    left_img = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')
    w, h = left_img.size
    print(f"图像尺寸: {w} x {h}")

    transform = get_transform()
    left_tensor = transform(left_img).unsqueeze(0).to(device)
    right_tensor = transform(right_img).unsqueeze(0).to(device)

    # -------------------- 4. 加载重投影矩阵 Q --------------------
    print("正在加载重投影矩阵 Q...")
    with open(q_path, 'r') as f:
        q_data = json.load(f)
    Q = np.array(q_data["reprojection-matrix"], dtype=np.float32)   # 4x4
    print("Q 矩阵:\n", Q)

    # -------------------- 5. 推理得到视差图（计时） --------------------
    print("正在进行推理...")
    start_infer = time.time()
    with torch.no_grad():
        disparity = model(left_tensor, right_tensor, return_both=False)
    end_infer = time.time()
    infer_time = end_infer - start_infer

    # 转换为 numpy 数组并去除 batch 和 channel 维度
    disp_np = disparity.squeeze().cpu().numpy()   # shape: (H, W)

    print(f"推理完成，耗时: {infer_time:.4f} 秒 ({infer_time*1000:.2f} 毫秒)")

    # -------------------- 6. 根据视差图和 Q 计算深度图与点云（计时） --------------------
    start_depth = time.time()

    # 构建像素坐标网格 (u, v)
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)          # 形状 (H, W)
    # 展平为向量，便于批量矩阵乘法
    u_flat = uu.flatten()                # (N,)
    v_flat = vv.flatten()
    d_flat = disp_np.flatten()           # (N,)
    ones_flat = np.ones_like(d_flat)     # (N,)

    # 构建齐次坐标矩阵 (4, N)
    points_4d = np.stack([u_flat, v_flat, d_flat, ones_flat], axis=0)  # (4, N)

    # 矩阵乘法： (4,4) @ (4,N) = (4,N)
    xyz_4d = Q @ points_4d

    # 分离齐次坐标的最后一维 W
    X = xyz_4d[0, :]   # X'
    Y = xyz_4d[1, :]   # Y'
    Z = xyz_4d[2, :]   # Z'
    W = xyz_4d[3, :]   # W

    # 避免除以零，只对 W 非零的点计算 3D 坐标
    valid_mask = np.abs(W) > 1e-8
    # 初始化无效点的坐标为 0（之后会被过滤）
    X_3d = np.zeros_like(X)
    Y_3d = np.zeros_like(Y)
    Z_3d = np.zeros_like(Z)

    X_3d[valid_mask] = X[valid_mask] / W[valid_mask]
    Y_3d[valid_mask] = Y[valid_mask] / W[valid_mask]
    Z_3d[valid_mask] = Z[valid_mask] / W[valid_mask]

    # 深度图就是 Z 坐标，重塑回 (H, W)
    depth_map = Z_3d.reshape(h, w)

    end_depth = time.time()
    depth_time = end_depth - start_depth
    print(f"深度图计算完成，耗时: {depth_time:.4f} 秒")

    # -------------------- 7. 保存深度图 --------------------
    depth_npy_path = os.path.join(output_dir, "depth.npy")
    np.save(depth_npy_path, depth_map)
    print(f"深度图 (npy) 已保存至: {depth_npy_path} (shape {depth_map.shape})")

    # 保存彩色深度图（仅对有效深度区域 > 0 进行归一化）
    mask = depth_map > 0
    if mask.sum() > 0:
        vmin_depth = depth_map[mask].min()
        vmax_depth = depth_map[mask].max()
    else:
        vmin_depth, vmax_depth = 0, 1

    depth_color_path = os.path.join(output_dir, "depth_color.png")
    saved_depth_color = False

    if HAS_MPL:
        try:
            norm = Normalize(vmin=vmin_depth, vmax=vmax_depth)
            cmap = plt.get_cmap('jet')
            colored_depth = cmap(norm(depth_map))          # RGBA (H, W, 4)
            colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(colored_depth_rgb).save(depth_color_path)
            print(f"彩色深度图 (matplotlib) 已保存至: {depth_color_path}")
            saved_depth_color = True
        except Exception as e:
            print(f"使用 matplotlib 生成彩色深度图失败: {e}")

    if not saved_depth_color and HAS_CV2:
        try:
            depth_norm = ((depth_map - vmin_depth) / (vmax_depth - vmin_depth + 1e-8) * 255).astype(np.uint8)
            colored_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(colored_rgb).save(depth_color_path)
            print(f"彩色深度图 (OpenCV) 已保存至: {depth_color_path}")
            saved_depth_color = True
        except Exception as e:
            print(f"使用 OpenCV 生成彩色深度图失败: {e}")

    if not saved_depth_color:
        print("警告: 未找到 matplotlib 或 OpenCV，无法生成彩色深度图。")

    # -------------------- 8. 生成点云（计时） --------------------
    start_cloud = time.time()

    # 获取左图颜色（用于彩色点云）
    left_np = np.array(left_img)                     # (H, W, 3) uint8
    colors_flat = left_np.reshape(-1, 3)             # (N, 3)

    # 过滤无效点（视差 <=0 或 深度 <=0）
    # 有效点的条件：视差 > 0 且 深度 > 0 且 W 有效（前面已用 valid_mask）
    valid_points = (d_flat > 0) & (depth_map.flatten() > 0) & valid_mask

    # 提取有效点坐标和颜色
    points_valid = np.stack([X_3d, Y_3d, Z_3d], axis=1)[valid_points]   # (M, 3)
    colors_valid = colors_flat[valid_points]                            # (M, 3)

    print(f"有效点数: {points_valid.shape[0]} (总点数: {d_flat.shape[0]})")

    # 保存无颜色点云
    ply_no_color_path = os.path.join(output_dir, "points.ply")
    write_ply(ply_no_color_path, points_valid, colors=None)
    print(f"无颜色点云已保存至: {ply_no_color_path}")

    # 保存带颜色点云
    ply_color_path = os.path.join(output_dir, "points_color.ply")
    write_ply(ply_color_path, points_valid, colors=colors_valid)
    print(f"彩色点云已保存至: {ply_color_path}")

    end_cloud = time.time()
    cloud_time = end_cloud - start_cloud
    print(f"点云生成完成，耗时: {cloud_time:.4f} 秒")

    # -------------------- 9. 汇总耗时 --------------------
    print("\n==================== 耗时统计 ====================")
    print(f"视差推理:    {infer_time:.4f} 秒 ({infer_time*1000:.2f} 毫秒)")
    print(f"深度计算:    {depth_time:.4f} 秒 ({depth_time*1000:.2f} 毫秒)")
    print(f"点云生成:    {cloud_time:.4f} 秒 ({cloud_time*1000:.2f} 毫秒)")
    print(f"总耗时:      {infer_time + depth_time + cloud_time:.4f} 秒")
    print("===================================================")


if __name__ == '__main__':
    main()