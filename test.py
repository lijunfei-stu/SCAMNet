# test.py
# 使用训练好的 NMSCANet 模型对双目图像进行视差预测，并基于重投影矩阵Q生成深度图、点云图（带耗时统计）
import torch
from PIL import Image
import numpy as np
import os
import time
from model.NMSCANet import NMSCANet
from dataset.data_io import get_transform
import json
from plyfile import PlyData, PlyElement  # 用于保存点云为PLY格式
import subprocess
import sys

# 从 JSON 文件中读取重投影矩阵 Q (4x4)
def load_q_matrix(json_path):
    """
    加载重投影矩阵Q
    参数:
        json_path: Q矩阵的JSON文件路径
    返回:
        Q: 4x4 numpy数组，重投影矩阵
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    q_matrix = np.array(data['reprojection-matrix'], dtype=np.float32)
    return q_matrix


def disp_to_depth(disp_np, Q):
    """
    视差图转换为深度图（基于重投影矩阵Q）
    核心公式：Z = Q[2,3] / (disp + Q[3,3] - Q[0,3])
    （Q矩阵结构：[1,0,0,-cx], [0,1,0,-cy], [0,0,0,f], [0,0,1/B,0]）
    参数:
        disp_np: 视差图numpy数组，shape=(H,W)
        Q: 4x4重投影矩阵
    返回:
        depth_np: 深度图numpy数组，shape=(H,W)
        convert_time: 转换耗时（秒）
    """
    start_time = time.time()

    # 提取Q矩阵关键参数
    fx = Q[2, 3]  # 焦距f
    cx = -Q[0, 3]  # 主点x坐标
    baseline_reciprocal = Q[3, 2]  # 1/基线距离B

    # 避免除零错误：将视差为0的区域设为极小值
    disp_np = np.where(disp_np == 0, 1e-8, disp_np)

    # 计算深度图：Z = fx * B / disp = fx / (disp * (1/B))
    depth_np = fx / (disp_np * baseline_reciprocal)

    # 处理异常值（如无穷大、NaN）
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    convert_time = time.time() - start_time
    return depth_np, convert_time


def depth_to_pointcloud(depth_np, Q):
    """
    深度图转换为点云（笛卡尔坐标系）
    公式：
        x = (u - cx) * Z / fx
        y = (v - cy) * Z / fy
        z = Z
    其中u,v为像素坐标，cx/cy为主点，fx/fy为焦距，Z为深度值
    参数:
        depth_np: 深度图numpy数组，shape=(H,W)
        Q: 4x4重投影矩阵
    返回:
        point_cloud: 点云数组，shape=(N,3)，N为有效像素数
        convert_time: 转换耗时（秒）
    """
    start_time = time.time()

    # 从Q矩阵提取相机内参
    cx = -Q[0, 3]  # 主点x坐标
    cy = -Q[1, 3]  # 主点y坐标
    fx = Q[2, 3]  # 焦距fx（假设fx=fy）
    fy = fx  # 简化处理，若有差异可从Q矩阵其他位置提取

    # 获取图像尺寸
    h, w = depth_np.shape

    # 生成像素坐标网格
    u = np.arange(w)  # 列坐标（宽度方向）
    v = np.arange(h)  # 行坐标（高度方向）
    u_grid, v_grid = np.meshgrid(u, v)  # shape: (H,W)

    # 计算点云坐标
    z = depth_np  # 深度值Z
    x = (u_grid - cx) * z / fx  # 世界坐标系x
    y = (v_grid - cy) * z / fy  # 世界坐标系y

    # 合并坐标并过滤无效点（深度为0的点）
    points = np.stack([x, y, z], axis=-1)  # shape: (H,W,3)
    points = points.reshape(-1, 3)  # shape: (H*W,3)
    valid_mask = points[:, 2] > 0  # 只保留深度>0的点
    point_cloud = points[valid_mask]  # shape: (N,3)

    convert_time = time.time() - start_time
    return point_cloud, convert_time


def save_pointcloud(point_cloud, save_path):
    """
    将点云保存为PLY格式文件
    参数:
        point_cloud: 点云数组，shape=(N,3)
        save_path: PLY文件保存路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转换为PLY格式所需的数据结构
    points = [(x, y, z) for x, y, z in point_cloud]
    ply_element = PlyElement.describe(
        np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]),
        'vertex'
    )

    # 保存PLY文件
    PlyData([ply_element], text=True).write(save_path)
    print(f"点云已保存至: {save_path} (有效点数: {len(point_cloud)})")


def main():
    # ==================== 配置参数（请根据实际情况修改） ====================
    # 输入图像路径
    left_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\left_finalpass\frame_data000000.png"  # 左图路径
    right_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\dataset_2\keyframe_1\data\right_finalpass\frame_data000000.png"  # 右图路径
    # 模型权重文件路径
    checkpoint_path = r"C:\Users\12700\Desktop\All_datasets\weights_results\SCAMNet\checkpoints\best.pth"  # 训练好的模型权重
    # 重投影矩阵Q文件路径
    q_json_path = r"C:\Users\12700\Desktop\All_datasets\SCARED\TEST\frame_data000000.json"
    # 输出文件路径配置
    output_disp_path = "./infer_result/disparity.npy"  # 视差图保存路径
    output_depth_path = "./infer_result/depth.npy"  # 深度图保存路径
    output_pcd_path = "./infer_result/point_cloud.ply"  # 点云保存路径

    # 模型结构参数（如果 checkpoint 中未保存 config，则使用以下默认值）
    max_disp = 192  # 最大视差
    base_channels = 32  # 基础通道数

    # 运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # -------------------- 1. 加载模型权重 --------------------
    print("=" * 50)
    print("1. 加载模型权重...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

    # 尝试从 checkpoint 中读取配置参数（如果保存了 config 对象）
    if 'config' in checkpoint:
        config = checkpoint['config']
        # 使用 getattr 安全获取参数，若不存在则使用默认值
        max_disp = getattr(config, 'max_disp', max_disp)
        base_channels = getattr(config, 'base_channels', base_channels)
        print(f"从 checkpoint 读取配置: max_disp={max_disp}, base_channels={base_channels}")
    else:
        print(f"checkpoint 中无配置信息，使用默认参数: max_disp={max_disp}, base_channels={base_channels}")

    # -------------------- 2. 构建模型并加载权重 --------------------
    print("=" * 50)
    print("2. 构建模型并加载权重...")
    model = NMSCANet(max_disp=max_disp, in_channels=3, base_channels=base_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    print("模型加载完成。")

    # -------------------- 3. 加载并预处理图像 --------------------
    print("=" * 50)
    print("3. 加载并预处理图像...")
    left_img = Image.open(left_path).convert('RGB')
    right_img = Image.open(right_path).convert('RGB')
    w, h = left_img.size
    print(f"图像尺寸: {w} x {h}")

    # 获取标准化变换 (ToTensor + Normalize)
    transform = get_transform()
    # 添加 batch 维度并移至指定设备
    left_tensor = transform(left_img).unsqueeze(0).to(device)  # shape: (1, 3, H, W)
    right_tensor = transform(right_img).unsqueeze(0).to(device)

    # -------------------- 4. 加载重投影矩阵Q --------------------
    print("=" * 50)
    print("4. 加载重投影矩阵Q...")
    Q = load_q_matrix(q_json_path)
    print(f"Q矩阵:\n{Q}")

    # -------------------- 5. 推理得到视差图 --------------------
    print("=" * 50)
    print("5. 开始视差图推理...")
    start_time = time.time()
    with torch.no_grad():
        disparity = model(left_tensor, right_tensor)
    disp_infer_time = time.time() - start_time

    # 转换为numpy数组（去除batch/channel维度）
    disp_np = disparity.squeeze().cpu().numpy()  # shape: (H, W)
    print(f"视差图推理完成 | 耗时: {disp_infer_time:.4f}秒 ({disp_infer_time * 1000:.2f}毫秒)")
    print(f"视差图范围: [{disp_np.min():.4f}, {disp_np.max():.4f}] | 形状: {disp_np.shape}")

    # -------------------- 6. 视差图转深度图 --------------------
    print("=" * 50)
    print("6. 视差图转换为深度图...")
    depth_np, depth_convert_time = disp_to_depth(disp_np, Q)
    print(f"深度图转换完成 | 耗时: {depth_convert_time:.4f}秒 ({depth_convert_time * 1000:.2f}毫秒)")
    print(f"深度图范围: [{depth_np.min():.4f}, {depth_np.max():.4f}] | 形状: {depth_np.shape}")

    # -------------------- 7. 深度图转点云 --------------------
    print("=" * 50)
    print("7. 深度图转换为点云...")
    point_cloud, pcd_convert_time = depth_to_pointcloud(depth_np, Q)
    print(f"点云转换完成 | 耗时: {pcd_convert_time:.4f}秒 ({pcd_convert_time * 1000:.2f}毫秒)")
    print(f"有效点云数量: {len(point_cloud)}")

    # -------------------- 8. 保存所有结果 --------------------
    print("=" * 50)
    print("8. 保存结果文件...")

    # 8.1 保存视差图（npy + 可视化PNG）
    os.makedirs(os.path.dirname(output_disp_path), exist_ok=True)
    np.save(output_disp_path, disp_np)
    # 视差图可视化（归一化到0-255）
    disp_vis = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8) * 255
    disp_vis_path = output_disp_path.replace('.npy', '.png')
    Image.fromarray(disp_vis.astype(np.uint8)).save(disp_vis_path)
    print(f"✅ 视差图已保存: {output_disp_path} | 可视化: {disp_vis_path}")

    # 8.2 保存深度图（npy + 可视化PNG）
    np.save(output_depth_path, depth_np)
    # 深度图可视化（截断异常值后归一化）
    depth_vis = np.clip(depth_np, 0, np.percentile(depth_np, 95))  # 截断95%分位数以上的异常值
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8) * 255
    depth_vis_path = output_depth_path.replace('.npy', '.png')
    Image.fromarray(depth_vis.astype(np.uint8)).save(depth_vis_path)
    print(f"✅ 深度图已保存: {output_depth_path} | 可视化: {depth_vis_path}")

    # 8.3 保存点云（PLY格式）
    save_pointcloud(point_cloud, output_pcd_path)

    # -------------------- 9. 输出总耗时统计 --------------------
    print("=" * 50)
    total_time = disp_infer_time + depth_convert_time + pcd_convert_time
    print("📊 总耗时统计:")
    print(f"  - 视差推理: {disp_infer_time:.4f}秒")
    print(f"  - 深度转换: {depth_convert_time:.4f}秒")
    print(f"  - 点云转换: {pcd_convert_time:.4f}秒")
    print(f"  - 总计耗时: {total_time:.4f}秒 ({total_time * 1000:.2f}毫秒)")
    print("=" * 50)
    print("所有处理完成！")

if __name__ == '__main__':
    main()