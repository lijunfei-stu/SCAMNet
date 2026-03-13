# config.py
# 训练配置参数

class Config:
    # 数据集路径
    datapath = 'C:/Users/12700/Desktop/All_datasets/SCARED'          # 请修改为实际数据集根目录
    train_list = './filenames/SCARED_train.txt'      # 训练集文件列表
    test_list = './filenames/SCARED_test.txt'        # 测试集文件列表

    # 训练参数
    max_disp = 192                          # 最大视差
    base_channels = 32                      # 特征提取基础通道数
    crop_width = 640                          # 训练裁剪宽度
    crop_height = 480                         # 训练裁剪高度
    batch_size = 1                            # 根据GPU内存调整
    num_workers = 4                            # 数据加载线程数
    epochs = 146                               # 论文中训练轮数
    lr = 0.001                                  # 初始学习率
    lr_decay_step = 50                          # 学习率衰减步长（epoch）
    lr_decay_gamma = 0.1                         # 学习率衰减因子
    weight_decay = 1e-4                          # 权重衰减
    save_freq = 10                                # 保存检查点的频率（epoch）
    log_freq = 50                                 # 打印日志的频率（iter）

    # 设备
    device = 'cuda'                               # 或 'cpu'
    seed = 2024                                    # 随机种子

    # 模型保存路径
    checkpoint_dir = './checkpoints2'
    log_file = '../train_logs/train2.log'