import torch
import torch.nn as nn
import torch.nn.functional as F

class NMSCA(nn.Module):
    """
    Normalized Multi-Scale Spatial Channel Attention Module
    对应论文图4.3
    """
    def __init__(self, in_channels, reduction=16, dilations=[1,2,3]):
        super(NMSCA, self).__init__()
        self.in_channels = in_channels

        # 通道注意力部分
        self.gap = nn.AdaptiveAvgPool2d(1)   # 全局平均池化
        self.gmp = nn.AdaptiveMaxPool2d(1)   # 全局最大池化
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.Softplus(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Softplus()
        )

        # 多尺度空间注意力部分
        self.spatial_convs = nn.ModuleList()
        for d in dilations:
            self.spatial_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.Softplus(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.Softplus()
                )
            )
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(dilations), in_channels, kernel_size=1, bias=False),
            nn.Softplus()
        )

        # Sigmoid 用于最终注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入 x: (B, C, H, W)
        identity = x
        x_norm = torch.tanh(x)   # 归一化到 [-1, 1]

        # 通道注意力
        b, c, h, w = x_norm.size()
        gap_out = self.gap(x_norm).view(b, c)   # (B, C)
        gmp_out = self.gmp(x_norm).view(b, c)
        a_a = self.mlp(gap_out)                  # (B, C)
        a_m = self.mlp(gmp_out)
        a_c = (a_a + a_m) / 2.0                   # (B, C)
        a_c = a_c.view(b, c, 1, 1)                # (B, C, 1, 1)

        # 多尺度空间注意力
        multi_spatial = []
        for conv in self.spatial_convs:
            multi_spatial.append(conv(x_norm))
        concat_spatial = torch.cat(multi_spatial, dim=1)   # (B, C*len, H, W)
        a_s = self.spatial_fusion(concat_spatial)          # (B, C, H, W)

        # 融合注意力和最终输出
        attention = self.sigmoid(a_c * a_s)                # (B, C, H, W)
        out = x_norm + x_norm * attention                   # 残差连接
        return out


class FeatureExtractor(nn.Module):
    """
    特征提取网络，每个卷积层后接 NMSCA
    输出多尺度特征（用于后续拼接）
    """
    def __init__(self, in_channels=3, base_channels=32):
        super(FeatureExtractor, self).__init__()
        self.base_channels = base_channels

        # 这里参考常见的立体匹配网络（如PSMNet/ACVNet）设计
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.nmsca1 = NMSCA(base_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.nmsca2 = NMSCA(base_channels*2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.nmsca3 = NMSCA(base_channels*4)

        # 最后再经过一个卷积将通道数统一到 base_channels*4（或可调）
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.final_nmsca = NMSCA(base_channels*4)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.nmsca1(conv1)

        conv2 = self.conv2(conv1)
        conv2 = self.nmsca2(conv2)

        conv3 = self.conv3(conv2)
        conv3 = self.nmsca3(conv3)

        out = self.final_conv(conv3)
        out = self.final_nmsca(out)
        return out


class CostVolumeAttention(nn.Module):
    """
    基于注意力的代价体构建模块（参考ACVNet）
    输入左右特征，输出注意力加权的拼接代价体
    """
    def __init__(self, max_disp, in_channels):
        super(CostVolumeAttention, self).__init__()
        self.max_disp = max_disp
        self.in_channels = in_channels

        # 用于生成注意力的相关代价体卷积（简化为3D卷积）
        # 先构建相关代价体，然后通过几个3D卷积生成注意力权重
        self.conv3d_att = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()   # 输出注意力权重 (0~1)
        )

    def forward(self, left_feat, right_feat):
        # left_feat, right_feat: (B, C, H, W)
        b, c, h, w = left_feat.shape
        d = self.max_disp

        # 构建相关代价体（correlation volume）
        # 通过shift操作计算每个视差下的相关性（点积）
        # 这里使用简单的循环，实际可优化
        corr_volume = []
        for i in range(d):
            if i == 0:
                corr = (left_feat * right_feat).sum(dim=1, keepdim=True)   # (B,1,H,W)
            else:
                shifted_right = torch.roll(right_feat, shifts=-i, dims=3)
                shifted_right[:, :, :, -i:] = 0   # 右侧移出部分补零
                corr = (left_feat * shifted_right).sum(dim=1, keepdim=True)
            corr_volume.append(corr)
        corr_volume = torch.stack(corr_volume, dim=2)   # (B,1,D,H,W)

        # 通过3D卷积生成注意力权重
        att_weight = self.conv3d_att(corr_volume)       # (B,1,D,H,W)
        att_weight = att_weight.squeeze(1)               # (B,D,H,W)

        # 构建拼接代价体（concatenation volume）
        # 将左右特征在通道维度拼接，然后沿视差维度stack
        concat_volume = []
        for i in range(d):
            if i == 0:
                concat = torch.cat([left_feat, right_feat], dim=1)   # (B,2C,H,W)
            else:
                shifted_right = torch.roll(right_feat, shifts=-i, dims=3)
                shifted_right[:, :, :, -i:] = 0
                concat = torch.cat([left_feat, shifted_right], dim=1)
            concat_volume.append(concat)
        concat_volume = torch.stack(concat_volume, dim=2)   # (B,2C,D,H,W)

        # 应用注意力权重：将注意力扩展到通道维度
        att_weight = att_weight.unsqueeze(1)                # (B,1,D,H,W)
        attended_volume = concat_volume * att_weight        # (B,2C,D,H,W)
        return attended_volume


class Hourglass3D(nn.Module):
    """
    3D沙漏网络用于代价体正则化（简版，可重复堆叠）
    """
    def __init__(self, in_channels):
        super(Hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels*2, in_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 下采样
        out = self.conv1(x)
        out = self.conv2(out)
        # 上采样
        out = self.conv3(out)
        # 残差连接
        return out + x


class DisparityRegression(nn.Module):
    """
    视差回归层：对代价体在视差维度进行softmax加权求和
    """
    def __init__(self, max_disp):
        super(DisparityRegression, self).__init__()
        self.max_disp = max_disp
        self.register_buffer('disp_values', torch.arange(max_disp).float().view(1, max_disp, 1, 1))

    def forward(self, cost_volume):
        # cost_volume: (B, D, H, W) 经过softmax后的概率体
        prob = F.softmax(cost_volume, dim=1)
        disp = torch.sum(prob * self.disp_values, dim=1, keepdim=True)   # (B,1,H,W)
        return disp


class NMSCANet(nn.Module):
    """
    完整的 NMSCANet 模型
    """
    def __init__(self, max_disp=192, in_channels=3, base_channels=32):
        super(NMSCANet, self).__init__()
        self.max_disp = max_disp

        # 特征提取网络
        self.feature_extractor = FeatureExtractor(in_channels, base_channels)

        # 代价体构建（注意力拼接）
        self.cost_volume_att = CostVolumeAttention(max_disp, base_channels*4)   # 特征输出通道为 base_channels*4

        # 3D 正则化网络（可堆叠多个Hourglass）
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        self.hourglass1 = Hourglass3D(base_channels*4)
        self.hourglass2 = Hourglass3D(base_channels*4)
        self.hourglass3 = Hourglass3D(base_channels*4)

        # 输出层，将通道压缩到1，得到代价体
        self.output_conv = nn.Sequential(
            nn.Conv3d(base_channels*4, 1, kernel_size=3, padding=1, bias=False)
        )

        # 视差回归
        self.regression = DisparityRegression(max_disp)

    def forward(self, left_img, right_img):
        # 输入左右图像 (B, C, H, W)
        # 提取特征
        left_feat = self.feature_extractor(left_img)       # (B, C', H/8, W/8)
        right_feat = self.feature_extractor(right_img)

        # 构建注意力拼接代价体
        cost_volume = self.cost_volume_att(left_feat, right_feat)   # (B, 2C', D, H/8, W/8)

        # 3D 正则化
        x = self.conv3d_1(cost_volume)                     # (B, C'/2? 实际为 base_channels*4, D, H/8, W/8)
        x = self.hourglass1(x)
        x = self.hourglass2(x)
        x = self.hourglass3(x)

        # 输出代价体 (B, 1, D, H/8, W/8)
        cost = self.output_conv(x)
        cost = cost.squeeze(1)                              # (B, D, H/8, W/8)

        # 上采样到原始分辨率（可选，论文可能直接输出低分辨率视差，然后上采样）
        # 这里我们上采样到原始输入尺寸
        cost = F.interpolate(cost, size=(left_img.shape[2], left_img.shape[3]), mode='bilinear', align_corners=False)

        # 视差回归
        disparity = self.regression(cost)                   # (B, 1, H, W)
        return disparity


if __name__ == "__main__":
    # 测试模型
    model = NMSCANet(max_disp=192, in_channels=3, base_channels=32)
    left = torch.randn(2, 3, 256, 512)
    right = torch.randn(2, 3, 256, 512)
    disp = model(left, right)
    print(disp.shape)   # 预期 (2, 1, 256, 512)