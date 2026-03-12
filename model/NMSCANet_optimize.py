import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== 从分割网络移植的核心模块 ======================
class BR_block(nn.Module):
    """
    边界细化残差模块 (Boundary Refinement Block)
    移植自前列腺分割网络，用于特征/视差图的残差优化，增强边缘细节
    """

    def __init__(self, in_channels, out_channels):
        super(BR_block, self).__init__()
        # 主分支双卷积（2D）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # 残差边1x1卷积，对齐通道数
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # 组归一化，适配小batchsize训练，比BN更稳定
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        residual = x
        # 主分支前向传播
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # 残差连接
        residual = self.shortcut(residual)
        x = x + residual
        x = self.relu(x)
        return x


class GCN_Attention_block(nn.Module):
    """
    基于通道注意力的多尺度全局卷积模块（2D）
    移植自前列腺分割网络，适配2D立体匹配特征提取，通过分解式1D卷积实现大核感受野
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, reduction=2):
        super(GCN_Attention_block, self).__init__()
        # 并行分支1：先H维度卷积，再W维度卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_sizes[0], 1),
                      padding=((kernel_sizes[0] - 1) // 2, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_sizes[1]),
                      padding=(0, (kernel_sizes[1] - 1) // 2)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU()
        )
        # 并行分支2：先W维度卷积，再H维度卷积，与分支1形成互补
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_sizes[1]),
                      padding=(0, (kernel_sizes[1] - 1) // 2)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_sizes[0], 1),
                      padding=((kernel_sizes[0] - 1) // 2, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
        )
        # 通道注意力模块，自适应调整各通道特征权重
        self.channel_attn = ChannelAttention2D(in_channels=out_channels,
                                               out_channels=out_channels,
                                               reduction=reduction)
        # 边界细化模块，增强边缘特征
        self.br_block = BR_block(out_channels, out_channels)

    def forward(self, x):
        # 双分支全局特征提取
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # 特征融合
        combined = x1 + x2
        # 通道注意力加权
        attn = self.channel_attn(combined)
        refined = attn * combined
        # 边界细化残差优化
        output = self.br_block(refined)
        return output


class ChannelAttention2D(nn.Module):
    """
    2D通道注意力模块，移植自分割网络的3D版本，适配立体匹配特征提取
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttention2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享MLP，通过1x1卷积实现通道维度的映射
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局池化+MLP映射
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 融合生成注意力权重
        attn = self.sigmoid(avg_out + max_out)  # shape: [N, C, 1, 1]
        # 扩展到特征图尺寸，逐通道加权
        attn = attn.expand(-1, -1, x.size(2), x.size(3))
        return attn


class ChannelAttention3D(nn.Module):
    """
    3D通道注意力模块，移植自分割网络，适配代价体正则化
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        attn = attn.expand(-1, -1, x.size(2), x.size(3), x.size(4))
        return attn


class SCAM_3D_Block(nn.Module):
    """
    空间-通道联合3D注意力模块 (Spatial-Channel Attention Module 3D)
    基于分割网络的SCAM模块重构，适配3D代价体的双域注意力加权
    """

    def __init__(self, in_channels, reduction=4):
        super(SCAM_3D_Block, self).__init__()
        # 空间注意力分支：捕捉视差-空间维度的匹配相关性
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.GroupNorm(16, in_channels // 2),
            nn.PReLU(),
            nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(16, in_channels // 2),
            nn.PReLU(),
            nn.Conv3d(in_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # 通道注意力分支：捕捉特征通道维度的全局依赖
        self.channel_attn = ChannelAttention3D(in_channels, in_channels, reduction=reduction)

    def forward(self, x):
        # 分别生成空间注意力权重和通道注意力权重
        spatial_weight = self.spatial_attn(x)
        channel_weight = self.channel_attn(x)
        # 双域注意力融合，逐元素相乘加权
        attn_weight = spatial_weight + channel_weight
        out = x * attn_weight
        return out


class ASPP_module_3D(nn.Module):
    """
    3D空洞空间金字塔池化模块 (Atrous Spatial Pyramid Pooling 3D)
    移植自分割网络，适配3D代价体正则化
    """

    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_3D, self).__init__()
        # 3D空洞卷积，适配视差维度的膨胀率
        rate_list = (rate, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)
        x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ====================== 新增：2D卷积残差块（修复核心错误） ======================
class Conv2D_Block(nn.Module):
    """
    2D卷积残差块，适配特征提取器的2D输入（替代错误的3D卷积块）
    """

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual='conv'):
        super(Conv2D_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inp_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.GroupNorm(32, out_feat),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.GroupNorm(32, out_feat),
            nn.ReLU())
        self.residual = residual
        if self.residual is not None:
            self.residual_upsampler = nn.Conv2d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Conv3D_Block(nn.Module):
    """
    3D卷积残差块，仅用于代价体正则化的3D UNet（保留原逻辑）
    """

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual='conv'):
        super(Conv3D_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inp_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.GroupNorm(32, out_feat),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel,
                      stride=stride, padding=padding, bias=True),
            nn.GroupNorm(32, out_feat),
            nn.ReLU())
        self.residual = residual
        if self.residual is not None:
            self.residual_upsampler = nn.Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(nn.Module):
    """
    3D反卷积上采样块，用于代价体正则化的解码阶段
    """

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=kernel,
                               stride=stride, padding=padding, output_padding=0, bias=True),
            nn.ReLU())

    def forward(self, x):
        return self.deconv(x)


# ====================== 原NMSCA模块保留 ======================
class NMSCA(nn.Module):
    """
    归一化多尺度空间通道注意力模块，保留原网络核心结构
    """

    def __init__(self, in_channels, reduction=16, dilations=[1, 2, 3]):
        super(NMSCA, self).__init__()
        self.in_channels = in_channels
        # 通道注意力部分
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x_norm = torch.tanh(x)  # 归一化到[-1,1]，提升稳定性
        b, c, h, w = x_norm.size()
        # 通道注意力计算
        gap_out = self.gap(x_norm).view(b, c)
        gmp_out = self.gmp(x_norm).view(b, c)
        a_a = self.mlp(gap_out)
        a_m = self.mlp(gmp_out)
        a_c = (a_a + a_m) / 2.0
        a_c = a_c.view(b, c, 1, 1)
        # 多尺度空间注意力计算
        multi_spatial = []
        for conv in self.spatial_convs:
            multi_spatial.append(conv(x_norm))
        concat_spatial = torch.cat(multi_spatial, dim=1)
        a_s = self.spatial_fusion(concat_spatial)
        # 注意力融合与残差输出
        attention = self.sigmoid(a_c * a_s)
        out = identity + x_norm * attention
        return out


# ====================== 重构后的特征提取器（修复2D/3D卷积混用问题） ======================
class FeatureExtractor(nn.Module):
    """
    重构后的特征提取网络（纯2D卷积），融合UNet多尺度编码器、GCN全局卷积注意力
    """

    def __init__(self, in_channels=3, base_channels=32):
        super(FeatureExtractor, self).__init__()
        self.base_channels = base_channels
        # 编码器下采样层（MaxPool2d）
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.pool4 = nn.MaxPool2d((2, 2))

        # ========== 修复核心错误：改用Conv2D_Block ==========
        # 编码器卷积块（2D残差结构）
        self.conv_blk1 = Conv2D_Block(in_channels, base_channels, residual='conv')
        self.conv_blk2 = Conv2D_Block(base_channels, base_channels * 2, residual='conv')
        self.conv_blk3 = Conv2D_Block(base_channels * 2, base_channels * 4, residual='conv')
        self.conv_blk4 = Conv2D_Block(base_channels * 4, base_channels * 4, residual='conv')
        self.conv_blk5 = Conv2D_Block(base_channels * 4, base_channels * 4, residual='conv')

        # 多尺度全局卷积注意力模块（GCN）
        self.gcn1 = GCN_Attention_block(base_channels, base_channels, kernel_sizes=(11, 11), reduction=2)
        self.gcn2 = GCN_Attention_block(base_channels * 2, base_channels * 2, kernel_sizes=(7, 7), reduction=2)
        self.gcn3 = GCN_Attention_block(base_channels * 4, base_channels * 4, kernel_sizes=(5, 5), reduction=2)
        self.gcn4 = GCN_Attention_block(base_channels * 4, base_channels * 4, kernel_sizes=(3, 3), reduction=2)

        # NMSCA注意力模块
        self.nmsca1 = NMSCA(base_channels)
        self.nmsca2 = NMSCA(base_channels * 2)
        self.nmsca3 = NMSCA(base_channels * 4)
        self.nmsca4 = NMSCA(base_channels * 4)

        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, base_channels * 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器前向传播（纯2D操作）
        # 第1层：1/1输入尺寸 -> 1/2尺寸
        x1 = self.conv_blk1(x)
        x1 = self.gcn1(x1)
        x1 = self.nmsca1(x1)
        x_low1 = self.pool1(x1)

        # 第2层：1/2输入尺寸
        x2 = self.conv_blk2(x_low1)
        x2 = self.gcn2(x2)
        x2 = self.nmsca2(x2)
        x_low2 = self.pool2(x2)

        # 第3层：1/4输入尺寸
        x3 = self.conv_blk3(x_low2)
        x3 = self.gcn3(x3)
        x3 = self.nmsca3(x3)
        x_low3 = self.pool3(x3)

        # 第4层：1/8输入尺寸
        x4 = self.conv_blk4(x_low3)
        x4 = self.gcn4(x4)
        x4 = self.nmsca4(x4)
        x_low4 = self.pool4(x4)

        # 第5层：1/16输入尺寸，基础语义特征
        base = self.conv_blk5(x_low4)
        base = self.fusion_conv(base)

        # 返回4层级多尺度特征
        return x2, x3, x4, base


class CostVolumeAttention(nn.Module):
    """
    优化后的代价体构建模块，融合SCAM空间-通道联合3D注意力
    """

    def __init__(self, max_disp, in_channels):
        super(CostVolumeAttention, self).__init__()
        self.max_disp = max_disp
        self.in_channels = in_channels
        # 相关代价体的注意力卷积
        self.conv3d_att = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # SCAM空间-通道联合3D注意力
        self.scam_attn = SCAM_3D_Block(in_channels * 2, reduction=4)
        # 代价体初始卷积
        self.cost_init_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, left_feat, right_feat):
        b, c, h, w = left_feat.shape
        d = self.max_disp // 16  # 代价体视差维度，与特征下采样倍数对齐
        # 1. 构建相关代价体
        corr_volume = []
        for i in range(d):
            if i == 0:
                corr = (left_feat * right_feat).sum(dim=1, keepdim=True)
            else:
                shifted_right = torch.roll(right_feat, shifts=-i, dims=3)
                shifted_right[:, :, :, -i:] = 0
                corr = (left_feat * shifted_right).sum(dim=1, keepdim=True)
            corr_volume.append(corr)
        corr_volume = torch.stack(corr_volume, dim=2)  # shape: [B, 1, D, H, W]
        att_weight = self.conv3d_att(corr_volume)  # 注意力权重

        # 2. 构建拼接代价体
        concat_volume = []
        for i in range(d):
            if i == 0:
                concat = torch.cat([left_feat, right_feat], dim=1)
            else:
                shifted_right = torch.roll(right_feat, shifts=-i, dims=3)
                shifted_right[:, :, :, -i:] = 0
                concat = torch.cat([left_feat, shifted_right], dim=1)
            concat_volume.append(concat)
        concat_volume = torch.stack(concat_volume, dim=2)  # shape: [B, 2C, D, H, W]

        # 3. 双重注意力加权优化
        attended_volume = concat_volume * att_weight
        attended_volume = self.scam_attn(attended_volume)
        out_volume = self.cost_init_conv(attended_volume)

        return out_volume


class UNet3D_Regularization(nn.Module):
    """
    基于3D UNet的代价体正则化网络（保留原3D逻辑）
    """

    def __init__(self, in_channels, base_channels=32):
        super(UNet3D_Regularization, self).__init__()
        # 编码器下采样层
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.pool3 = nn.MaxPool3d((2, 2, 2))

        # 编码器卷积残差块（3D）
        self.conv_blk1 = Conv3D_Block(in_channels, base_channels, residual='conv')
        self.conv_blk2 = Conv3D_Block(base_channels, base_channels * 2, residual='conv')
        self.conv_blk3 = Conv3D_Block(base_channels * 2, base_channels * 4, residual='conv')
        self.conv_blk4 = Conv3D_Block(base_channels * 4, base_channels * 8, residual='conv')

        # 3D ASPP模块
        rates = (1, 2, 4, 6)
        self.aspp1 = ASPP_module_3D(base_channels * 8, base_channels * 2, rate=rates[0])
        self.aspp2 = ASPP_module_3D(base_channels * 8, base_channels * 2, rate=rates[1])
        self.aspp3 = ASPP_module_3D(base_channels * 8, base_channels * 2, rate=rates[2])
        self.aspp4 = ASPP_module_3D(base_channels * 8, base_channels * 2, rate=rates[3])
        self.aspp_fusion = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 8, kernel_size=1, bias=False),
            nn.GroupNorm(32, base_channels * 8),
            nn.ReLU(inplace=True)
        )

        # 解码器反卷积上采样块
        self.deconv_blk4 = Deconv3D_Block(base_channels * 8, base_channels * 4)
        self.deconv_blk3 = Deconv3D_Block(base_channels * 4, base_channels * 2)
        self.deconv_blk2 = Deconv3D_Block(base_channels * 2, base_channels)

        # 解码器卷积残差块
        self.dec_conv_blk4 = Conv3D_Block(2 * base_channels * 4, base_channels * 4, residual='conv')
        self.dec_conv_blk3 = Conv3D_Block(2 * base_channels * 2, base_channels * 2, residual='conv')
        self.dec_conv_blk2 = Conv3D_Block(2 * base_channels, base_channels, residual='conv')

        # 最终输出卷积
        self.output_conv = nn.Conv3d(base_channels, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.conv_blk1(x)  # 1/1尺寸
        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)  # 1/2尺寸
        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)  # 1/4尺寸
        x_low3 = self.pool3(x3)
        base = self.conv_blk4(x_low3)  # 1/8尺寸

        # ASPP多尺度空洞卷积
        aspp1 = self.aspp1(base)
        aspp2 = self.aspp2(base)
        aspp3 = self.aspp3(base)
        aspp4 = self.aspp4(base)
        aspp_out = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)
        base = self.aspp_fusion(aspp_out)

        # 解码器前向传播
        d4 = torch.cat(
            [F.interpolate(self.deconv_blk4(base), size=x3.shape[2:], mode='trilinear', align_corners=False), x3],
            dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat(
            [F.interpolate(self.deconv_blk3(d_high4), size=x2.shape[2:], mode='trilinear', align_corners=False), x2],
            dim=1)
        d_high3 = self.dec_conv_blk3(d3)

        d2 = torch.cat(
            [F.interpolate(self.deconv_blk2(d_high3), size=x1.shape[2:], mode='trilinear', align_corners=False), x1],
            dim=1)
        d_high2 = self.dec_conv_blk2(d2)

        # 输出最终正则化后的代价体
        out = self.output_conv(d_high2)
        return out


class DisparityRegression(nn.Module):
    """
    视差回归层，保留原网络结构
    """

    def __init__(self, max_disp):
        super(DisparityRegression, self).__init__()
        self.max_disp = max_disp
        self.register_buffer('disp_values', torch.arange(max_disp).float().view(1, max_disp, 1, 1))

    def forward(self, cost_volume):
        # cost_volume: (B, D, H, W) 视差维度的代价体
        prob = F.softmax(cost_volume, dim=1)
        disp = torch.sum(prob * self.disp_values, dim=1, keepdim=True)  # (B, 1, H, W)
        return disp


# ====================== 最终重构的NMSCANet主网络 ======================
class NMSCANet(nn.Module):
    """
    完整的重构后立体匹配网络（修复2D/3D卷积混用问题）
    """

    def __init__(self, max_disp=192, in_channels=3, base_channels=32):
        super(NMSCANet, self).__init__()
        self.max_disp = max_disp
        self.base_channels = base_channels
        # 1. 共享权重的多尺度特征提取器（纯2D）
        self.feature_extractor = FeatureExtractor(in_channels, base_channels)
        # 2. 注意力引导的代价体构建模块
        self.cost_volume_att = CostVolumeAttention(max_disp, base_channels * 4)
        # 3. 基于3D UNet+ASPP的代价体正则化网络
        self.cost_regularization = UNet3D_Regularization(base_channels * 8, base_channels=32)
        # 4. 视差回归层
        self.regression = DisparityRegression(max_disp)
        # 5. 边界细化模块，对视差图做残差优化
        self.disp_refine = nn.Sequential(
            nn.Conv2d(4, base_channels, kernel_size=3, padding=1),
            BR_block(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )
        # 多尺度视差预测头
        self.disp_pred_1x = nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1)
        self.disp_pred_2x = nn.Conv2d(base_channels * 4, 1, kernel_size=3, padding=1)
        self.disp_pred_4x = nn.Conv2d(base_channels * 2, 1, kernel_size=3, padding=1)

    def forward_single(self, left_img, right_img):
        """单方向视差预测（左→右 或 右→左）"""
        b, c, h, w = left_img.shape
        # 1. 提取左右图像的多尺度特征（共享权重）
        left_x2, left_x3, left_x4, left_base = self.feature_extractor(left_img)
        right_x2, right_x3, right_x4, right_base = self.feature_extractor(right_img)

        # 2. 构建注意力加权的3D代价体
        cost_volume = self.cost_volume_att(left_base, right_base)  # (B, 2C, D/16, H/16, W/16)

        # 3. 代价体正则化（保留通道维度，确保5维输入）
        cost_reg = self.cost_regularization(cost_volume)  # (B, 1, D/16, H/16, W/16)

        # 4. 上采样到原始分辨率的代价体（trilinear要求5维输入）
        cost_up = F.interpolate(cost_reg, size=(self.max_disp, h, w), mode='trilinear', align_corners=False)
        # 去掉通道维度，得到 (B, max_disp, h, w)
        cost_up = cost_up.squeeze(1)
        # 此时cost_up已为 (B, D, H, W)，无需permute
        cost_final = cost_up

        # 5. 初始视差回归
        disp_init = self.regression(cost_final)  # (B, 1, H, W)

        # 6. 边界细化残差优化
        refine_input = torch.cat([left_img, disp_init], dim=1)  # (B, 4, H, W)
        disp_residual = self.disp_refine(refine_input)
        disp_final = disp_init + disp_residual  # 残差相加，得到最终视差图

        # 多尺度视差预测（训练模式）
        if self.training:
            # 1/16尺度视差 -> 原始尺寸
            disp_1x = self.disp_pred_1x(left_base)
            disp_1x = F.interpolate(disp_1x, size=(h, w), mode='bilinear', align_corners=False) * 16
            # 1/8尺度视差 -> 原始尺寸
            disp_2x = self.disp_pred_2x(left_x4)
            disp_2x = F.interpolate(disp_2x, size=(h, w), mode='bilinear', align_corners=False) * 8
            # 1/4尺度视差 -> 原始尺寸
            disp_4x = self.disp_pred_4x(left_x3)
            disp_4x = F.interpolate(disp_4x, size=(h, w), mode='bilinear', align_corners=False) * 4
            return disp_final, disp_init, disp_1x, disp_2x, disp_4x
        else:
            return disp_final

    def forward(self, left_img, right_img, return_both=False):
        """
        主前向传播函数
        """
        # 预测左视差（左图→右图）
        if self.training:
            disp_left, disp_init_left, disp_1x, disp_2x, disp_4x = self.forward_single(left_img, right_img)
        else:
            disp_left = self.forward_single(left_img, right_img)

        if not return_both:
            if self.training:
                return disp_left, disp_init_left, disp_1x, disp_2x, disp_4x
            else:
                return disp_left

        # 预测右视差（右图→左图）
        if self.training:
            disp_right, disp_init_right, _, _, _ = self.forward_single(right_img, left_img)
            return disp_left, disp_right, disp_init_left, disp_init_right, disp_1x, disp_2x, disp_4x
        else:
            disp_right = self.forward_single(right_img, left_img)
            return disp_left, disp_right


# ====================== 网络测试代码 ======================
if __name__ == "__main__":
    # 测试模型输入输出尺寸
    model = NMSCANet(max_disp=192, in_channels=3, base_channels=32)
    # 模拟输入：batchsize=2，3通道，1024×1280分辨率（H×W）
    left = torch.randn(1, 3, 1024, 1280)
    right = torch.randn(1, 3, 1024, 1280)

    # 推理模式测试
    model.eval()
    with torch.no_grad():
        disp_left, disp_right = model(left, right, return_both=True)
        print(f"推理模式 - 左视差形状: {disp_left.shape}")  # 预期: [2, 1, 1024, 1280]
        print(f"推理模式 - 右视差形状: {disp_right.shape}")  # 预期: [2, 1, 1024, 1280]

    # 训练模式测试
    model.train()
    outputs = model(left, right, return_both=True)
    print(f"\n训练模式 - 最终左视差形状: {outputs[0].shape}")
    print(f"训练模式 - 最终右视差形状: {outputs[1].shape}")
    print(f"训练模式 - 初始左视差形状: {outputs[2].shape}")
    print(f"训练模式 - 1/16尺度视差形状: {outputs[4].shape}")
    print(f"训练模式 - 1/8尺度视差形状: {outputs[5].shape}")
    print(f"训练模式 - 1/4尺度视差形状: {outputs[6].shape}")