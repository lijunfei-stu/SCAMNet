from torch import nn
import torch
import torch.nn.functional as F
from GCN_SCAM_net.BackBone3D import BackBone3D

class BR_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BR_block, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        # 1x1x1卷积用于调整输入通道数（如果输入和输出通道数不同）
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        # 归一化层（可选）
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        # 残差分支
        residual = x
        # 主分支
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
    def __init__(self, in_channels, out_channels, kernel_sizes, reduction=2):
        """
        kernel_sizes: tuple (k1, k2, k3)，对应三个维度的卷积核尺寸
        reduction: 通道注意力的缩减率（与原代码中的ChannelAttention3D保持一致）
        """
        super(GCN_Attention_block, self).__init__()
        # 并行分支1：层内(x-y) -> 层间(z)
        self.branch1 = nn.Sequential(
            # k1 × 1 × 1
            nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_sizes[0], 1, 1),
                      padding=((kernel_sizes[0]-1)//2, 0, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            # 1 × 1 × k3
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_sizes[2]),
                      padding=(0, 0, (kernel_sizes[2]-1)//2)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            # 1 × k2 × 1
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_sizes[1], 1),
                      padding=(0, (kernel_sizes[1] - 1) // 2, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU()
        )
        # 并行分支2：层间(z) -> 层内(x-y)
        self.branch2 = nn.Sequential(
            # 1 × k2 × 1
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_sizes[1], 1),
                      padding=(0, (kernel_sizes[1] - 1) // 2, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            # 1 × 1 × k3
            nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_sizes[2]),
                      padding=(0, 0, (kernel_sizes[2] - 1) // 2)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
            # k1 × 1 × 1
            nn.Conv3d(out_channels, out_channels, kernel_size=(kernel_sizes[0], 1, 1),
                      padding=((kernel_sizes[0] - 1) // 2, 0, 0)),
            nn.GroupNorm(32, out_channels),
            nn.PReLU(),
        )
        # 通道注意力模块（与原代码一致）
        self.channel_attn = ChannelAttention3D(in_channels=out_channels,
                                              out_channels=out_channels,
                                              reduction=reduction)

        # 引入 Boundery Refinement (BR block)
        self.br_block = BR_block(out_channels, out_channels)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        combined = x1 + x2
        attn = self.channel_attn(combined)
        refined = attn * combined
        # 应用 BR block
        output = self.br_block(refined)
        return output

# 新增通道注意力模块（Channel Attention）
class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        """
        in_channels: 输入特征图的通道数（例如 192，来自 downX 和 fuse1 的拼接）
        out_channels: 输出的注意力权重通道数（应与 fuse1 的通道数一致，例如 64）
        reduction: 通道注意力中隐藏层的通道缩减率
        """
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # 共享的 MLP：将 in_channels 经过两层1×1×1卷积映射到 out_channels
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, out_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)  # shape: [N, out_channels, 1, 1, 1]
        attn = attn.expand(-1, -1, x.size(2), x.size(3), x.size(4))
        return attn

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(32, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DAF3D(nn.Module):
    def __init__(self):
        super(DAF3D, self).__init__()
        self.backbone = BackBone3D()

        # 调整down层的输入通道数（假设U-Net输出通道为[64, 128, 256, 512, 1024]）
        # 替换原始的1x1卷积为多尺度全局卷积模块
        self.down4 = GCN_Attention_block(in_channels=1024, out_channels=128,
                                         kernel_sizes=(3, 3, 1), reduction=2)
        self.down3 = GCN_Attention_block(in_channels=512, out_channels=128,
                                         kernel_sizes=(5, 5, 3), reduction=2)
        self.down2 = GCN_Attention_block(in_channels=256, out_channels=128,
                                         kernel_sizes=(7, 7, 5), reduction=2)
        self.down1 = GCN_Attention_block(in_channels=128, out_channels=128,
                                         kernel_sizes=(11, 11, 7), reduction=2)

        self.fuse1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )

        # 保留原始的空间注意力模块（对输入 torch.cat((downX, fuse1), 1)，通道数 192）
        self.attention4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.attention1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )

        # 新增通道注意力模块（输入同样为 192，输出 64）
        self.channel_attention4 = ChannelAttention3D(in_channels=192, out_channels=64, reduction=2)
        self.channel_attention3 = ChannelAttention3D(in_channels=192, out_channels=64, reduction=2)
        self.channel_attention2 = ChannelAttention3D(in_channels=192, out_channels=64, reduction=2)
        self.channel_attention1 = ChannelAttention3D(in_channels=192, out_channels=64, reduction=2)

        self.refine4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.PReLU()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1),
            nn.GroupNorm(32, 64),
            nn.PReLU(),
        )

        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(64, 64, rate=rates[0])
        self.aspp2 = ASPP_module(64, 64, rate=rates[1])
        self.aspp3 = ASPP_module(64, 64, rate=rates[2])
        self.aspp4 = ASPP_module(64, 64, rate=rates[3])

        self.aspp_conv = nn.Conv3d(256, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)

        self.predict1_4 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_3 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_2 = nn.Conv3d(128, 1, kernel_size=1)
        self.predict1_1 = nn.Conv3d(128, 1, kernel_size=1)

        self.predict2_4 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_3 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv3d(64, 1, kernel_size=1)
        self.predict2_1 = nn.Conv3d(64, 1, kernel_size=1)

        self.predict = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        # 获取U-Net编码器的各层输出（x1到x5）
        x1, x2, x3, x4, x5 = self.backbone(x)

        # 将x5作为最深层输入（原layer4对应x5）
        down4 = self.down4(x5)
        down3 = torch.add(
            F.interpolate(down4, size=x4.size()[2:], mode='trilinear', align_corners=True),
            self.down3(x4)
        )
        down2 = torch.add(
            F.interpolate(down3, size=x3.size()[2:], mode='trilinear', align_corners=True),
            self.down2(x3)
        )
        down1 = torch.add(
            F.interpolate(down2, size=x2.size()[2:], mode='trilinear', align_corners=True),
            self.down1(x2)
        )

        # 关键修复：将所有down层上采样到x2的尺寸（即x1经过pool1后的尺寸）
        target_size = x2.size()[2:]  # 即 [81, 59, 44]
        down4 = F.interpolate(down4, size=target_size, mode='trilinear', align_corners=True)
        down3 = F.interpolate(down3, size=target_size, mode='trilinear', align_corners=True)
        down2 = F.interpolate(down2, size=target_size, mode='trilinear', align_corners=True)
        # down1已经是x2的尺寸，无需上采样

        predict1_4 = self.predict1_4(down4)
        predict1_3 = self.predict1_3(down3)
        predict1_2 = self.predict1_2(down2)
        predict1_1 = self.predict1_1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        # 对每个分支分别计算空间和通道注意力，然后相加融合
        attn_input4 = torch.cat((down4, fuse1), 1)  # shape: [N, 192, H, W, D]
        attn_spatial4 = self.attention4(attn_input4)
        attn_channel4 = self.channel_attention4(attn_input4)
        attn4 = attn_spatial4 + attn_channel4

        attn_input3 = torch.cat((down3, fuse1), 1)
        attn_spatial3 = self.attention3(attn_input3)
        attn_channel3 = self.channel_attention3(attn_input3)
        attn3 = attn_spatial3 + attn_channel3

        attn_input2 = torch.cat((down2, fuse1), 1)
        attn_spatial2 = self.attention2(attn_input2)
        attn_channel2 = self.channel_attention2(attn_input2)
        attn2 = attn_spatial2 + attn_channel2

        attn_input1 = torch.cat((down1, fuse1), 1)
        attn_spatial1 = self.attention1(attn_input1)
        attn_channel1 = self.channel_attention1(attn_input1)
        attn1 = attn_spatial1 + attn_channel1

        refine4 = self.refine4(torch.cat((down4, attn4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attn3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attn2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attn1 * fuse1), 1))

        refine = self.refine(torch.cat((refine1, refine2, refine3, refine4), 1))

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)
        aspp = self.aspp_gn(self.aspp_conv(aspp))
        predict = self.predict(aspp)

        predict1_1 = F.interpolate(predict1_1, size=x.size()[2:], mode='trilinear', align_corners=True)
        predict1_2 = F.interpolate(predict1_2, size=x.size()[2:], mode='trilinear', align_corners=True)
        predict1_3 = F.interpolate(predict1_3, size=x.size()[2:], mode='trilinear', align_corners=True)
        predict1_4 = F.interpolate(predict1_4, size=x.size()[2:], mode='trilinear', align_corners=True)

        predict2_1 = F.interpolate(self.predict2_1(refine1), size=x.size()[2:], mode='trilinear', align_corners=True)
        predict2_2 = F.interpolate(self.predict2_2(refine2), size=x.size()[2:], mode='trilinear', align_corners=True)
        predict2_3 = F.interpolate(self.predict2_3(refine3), size=x.size()[2:], mode='trilinear', align_corners=True)
        predict2_4 = F.interpolate(self.predict2_4(refine4), size=x.size()[2:], mode='trilinear', align_corners=True)
        predict = F.interpolate(predict, size=x.size()[2:], mode='trilinear', align_corners=True)

        if self.training:
            return predict1_1, predict1_2, predict1_3, predict1_4, \
                predict2_1, predict2_2, predict2_3, predict2_4, predict
        else:
            return predict
