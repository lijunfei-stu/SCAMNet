from GCN_SCAM_net.Unet3D  import UNet3D
from torch import nn

class BackBone3D(nn.Module):
    def __init__(self):
        super(BackBone3D, self).__init__()
        # 初始化3D U-Net，调整编码器输出通道以匹配原ResNeXt的层次
        self.unet = UNet3D(num_channels=1, feat_channels=[64, 128, 256, 512, 1024], residual='conv')

        # 提取编码器各层（假设UNet3D的编码器输出为x1, x2, x3, x4, base）
        self.layer1 = self.unet.conv_blk1
        self.layer2 = self.unet.conv_blk2
        self.layer3 = self.unet.conv_blk3
        self.layer4 = self.unet.conv_blk4
        self.layer5 = self.unet.conv_blk5  # 最深层的输出

    def forward(self, x):
        # 编码器各层前向传播
        x1 = self.layer1(x)  # [N, 64, D, H, W]
        x2 = self.layer2(self.unet.pool1(x1))  # [N, 128, D, H/2, W/2]
        x3 = self.layer3(self.unet.pool2(x2))  # [N, 256, D, H/4, W/4]
        x4 = self.layer4(self.unet.pool3(x3))  # [N, 512, D, H/8, W/8]
        x5 = self.layer5(self.unet.pool4(x4))  # [N, 1024, D, H/16, W/16]
        return x1, x2, x3, x4, x5