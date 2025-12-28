import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CBAM 注意力模块 (提分关键)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# ==========================================
# 2. 带有注意力的残差块 (AttResBlock)
# ==========================================
class AttResBlock(nn.Module):
    def __init__(self, channels):
        super(AttResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        
        # 添加注意力
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        
        # 应用注意力权重
        res = self.ca(res) * res
        res = self.sa(res) * res
        
        return x + res

# ==========================================
# 3. 解码器模块
# ==========================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # 1x1卷积降维更高效
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.res = AttResBlock(out_channels) # 使用注意力块

    def forward(self, x):
        x = self.reduce(x)
        x = self.res(x)
        return x

# ==========================================
# 4. 主模型 UNet (Attention Enhanced)
# ==========================================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # 权重初始化
        self._init_weights()

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.enc1 = nn.Sequential(AttResBlock(features[0]), AttResBlock(features[0]))
        self.pool1 = nn.Conv2d(features[0], features[1], kernel_size=4, stride=2, padding=1)
        
        self.enc2 = nn.Sequential(AttResBlock(features[1]), AttResBlock(features[1]))
        self.pool2 = nn.Conv2d(features[1], features[2], kernel_size=4, stride=2, padding=1)
        
        self.enc3 = nn.Sequential(AttResBlock(features[2]), AttResBlock(features[2]))
        self.pool3 = nn.Conv2d(features[2], features[3], kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            AttResBlock(features[3]),
            AttResBlock(features[3]),
            AttResBlock(features[3])
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(features[2] * 2, features[2])
        
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(features[1] * 2, features[1])
        
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(features[0] * 2, features[0])
        
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        
        # Encoder
        res1 = self.enc1(x1)
        x2 = self.pool1(res1)
        res2 = self.enc2(x2)
        x3 = self.pool2(res2)
        res3 = self.enc3(x3)
        x4 = self.pool3(res3)
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder
        d3 = self.up3(x4)
        d3 = torch.cat((res3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((res2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((res1, d1), dim=1)
        d1 = self.dec1(d1)
        
        rain_streak = self.out_conv(d1)
        return x + rain_streak