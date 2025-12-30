import torch
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInverse

class DWTForward(nn.Module):
    def __init__(self, J=1, mode='zero', wave='haar'):
        super(DWTForward, self).__init__()
        self.J = J
        self.mode = mode
        self.wave = wave
        assert wave == 'haar', "Only Haar wavelet is supported in this local implementation"

    def forward(self, x):
        # x: [B, C, H, W]
        # output: yL, yH
        # yL: [B, C, H/2, W/2] (low freq)
        # yH: list of [B, C, 3, H/2, W/2] (high freq)
        
        yl = x
        yh = []
        for j in range(self.J):
            b, c, h, w = yl.shape
            
            # Haar Wavelet Transform
            # Low-Low (Approximation)
            ll = (yl[:, :, 0::2, 0::2] + yl[:, :, 0::2, 1::2] + 
                  yl[:, :, 1::2, 0::2] + yl[:, :, 1::2, 1::2]) / 2
            
            # High-Low (Horizontal Detail)
            lh = (yl[:, :, 0::2, 0::2] + yl[:, :, 0::2, 1::2] - 
                  yl[:, :, 1::2, 0::2] - yl[:, :, 1::2, 1::2]) / 2
            
            # Low-High (Vertical Detail)
            hl = (yl[:, :, 0::2, 0::2] - yl[:, :, 0::2, 1::2] + 
                  yl[:, :, 1::2, 0::2] - yl[:, :, 1::2, 1::2]) / 2
            
            # High-High (Diagonal Detail)
            hh = (yl[:, :, 0::2, 0::2] - yl[:, :, 0::2, 1::2] - 
                  yl[:, :, 1::2, 0::2] + yl[:, :, 1::2, 1::2]) / 2
            
            yl = ll
            
            # Stack high frequency components: [B, C, 3, H/2, W/2]
            # Order: LH, HL, HH (Horizontal, Vertical, Diagonal)
            # Match pytorch_wavelets output format
            # pytorch_wavelets returns yH as a LIST of tensors (one per level)
            current_yh = torch.stack((lh, hl, hh), dim=2)
            yh.append(current_yh)
            
        return yl, yh
#==================================================================================================================================

class LearnableDirectionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced = max(in_channels // reduction, 1)
        
        # 固定水平和垂直核
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, reduced, (1,5), padding=(0,2)),
            nn.ReLU(),
            nn.Conv2d(reduced, in_channels, (1,5), padding=(0,2))
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels, reduced, (5,1), padding=(2,0)),
            nn.ReLU(),
            nn.Conv2d(reduced, in_channels, (5,1), padding=(2,0))
        )
        
        # 可学习权重（softmax 归一化）
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0]))  # 初始等权

    def forward(self, x):
        attn_h = torch.sigmoid(self.conv_h(x))
        attn_v = torch.sigmoid(self.conv_v(x))
        
        # softmax 归一化权重
        w = F.softmax(self.weights, dim=0)
        
        attn = w[0] * attn_h + w[1] * attn_v
        return attn * x

class WaveletAttentionModule(nn.Module):
    def __init__(self, in_channels, wave='haar', reduction=16, save_dir='attention_maps'):
        super(WaveletAttentionModule, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)
        self.in_channels = in_channels
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # 确保通道数至少为1
        reduced_channels = max(in_channels // reduction, 1)

        # 条状卷积用于水平和垂直分量
        self.horizontal_attention = LearnableDirectionAttention(in_channels, reduction)
        self.vertical_attention = LearnableDirectionAttention(in_channels, reduction)

        # 对角线分量的常规卷积
        self.diagonal_attention = self.create_diagonal_attention(in_channels, reduction)

        # 低频分量的全局通道注意力机制
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def create_diagonal_attention(self, in_channels, reduction):
        """对角线高频分量的标准卷积"""
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),  # 标准卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 小波变换
        yL, yH = self.dwt(x)

        # 处理低频分量
        low_freq = yL
        low_freq_attn = self.low_freq_attention(low_freq)
        low_freq = low_freq * low_freq_attn

        # 处理高频分量
        high_freq_h = yH[0][:, :, 0, :, :]  # 水平分量
        high_freq_v = yH[0][:, :, 1, :, :]  # 垂直分量
        high_freq_d = yH[0][:, :, 2, :, :]  # 对角线分量

        # 应用不同的注意力机制
        high_freq_h = high_freq_h * self.horizontal_attention(high_freq_h)
        # high_freq_v = high_freq_v * self.vertical_attention(high_freq_v)
        high_freq_d = high_freq_d * self.diagonal_attention(high_freq_d)

        # 合并高频分量
        high_freq = torch.cat([high_freq_h, high_freq_v, high_freq_d], dim=1)

        # 特征融合，低频 + 高频
        combined = torch.cat([low_freq, high_freq], dim=1)
        out = self.fusion_conv(combined)

        return out

#==============================================================================================================
class RainDropFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RainDropFeatureExtractor, self).__init__()
        self.wavelet_attention = WaveletAttentionModule(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # Apply Wavelet Attention Module
        attention_out = self.wavelet_attention(x)
        # Adjust channel dimensions
        conv_out = self.conv(attention_out)
        # Residual connection
        out = conv_out
        return out, attention_out
#====================================================================================================================================
def bilinear_interpolate(input, grid):
    # grid 的范围是 [-1, 1]，这里我们假设输入是 BxCxHxW 的形状
    return F.grid_sample(input, grid, align_corners=True)

# 升级后的 Deformable ConvTranspose 实现，带残差连接
# ======================== 正确的 MS-DSB 实现 ========================
# 替换原来的 DeformConvTranspose2d 类即可
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(DeformConvTranspose2d, self).__init__()
        if out_channels is None:
            out_channels = in_channels // 2

        # 1. 普通上采样（PixelShuffle 或 ConvTranspose 均可，这里用 ConvTranspose）
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False
        )
        nn.init.kaiming_normal_(self.upsample.weight, mode='fan_out', nonlinearity='relu')

        # 2. 多尺度全局 offset 生成器（论文中 "Multi-Scale Offset Map Generator"）
        self.offset_conv3 = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=True)
        self.offset_conv5 = nn.Conv2d(in_channels, 2, kernel_size=5, padding=2, bias=True)
        # self.offset_conv7 = nn.Conv2d(in_channels, 2, kernel_size=7, padding=3, bias=True)
        self.offset_fuse  = nn.Conv2d(4, 2, kernel_size=1, bias=True)   # 融合 3×3 + 5×5 + 7×7

        # 3. 归一化 + 激活
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 初始化 offset 相关层为 0（避免训练初期偏移过大）
        nn.init.constant_(self.offset_conv3.weight, 0.)
        nn.init.constant_(self.offset_conv3.bias, 0.)
        nn.init.constant_(self.offset_conv5.weight, 0.)
        nn.init.constant_(self.offset_conv5.bias, 0.)
        # nn.init.constant_(self.offset_conv7.weight, 0.)
        # nn.init.constant_(self.offset_conv7.bias, 0.)
        nn.init.constant_(self.offset_fuse.weight, 0.)
        nn.init.constant_(self.offset_fuse.bias, 0.)

    def forward(self, x):
        # Step 1: 普通上采样
        up = self.upsample(x)                      # [B, C_out, 2H, 2W]

        # Step 2: 生成多尺度全局 offset
        o1 = self.offset_conv3(x)
        o2 = self.offset_conv5(x)
        # o3 = self.offset_conv7(x)
        offset = torch.cat([o1, o2], dim=1)          # [B,6,H,W]
        offset = F.interpolate(offset, size=up.shape[2:], mode='bilinear', align_corners=True)  # [B,6,2H,2W]
        offset = self.offset_fuse(offset)
        offset = torch.tanh(offset) * 1.0           # 控制偏移幅度在 ±1 像素以内（可调）

        # Step 3: 生成标准网格 + 添加偏移
        B, _, H, W = up.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).float()           # [H,W,2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)                 # [B,H,W,2]

        offset = offset.permute(0, 2, 3, 1)                            # [B,H,W,2]
        grid = grid + offset

        # 归一化到 [-1, 1] 以供 grid_sample 使用
        grid = grid / torch.tensor([W-1, H-1], dtype=torch.float32, device=x.device) * 2 - 1

        # Step 4: 真正的可变形采样
        aligned = F.grid_sample(up, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Step 5: 残差 + 激活
        out = self.bn(aligned)
        out = self.relu(out + up)   # 残差连接（可要可不要，建议保留）

        return out

#==============================================================================================================
class AdaptivePConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, partial_ratio=0.25, stride=1, padding=1, bias=True):
        super(AdaptivePConv, self).__init__()
        
        self.in_channels = in_channels
        self.partial_channels = int(in_channels * partial_ratio)
        
        # 卷积层
        self.conv = nn.Conv2d(self.partial_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding, 
                             bias=bias)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算每个通道的活跃度 (使用熵作为度量)
        # 对每个通道计算熵
        eps = 1e-8
        # 将每个通道的值归一化到 [0, 1] 并转换为概率分布
        x_flat = x.view(batch_size, self.in_channels, -1)  # [B, C, H*W]
        x_min = x_flat.min(dim=2, keepdim=True)[0]  # [B, C, 1]
        x_max = x_flat.max(dim=2, keepdim=True)[0]  # [B, C, 1]
        x_range = x_max - x_min + eps
        x_norm = (x_flat - x_min) / x_range  # [B, C, H*W]
        
        # 计算直方图（将值分成 bins，使用向量化方法）
        num_bins = 256
        # 将归一化的值映射到 bin 索引 [0, num_bins-1]
        bin_indices = (x_norm * num_bins).long().clamp(0, num_bins - 1)  # [B, C, H*W]
        
        # 使用 scatter_add 计算直方图
        hist = torch.zeros(batch_size, self.in_channels, num_bins, device=x.device)
        hist.scatter_add_(2, bin_indices, torch.ones_like(x_norm))
        
        # 归一化为概率分布
        hist = hist + eps  # 避免 log(0)
        prob = hist / hist.sum(dim=2, keepdim=True)  # [B, C, num_bins]
        
        # 计算熵: -sum(p * log(p))
        channel_activity = -(prob * torch.log(prob + eps)).sum(dim=2)  # [B, C]
        
        # 选择活跃度最高的 partial_channels 个通道的索引
        _, top_indices = torch.topk(channel_activity, self.partial_channels, dim=1)
        
        # 为每个批次样本收集重要通道
        partial_outputs = []
        untouched_outputs = []
        
        for i in range(batch_size):
            # 获取当前样本的活跃通道
            selected_channels = x[i, top_indices[i], :, :]
            selected_channels = selected_channels.unsqueeze(0)
            
            # 对选中的通道进行卷积
            partial_out = self.conv(selected_channels)
            partial_outputs.append(partial_out)
            
            # 获取未选中的通道
            mask = torch.ones(self.in_channels, device=x.device)
            mask[top_indices[i]] = 0
            unselected_indices = mask.nonzero().squeeze()
            untouched_channels = x[i, unselected_indices, :, :]
            untouched_channels = untouched_channels.unsqueeze(0)
            
            untouched_outputs.append(untouched_channels)
        
        # 将所有批次的输出组合
        partial_output = torch.cat(partial_outputs, dim=0)
        untouched_output = torch.cat(untouched_outputs, dim=0)
        
        # 拼接处理过的和未处理的通道
        out = torch.cat((partial_output, untouched_output), dim=1)
        
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        
        # 全连接层
        self.fc1 = nn.Linear(in_features=in_channels, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=in_channels)
        
        # Dropout 和激活函数
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # LayerNorm
        self.norm = nn.LayerNorm(in_channels)
        
        # 自适应部分卷积层
        self.gated_conv = AdaptivePConv(in_channels=hidden_dim, out_channels=hidden_dim//4)

    def forward(self, x):
        # 获取输入张量的形状
        b, c, h, w = x.shape

        # 展平空间维度并交换轴，使之适应通道的 MLP 操作
        x = x.view(b, c, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 保存残差
        residual = x
        
        # 第一层 Linear 变换
        x = self.fc1(x)
        
        # 恢复空间维度以便使用门控卷积
        x = x.transpose(1, 2).view(b, -1, h, w)  # [B, hidden_dim, H, W]
        # 使用门控卷积进行稀疏化
        x = self.gated_conv(x)
        
        # 再次展平空间维度
        x = x.view(b, -1, h * w).transpose(1, 2)  # [B, H*W, hidden_dim]
        
        # 激活、Dropout
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层 Linear 变换
        x = self.fc2(x)
        
        # 激活、Dropout
        x = self.relu(x)
        x = self.dropout(x)
        
        # 残差连接 + 归一化
        x = self.norm(x + residual)

        # 恢复为原始形状 [B, C, H, W]
        x = x.transpose(1, 2).view(b, c, h, w)
        
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积：在每个通道上单独应用卷积，不跨通道
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, bias=bias)
        # 逐点卷积：1x1 卷积用于通道之间的线性组合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        # 逐点卷积
        x = self.pointwise(x)
        return x
#==============================================================================================================
class WDNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(WDNet, self).__init__()

        # Encoder
        self.encoder1 = RainDropFeatureExtractor(in_channels, 64)
        self.encoder2 = RainDropFeatureExtractor(64, 128)
        self.encoder3 = RainDropFeatureExtractor(128, 256)
        self.encoder4 = RainDropFeatureExtractor(256, 512)

        # Middle part
        
        self.middle = nn.Sequential(
            DepthwiseSeparableConv(in_channels=512, out_channels=1024),
            FeedForwardBlock(1024, 512),
            DepthwiseSeparableConv(in_channels=1024, out_channels=512),
        )
        
        # Decoder
        self.upconv4 = DeformConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = self.double_conv(256 + 256, 256)
        self.upconv3 = DeformConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = self.double_conv(128 + 128, 128)
        self.upconv2 = DeformConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = self.double_conv(64 + 64, 64)
        self.upconv1 = DeformConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = self.double_conv(32 + 3, 64)
        self.up = DeformConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        res = x
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        # Middle part
        x5 = self.middle(self.maxpool(x4))
        # Decoder
        x = self.upconv4(x5)
        x = F.interpolate(x, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder4(x)
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder1(x)
        # Output layer
        x = self.up(x)
        x = self.out_conv(x)
        x = x + res
        x = self.final_conv(x)
        return x

    def maxpool(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

#==============================================================================================================
# 测试 U-Net 模型
if __name__ == '__main__':
    model = WDNet(in_channels=3, out_channels=3)  # 输入通道数为 1，输出通道数为 1
    input_tensor = torch.rand(1, 3, 480, 720)  # 示例输入：batch_size=1, channels=1, height=256, width=256
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # 输出张量的尺寸
