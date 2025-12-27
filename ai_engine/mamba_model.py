import torch
import torch.nn as nn
import torch.nn.functional as F

# 可变形反卷积模块
class DeformConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(DeformConvTranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.offset_conv1 = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.offset_conv2 = nn.Conv2d(in_channels, 2, kernel_size=5, padding=2)
        self.offset_fuse = nn.Conv2d(4, 2, kernel_size=1)
        self._init_weights()
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_transpose.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_fuse.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        offset1 = self.offset_conv1(x)
        offset2 = self.offset_conv2(x)
        offset = torch.cat([offset1, offset2], dim=1)
        offset = torch.tanh(self.offset_fuse(offset))
        out = self.conv_transpose(x)
        res = out
        n, c, h, w = out.shape
        offset = F.interpolate(offset, size=(h, w), mode='bilinear', align_corners=True)
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        offset = offset.permute(0, 2, 3, 1)
        grid = grid + offset
        grid = (grid / torch.tensor([w-1, h-1], device=x.device) - 0.5) * 2
        out = F.grid_sample(res, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        out = out + res
        return out

class HaarDWT(nn.Module):
    def __init__(self):
        super(HaarDWT, self).__init__()
        self.LL_filter = nn.Parameter(torch.tensor([[0.25, 0.25], [0.25, 0.25]], dtype=torch.float32).view(1, 1, 2, 2), requires_grad=False)
        self.LH_filter = nn.Parameter(torch.tensor([[-0.25, -0.25], [0.25, 0.25]], dtype=torch.float32).view(1, 1, 2, 2), requires_grad=False)
        self.HL_filter = nn.Parameter(torch.tensor([[-0.25, 0.25], [-0.25, 0.25]], dtype=torch.float32).view(1, 1, 2, 2), requires_grad=False)
        self.HH_filter = nn.Parameter(torch.tensor([[0.25, -0.25], [-0.25, 0.25]], dtype=torch.float32).view(1, 1, 2, 2), requires_grad=False)

    def forward(self, x):
        LL = F.conv2d(x, self.LL_filter.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        LH = F.conv2d(x, self.LH_filter.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        HL = F.conv2d(x, self.HL_filter.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        HH = F.conv2d(x, self.HH_filter.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        return LL, LH, HL, HH

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class WaveletAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(WaveletAttentionModule, self).__init__()
        self.dwt = HaarDWT()
        self.in_channels = in_channels
        reduced_channels = max(in_channels // reduction, 1)
        self.horizontal_attention = self.create_horizontal_attention(in_channels, reduction)
        self.vertical_attention = self.create_vertical_attention(in_channels, reduction)
        self.diagonal_attention = self.create_diagonal_attention(in_channels, reduction)
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def create_horizontal_attention(self, in_channels, reduction):
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),
            nn.Sigmoid()
        )

    def create_vertical_attention(self, in_channels, reduction):
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.Sigmoid()
        )

    def create_diagonal_attention(self, in_channels, reduction):
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        low_freq = LL
        high_freq_h = LH
        high_freq_v = HL
        high_freq_d = HH

        low_freq_attn = self.low_freq_attention(low_freq)
        low_freq = low_freq * low_freq_attn
        high_freq_h = high_freq_h * self.horizontal_attention(high_freq_h)
        high_freq_v = high_freq_v * self.vertical_attention(high_freq_v)
        high_freq_d = high_freq_d * self.diagonal_attention(high_freq_d)

        high_freq = torch.cat([high_freq_h, high_freq_v, high_freq_d], dim=1)
        combined = torch.cat([low_freq, high_freq], dim=1)
        out = self.fusion_conv(combined)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

class MambaBlock(nn.Module):
    # Removed spatial_size argument to support dynamic resolutions
    def __init__(self, channels):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        residual = x
        # Dynamic LayerNorm: Permute (N, C, H, W) -> (N, H, W, C) -> Norm -> Permute back
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1) # N, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # N, C, H, W
        x1 = self.conv1(x)
        g = torch.sigmoid(self.gate(x))
        x = x1 * g
        x = self.act(x)
        x = self.conv2(x)
        return x + residual

class SS2D(nn.Module):
    # Removed spatial_size argument
    def __init__(self, channels):
        super(SS2D, self).__init__()
        self.dw_conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dw_conv_w = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x):
        residual = x
        residual = x
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        h_feat = self.dw_conv_h(x)
        w_feat = self.dw_conv_w(x)
        x = h_feat + w_feat
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = self.act(x)
        x = self.pw_conv(x)
        return x + residual

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.mamba = MambaBlock(features[3] * 2)
        self.ss2d = SS2D(features[3] * 2)
        self.wavelet_attn = WaveletAttentionModule(features[3] * 2)
        # 替换上采样层为DeformConvTranspose2d
        self.upconv4 = DeformConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])
        self.upconv3 = DeformConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])
        self.upconv2 = DeformConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])
        self.upconv1 = DeformConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(self.pool(enc1_out))
        enc3_out = self.enc3(self.pool(enc2_out))
        enc4_out = self.enc4(self.pool(enc3_out))
        bottleneck_out = self.bottleneck(self.pool(enc4_out))
        mamba_out = self.mamba(bottleneck_out)
        ss2d_out = self.ss2d(mamba_out)
        wavelet_attn_out = self.wavelet_attn(ss2d_out)
        dec4_in = self.upconv4(wavelet_attn_out)
        dec4_in = torch.cat([dec4_in, enc4_out], dim=1)
        dec4_out = self.dec4(dec4_in)
        dec3_in = self.upconv3(dec4_out)
        dec3_in = torch.cat([dec3_in, enc3_out], dim=1)
        dec3_out = self.dec3(dec3_in)
        dec2_in = self.upconv2(dec3_out)
        dec2_in = torch.cat([dec2_in, enc2_out], dim=1)
        dec2_out = self.dec2(dec2_in)
        dec1_in = self.upconv1(dec2_out)
        dec1_in = torch.cat([dec1_in, enc1_out], dim=1)
        dec1_out = self.dec1(dec1_in)
        output = self.out_conv(dec1_out)
        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(in_channels=3, out_channels=3).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")