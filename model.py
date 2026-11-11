import torch
import torch.nn as nn


class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )
        self.attn = ChannelAttention3D(out_ch)

    def forward(self, x):
        return self.attn(self.conv(x))


class AttentionUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv3D(1, 16)
        self.pool = nn.MaxPool3d(2)
        self.down2 = DoubleConv3D(16, 32)
        self.down3 = DoubleConv3D(32, 64)
        self.mid = DoubleConv3D(64, 128)
        self.up3 = nn.ConvTranspose3d(128,64,2,2)
        self.conv3 = DoubleConv3D(128,64)
        self.up2 = nn.ConvTranspose3d(64,32,2,2)
        self.conv2 = DoubleConv3D(64,32)
        self.up1 = nn.ConvTranspose3d(32,16,2,2)
        self.conv1 = DoubleConv3D(32,16)
        self.out = nn.Conv3d(16,1,1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        m = self.mid(self.pool(d3))
        u3 = self.up3(m)
        u3 = self.conv3(torch.cat([u3, d3],1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2],1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1],1))
        return torch.sigmoid(self.out(u1))
