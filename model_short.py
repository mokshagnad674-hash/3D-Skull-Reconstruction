import torch
import torch.nn as nn

class AttentionUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._block(1, 16)
        self.enc2 = self._block(16, 32)
        self.enc3 = self._block(32, 64)
        self.mid = self._block(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec3 = self._block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec2 = self._block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.dec1 = self._block(32, 16)
        self.out = nn.Conv3d(16, 1, 1)
        
        self.pool = nn.MaxPool3d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(out_ch, out_ch//8),
            nn.ReLU(),
            nn.Linear(out_ch//8, out_ch),
            nn.Sigmoid(),
            nn.Unflatten(1, (out_ch, 1, 1, 1))
        )

    def forward(self, x):
        # Encoder
        d1 = self.enc1(x)
        d2 = self.enc2(self.pool(d1))
        d3 = self.enc3(self.pool(d2))
        m = self.mid(self.pool(d3))
        
        # Decoder
        u3 = self.dec3(torch.cat([self.up3(m), d3], 1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], 1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], 1))
        
        return torch.sigmoid(self.out(u1))