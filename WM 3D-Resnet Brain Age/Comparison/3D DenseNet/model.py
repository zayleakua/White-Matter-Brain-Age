"""
3D DenseNet Model for Predicting White Matter Age

@author: Puzhen & Ruijia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) Basic 3D conv: Conv3d + BN + ReLU
# ---------------------------
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=3, stride=1, padding=1):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ----------------------------
# 2) DenseLayer (simplified)
# ----------------------------
class DenseLayer3D(nn.Module):
    """
    Simplified DenseLayer.
    In classic DenseNet, a DenseLayer is BN->ReLU->1x1Conv -> BN->ReLU->3x3Conv.
    For simplicity, we only use a single 3x3 conv here (i.e., omit the 1x1 bottleneck).
    
    Input channels: in_channels
    Output channels: in_channels + growth_rate
    """
    def __init__(self, in_channels, growth_rate=8):
        super(DenseLayer3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        """
        x: (B, in_channels, D, H, W)
        return: (B, in_channels + growth_rate, D, H, W)
        """
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)

# ----------------------------
# 3) DenseBlock (multiple DenseLayer)
# ----------------------------
class DenseBlock3D(nn.Module):
    """
    A stack of num_layers DenseLayer3D blocks; each layer's output is concatenated.
    Final channels = in_channels + num_layers * growth_rate
    """
    def __init__(self, in_channels, growth_rate=8, num_layers=2):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        for i in range(num_layers):
            self.layers.append(DenseLayer3D(curr_channels, growth_rate))
            curr_channels += growth_rate
        self.out_channels = curr_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ------------------------------
# 4) 3D Transition (simplified)
# ------------------------------
class Transition3D(nn.Module):
    """
    Transition layer does:
      1) Channel compression via 1x1 conv
      2) Spatial downsampling via AvgPool3d(kernel_size=2, stride=2)
    """
    def __init__(self, in_channels, out_channels):
        super(Transition3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))

# ------------------------------
# 5) Simplified DenseNet (2 blocks)
# ------------------------------
class Simple3DDenseNet(nn.Module):
    """
    Input:  (B, 3, 64, 64, 64)
    Output: (B, 512, 3, 3, 3)

    Components:
      - Initial conv (stride=2), channels=16
      - Dense Block1 (2 layers), out=32
      - Transition1 (channels 32->16, spatial 32->16)
      - Dense Block2 (2 layers), out=32
      - Transition2 (channels 32->16, spatial 16->8)
      - Final conv (channels 16->512), spatial 8->8
      - AdaptiveAvgPool3d to (3,3,3)
    """
    def __init__(self, growth_rate=8, num_layers_per_block=2):
        super(Simple3DDenseNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = DenseBlock3D(16, growth_rate, num_layers_per_block)
        self.trans1 = Transition3D(self.block1.out_channels, 16)

        self.block2 = DenseBlock3D(16, growth_rate, num_layers_per_block)
        self.trans2 = Transition3D(self.block2.out_channels, 16)

        self.conv2 = nn.Conv3d(16, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(512)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((3,3,3))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return self.adaptive_pool(x)

# ------------------------------
# 6) Age prediction head
# ------------------------------
class AgePredictionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AgePredictionNet, self).__init__()
        # Use simplified 3D DenseNet as the feature extractor
        self.feature_extractor = Simple3DDenseNet(growth_rate=8, num_layers_per_block=2)
        self.flattened_size = 512 * 3 * 3 * 3  # 13824
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128 + 1, 1)  # Concatenate sex_feature => +1
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sex_feature):
        """
        x: (B,3,64,64,64)
        sex_feature: (B,) tensor representing sex feature
        """
        x = self.feature_extractor(x)         # -> (B, 512, 3, 3, 3)
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        sex_feature = sex_feature.unsqueeze(1)
        x = torch.cat((x, sex_feature), dim=1)
        return F.relu(self.fc2(x))            # Final regression
