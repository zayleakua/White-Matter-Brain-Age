"""
3D ResNet Model for Predicting White Matter Age

@author: Puzhen & Ruijia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=23, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BasicBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class AgePredictionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AgePredictionNet, self).__init__()
        self.feature_extractor = VisualFeatureExtractor()
        self.flattened_size = 512 * 3 * 3 * 3
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128 + 1, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sex_feature):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        sex_feature = sex_feature.unsqueeze(1)
        x = torch.cat((x, sex_feature), dim=1)
        x = F.relu(self.fc2(x))
        return x
