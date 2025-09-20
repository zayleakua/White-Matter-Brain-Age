import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Conv3d block
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Inception3D block
class Inception3D(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, out_pool):
        super(Inception3D, self).__init__()
        self.branch1x1 = BasicConv3d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3_1 = BasicConv3d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = BasicConv3d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        self.branch_pool_proj = BasicConv3d(in_channels, out_pool, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_2(self.branch3x3_1(x))
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_proj(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch_pool], dim=1)

# Feature extractor with 2 Inception blocks
class SimpleInception3DNet(nn.Module):
    def __init__(self):
        super(SimpleInception3DNet, self).__init__()
        self.conv1 = BasicConv3d(3, 32, kernel_size=3, stride=2, padding=1)
        self.inception1 = Inception3D(32, 16, 16, 32, 16)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.inception2 = Inception3D(64, 32, 32, 64, 32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = BasicConv3d(128, 512, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3,3,3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.maxpool1(x)
        x = self.inception2(x)
        x = self.maxpool2(x)
        x = self.conv2(x)
        return self.adaptive_pool(x)

# Age prediction net
class AgePredictionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AgePredictionNet, self).__init__()
        self.feature_extractor = SimpleInception3DNet()
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
        return F.relu(self.fc2(x))
