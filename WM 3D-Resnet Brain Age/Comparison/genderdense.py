"""
3D DenseNet dMRI (FA/MD/MO) Pipeline for White Matter Brain Age 

@author: Puzhen & Ruijia
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import random
import os  # For file path operations
import numpy as np  # For numerical computing and array operations
import nibabel as nib  # For loading NIfTI-format medical images
import torch  # Core PyTorch library
from torch.utils.data import Dataset  # For custom dataset classes
import random  # For random number generation
from scipy.ndimage import rotate  # For image rotation
from torch.optim.lr_scheduler import StepLR



# Load CSV file with CCID, age, and sex labels
csv_file = 'data3.csv'
df = pd.read_csv(csv_file)
df = df[['CCID', 'Age', 'Sex']]


# Base directory containing per-CCID subfolders with images
image_base_dir = 'output'


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_base_dir, transform=None, augment=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'CCID', 'Age', and 'Sex'.
            image_base_dir (str): Path to the base directory containing image folders (one per CCID).
            transform (callable, optional): Optional transform applied to a sample (unused here).
            augment (bool, optional): If True, apply data augmentation (default: True).
        """
        self.dataframe = dataframe
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.augment = augment  # Enable augmentation by default

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Fetch CCID, sex, and ground-truth age
        ccid = self.dataframe.iloc[idx]['CCID']
        sex = self.dataframe.iloc[idx]['Sex']  # Binary feature: 0 or 1
        real_age = self.dataframe.iloc[idx]['Age']  # Ground-truth age

        # Load three NIfTI volumes for the current CCID
        ccid_folder = os.path.join(self.image_base_dir, ccid)
        img_files = ['FA.nii', 'MD.nii', 'MO.nii']

        # Stack the three volumes as channels
        images = []
        for img_file in img_files:
            img_path = os.path.join(ccid_folder, img_file)
            img = nib.load(img_path).get_fdata()
            images.append(img)

        images = np.stack(images, axis=0)  # Shape: (3, H, W, D)

        # Per-channel normalization
        images = self._normalize_images(images)

        # Optional data augmentation
        if self.augment:
            images = self._augment_images(images)

        # Ensure memory is contiguous (avoid negative strides after flips)
        images = images.copy()

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(sex, dtype=torch.float32),
            torch.tensor(real_age, dtype=torch.float32)
        )

    def _normalize_images(self, images):
        """
        Normalize each channel to zero mean and unit variance.
        """
        means = images.mean(axis=(1, 2, 3), keepdims=True)  # Per-channel mean
        stds = images.std(axis=(1, 2, 3), keepdims=True)  # Per-channel std
        normalized = (images - means) / (stds + 1e-8)  # Numerical stability
        return normalized

    def _augment_images(self, images):
        """
        Apply random 3D augmentations: flips, rotation, and random crop.
        """
        # Random flips along spatial axes
        if random.random() > 0.5:
            images = np.flip(images, axis=1)  # Flip along height
        if random.random() > 0.5:
            images = np.flip(images, axis=2)  # Flip along width
    
        # Random in-plane rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)  # Degrees
            for i in range(images.shape[0]):
                images[i] = rotate(images[i], angle, axes=(1, 2), reshape=False, mode='nearest')
    
        # Random cropping
        crop_size = 100  # Target crop size for each spatial dimension
        h, w, d = images.shape[1:]
        crop_size_h = min(crop_size, h)
        crop_size_w = min(crop_size, w)
        crop_size_d = min(crop_size, d)
    
        start_h = random.randint(0, h - crop_size_h) if h > crop_size_h else 0
        start_w = random.randint(0, w - crop_size_w) if w > crop_size_w else 0
        start_d = random.randint(0, d - crop_size_d) if d > crop_size_d else 0
    
        images = images[:, start_h:start_h + crop_size_h, start_w:start_w + crop_size_w, start_d:start_d + crop_size_d]
    
        return images




# Split into train/validation subsets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Default: normalization + augmentation for training; augmentation disabled for validation
train_dataset = ImageDataset(train_df, image_base_dir)
valid_dataset = ImageDataset(valid_df, image_base_dir, augment=False)  # Disable augmentation for validation

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)


import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

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
        self.conv = nn.Conv3d(
            in_channels, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x):
        """
        x: (B, in_channels, D, H, W)
        return: (B, in_channels + growth_rate, D, H, W)
        """
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        # DenseNet core: concatenate input and new features along channels
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
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        
        for i in range(num_layers):
            layer = DenseLayer3D(curr_channels, growth_rate)
            self.layers.append(layer)
            curr_channels += growth_rate  # Each layer adds growth_rate channels
        
        self.out_channels = curr_channels  # Final number of channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # (B, in_channels + num_layers*growth_rate, D, H, W)

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
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1, stride=1, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x  # (B, out_channels, D/2, H/2, W/2)


class Simple3DDenseNet(nn.Module):
    """
    Input:  (B, 3, 64, 64, 64)
    Output: (B, 512, 3, 3, 3)

    Components:
      - Initial conv (stride=2), channels=16
      - Dense Block1 (2 layers), out=32
      - Transition1 (channels 32->16, spatial 32->16)
      - Dense Block2 (2 layers), out=16 + 2*8 = 32
      - Transition2 (channels 32->16, spatial 16->8)
      - Final conv (channels 16->512), spatial 8->8
      - AdaptiveAvgPool3d to (3,3,3)
    """
    def __init__(self, growth_rate=8, num_layers_per_block=2):
        super(Simple3DDenseNet, self).__init__()
        
        # 1) Initial conv: channels 3->16, spatial 64->32
        self.conv1 = nn.Conv3d(
            3, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 2) DenseBlock #1
        self.block1 = DenseBlock3D(
            in_channels=16,
            growth_rate=growth_rate,
            num_layers=num_layers_per_block
        )
        block1_out = self.block1.out_channels  # 16 + 2*8 = 32
        
        # 3) Transition #1: channels 32->16, spatial 32->16
        self.trans1 = Transition3D(
            in_channels=block1_out, 
            out_channels=16
        )
        
        # 4) DenseBlock #2
        self.block2 = DenseBlock3D(
            in_channels=16,
            growth_rate=growth_rate,
            num_layers=num_layers_per_block
        )
        block2_out = self.block2.out_channels  # 16 + 2*8 = 32
        
        # 5) Transition #2: channels 32->16, spatial 16->8
        self.trans2 = Transition3D(
            in_channels=block2_out, 
            out_channels=16
        )
        
        # 6) Final conv: channels 16->512 (spatial stays 8->8)
        self.conv2 = nn.Conv3d(
            16, 512, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(512)
        
        # 7) Adaptive average pooling to (3,3,3)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3,3,3))
        
    def forward(self, x):
        # x: (B,3,64,64,64)
        # 1) Initial conv
        x = self.conv1(x)  # -> (B,16,32,32,32)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 2) DenseBlock #1
        x = self.block1(x) # -> (B,32,32,32,32)
        
        # 3) Transition #1
        x = self.trans1(x) # -> (B,16,16,16,16)
        
        # 4) DenseBlock #2
        x = self.block2(x) # -> (B,32,16,16,16)
        
        # 5) Transition #2
        x = self.trans2(x) # -> (B,16,8,8,8)
        
        # 6) Final conv
        x = self.conv2(x)  # -> (B,512,8,8,8)
        x = self.bn2(x)
        x = self.relu(x)
        
        # 7) Adaptive pooling to (3,3,3)
        x = self.adaptive_pool(x) # -> (B,512,3,3,3)
        return x


class AgePredictionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AgePredictionNet, self).__init__()
        # Use simplified 3D DenseNet as the feature extractor
        self.feature_extractor = Simple3DDenseNet(
            growth_rate=8, 
            num_layers_per_block=2
        )
        
        # Output is fixed as (B,512,3,3,3)
        self.flattened_size = 512 * 3 * 3 * 3  # 512 * 27 = 13824
        
        # Fully connected head
        self.fc1 = nn.Linear(self.flattened_size, 128)
        # Concatenate sex_feature => +1
        self.fc2 = nn.Linear(128 + 1, 1)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sex_feature):
        """
        x: (B,3,64,64,64)
        sex_feature: (B,) tensor representing sex feature
        """
        # 1) Extract 3D features
        x = self.feature_extractor(x)  # -> (B, 512, 3, 3, 3)
        
        # 2) Flatten
        x = x.view(x.size(0), -1)      # (B, 13824)
        
        # 3) Fully connected + concat sex feature
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        sex_feature = sex_feature.unsqueeze(1)  # (B,1)
        x = torch.cat((x, sex_feature), dim=1)  # (B, 129)
        
        # 4) Final regression
        x = F.relu(self.fc2(x))  # (B,1)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMSEMAELossWithConstraint(nn.Module):
    def __init__(self, alpha=0.5, min_age=0, max_age=120, penalty=1.0):
        """
        Combined loss with MSE + MAE and a range constraint on predictions.

        Args:
            alpha (float): Weight between MSE and MAE; larger alpha emphasizes MSE.
            min_age (float): Minimum valid age (lower bound for predictions).
            max_age (float): Maximum valid age (upper bound for predictions).
            penalty (float): Penalty weight for out-of-range predictions.
        """
        super(CombinedMSEMAELossWithConstraint, self).__init__()
        self.alpha = alpha
        self.min_age = min_age
        self.max_age = max_age
        self.penalty = penalty

    def forward(self, predictions, targets):
        # Base loss: weighted MSE + MAE
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        base_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss

        # Penalty for predictions outside [min_age, max_age]
        lower_bound_penalty = torch.relu(self.min_age - predictions)
        upper_bound_penalty = torch.relu(predictions - self.max_age)
        penalty_loss = torch.mean(lower_bound_penalty + upper_bound_penalty)

        # Total loss
        total_loss = base_loss + self.penalty * penalty_loss

        return total_loss




# Evaluate on validation set and compute MAE
def evaluate_model(valid_loader, model, criterion):
    """
    Evaluate model performance using CombinedMSEMAELossWithConstraint.

    Args:
        valid_loader (DataLoader): Validation DataLoader.
        model (torch.nn.Module): Model to evaluate.
        criterion (nn.Module): CombinedMSEMAELossWithConstraint.

    Returns:
        tuple: (average validation loss, MAE).
    """
    model.eval()  # Evaluation mode
    val_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(valid_loader, desc="Validation")):
            inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)

            # Forward pass
            outputs = model(inputs, sex_labels)

            # Compute loss
            loss = criterion(outputs, real_ages.unsqueeze(1))
            val_loss += loss.item()

            # Accumulate labels and predictions
            all_labels.extend(real_ages.tolist())
            all_predictions.extend(outputs.squeeze(1).tolist())

    # Compute MAE
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"Validation MAE: {mae:.4f}")

    return val_loss / len(valid_loader), mae


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42) 

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgePredictionNet().to(device)
criterion = CombinedMSEMAELossWithConstraint(alpha=0.5, min_age=0, max_age=120, penalty=1.0)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50)


# Track the best validation loss
best_val_loss = float('inf')  # Initialize with a large value

# Prepare log file
log_file = 'experiments/genderdense.txt'
with open(log_file, 'w') as f:
    f.write("Epoch,Training Loss,Validation Loss,Training MAE,Validation MAE\n")

# Training loop with validation, checkpointing, and MAE reporting
for epoch in range(5000):  # Number of epochs
    model.train()  # Training mode
    running_loss = 0.0
    all_labels_train = []
    all_predictions_train = []

    # Per-batch training with tqdm progress bar
    for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
        inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs, sex_labels)  # Forward pass (model takes sex_labels)
        loss = criterion(outputs, real_ages.unsqueeze(1))  # Compute loss against real ages
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect labels and predictions for MAE calculation
        all_labels_train.extend(real_ages.tolist())
        all_predictions_train.extend(outputs.squeeze(1).tolist())

    # Training MAE for this epoch
    mae_train = mean_absolute_error(all_labels_train, all_predictions_train)
    print(f"Epoch {epoch + 1}, Training MAE: {mae_train:.4f}")

    # Validation evaluation
    val_loss, mae_val = evaluate_model(valid_loader, model, criterion)

    # Save checkpoint if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'experiments/genderdense.pth')
        print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

    # Append metrics to log file
    with open(log_file, 'a') as f:
        f.write(f"{epoch + 1},{running_loss / len(train_loader):.4f},{val_loss:.4f},{mae_train:.4f},{mae_val:.4f}\n")

    print(
        f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {mae_val:.4f}')

print('Finished Training')
