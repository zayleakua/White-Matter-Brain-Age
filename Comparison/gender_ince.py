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
import numpy as np  # For numerical computation and array operations
import nibabel as nib  # For loading NIfTI format medical images
import torch  # Core PyTorch library
from torch.utils.data import Dataset  # For creating custom dataset class
import random  # For generating random numbers
from scipy.ndimage import rotate  # For performing image rotation
from torch.optim.lr_scheduler import StepLR



# Load CSV file with CCID and age labels
csv_file = 'data3.csv'
df = pd.read_csv(csv_file)
df = df[['CCID', 'Age', 'Sex']]


# Set the folder containing subfolders named after CCIDs
image_base_dir = 'output'


class ImageDataset(Dataset):
    def __init__(self, dataframe, image_base_dir, transform=None, augment=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'CCID' and 'Age'.
            image_base_dir (str): Path to the base directory containing image folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool, optional): If True, apply data augmentation (default: True).
        """
        self.dataframe = dataframe
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.augment = augment  # Default augment to True

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get CCID, sex, and age
        ccid = self.dataframe.iloc[idx]['CCID']
        sex = self.dataframe.iloc[idx]['Sex']  # Binary feature: 0 or 1
        real_age = self.dataframe.iloc[idx]['Age']  # Ground-truth age

        # Load the 3 NIfTI images from the corresponding CCID folder
        ccid_folder = os.path.join(self.image_base_dir, ccid)
        img_files = ['FA.nii', 'MD.nii', 'MO.nii']

        # Stack the 3 images along the channel dimension
        images = []
        for img_file in img_files:
            img_path = os.path.join(ccid_folder, img_file)
            img = nib.load(img_path).get_fdata()
            images.append(img)

        images = np.stack(images, axis=0)  # Shape: (3, H, W, D)

        # Normalize the images
        images = self._normalize_images(images)

        # Apply data augmentation
        if self.augment:
            images = self._augment_images(images)

        # Ensure no negative strides by making a copy
        images = images.copy()

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(sex, dtype=torch.float32),
            torch.tensor(real_age, dtype=torch.float32)
        )



    def _normalize_images(self, images):
        """
        Normalize each channel individually to have zero mean and unit variance.
        """
        means = images.mean(axis=(1, 2, 3), keepdims=True)  # Per channel mean
        stds = images.std(axis=(1, 2, 3), keepdims=True)  # Per channel std
        normalized = (images - means) / (stds + 1e-8)  # Avoid division by zero
        return normalized

    def _augment_images(self, images):
        """
        Apply random augmentations to the images.
        """
        # Random flipping along spatial axes
        if random.random() > 0.5:
            images = np.flip(images, axis=1)  # Flip along height
        if random.random() > 0.5:
            images = np.flip(images, axis=2)  # Flip along width
    
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)  # Rotate by a random angle between -15 and 15 degrees
            for i in range(images.shape[0]):
                images[i] = rotate(images[i], angle, axes=(1, 2), reshape=False, mode='nearest')
    
        # Random cropping
        crop_size = 100  # Set desired crop size
        h, w, d = images.shape[1:]
        # Ensure the crop size does not exceed the image dimensions
        crop_size_h = min(crop_size, h)
        crop_size_w = min(crop_size, w)
        crop_size_d = min(crop_size, d)
    
        start_h = random.randint(0, h - crop_size_h) if h > crop_size_h else 0
        start_w = random.randint(0, w - crop_size_w) if w > crop_size_w else 0
        start_d = random.randint(0, d - crop_size_d) if d > crop_size_d else 0
    
        images = images[:, start_h:start_h + crop_size_h, start_w:start_w + crop_size_w, start_d:start_d + crop_size_d]
    
        return images



# Split data into train and test sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Default behavior: both normalization and augmentation enabled
train_dataset = ImageDataset(train_df, image_base_dir)
valid_dataset = ImageDataset(valid_df, image_base_dir, augment=False)  # Disable augmentation for validation

# Create DataLoader
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Basic 3D convolutional module: Conv3d + BN + ReLU
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# 2. Simplified Inception3D module
class Inception3D(nn.Module):
    """
    Keep only three main branches:
      1) 1x1
      2) 1x1 -> 3x3
      3) 3x3 AvgPool -> 1x1 (pool projection)
    """
    def __init__(self, in_channels,
                 out_1x1,
                 reduce_3x3, out_3x3,
                 out_pool):
        super(Inception3D, self).__init__()
        # Branch 1: 1x1 conv
        self.branch1x1 = BasicConv3d(in_channels, out_1x1, kernel_size=1)

        # Branch 2: 1x1 -> 3x3
        self.branch3x3_1 = BasicConv3d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3_2 = BasicConv3d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        # Branch 3: 3x3 AvgPool -> 1x1
        self.branch_pool_proj = BasicConv3d(in_channels, out_pool, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        # 3x3 average pooling (stride=1, padding=1) keeps spatial size
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_proj(branch_pool)

        outputs = [branch1x1, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

# 3. Feature extractor with only two Inception blocks
class SimpleInception3DNet(nn.Module):
    """
    Input:  (B, 3, 64, 64, 64)
    Output: (B, 512, 3, 3, 3)

    Architecture:
      - Conv3D(stride=2) -> downsample (64->32)
      - Inception block #1
      - MaxPool3d -> downsample (32->16)
      - Inception block #2
      - MaxPool3d -> downsample (16->8)
      - Conv3D increase to 512 channels (8->8)
      - Adaptive average pooling to (3,3,3)
    """
    def __init__(self):
        super(SimpleInception3DNet, self).__init__()
        # 1) Initial conv: spatial 64 -> 32, channels 3 -> 32
        self.conv1 = BasicConv3d(3, 32, kernel_size=3, stride=2, padding=1)   # -> (B,32,32,32,32)

        # 2) Inception block #1
        self.inception1 = Inception3D(
            in_channels=32,
            out_1x1=16,
            reduce_3x3=16, out_3x3=32,
            out_pool=16
        )  # Output channels: 16+32+16=64
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)                          # -> size: 32->16

        # 3) Inception block #2
        self.inception2 = Inception3D(
            in_channels=64,
            out_1x1=32,
            reduce_3x3=32, out_3x3=64,
            out_pool=32
        )  # Output channels: 32+64+32=128
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)                          # -> size: 16->8

        # 4) Increase channels to 512
        #    Current channels=128, spatial size=8,8,8
        self.conv2 = BasicConv3d(128, 512, kernel_size=3, stride=1, padding=1) # (8->8)

        # 5) Adaptive average pooling to (3,3,3)
        #    Spatial dimension 8->3
        self.adaptive_pool = nn.AdaptiveAvgPool3d((3,3,3))

    def forward(self, x):
        # x: (B, 3, 64, 64, 64)
        x = self.conv1(x)           # -> (B, 32, 32, 32, 32)
        x = self.inception1(x)      # -> (B, 64, 32, 32, 32)
        x = self.maxpool1(x)        # -> (B, 64, 16, 16, 16)

        x = self.inception2(x)      # -> (B, 128, 16, 16, 16)
        x = self.maxpool2(x)        # -> (B, 128, 8, 8, 8)

        x = self.conv2(x)           # -> (B, 512, 8, 8, 8)
        x = self.adaptive_pool(x)   # -> (B, 512, 3, 3, 3)
        return x

# 4. Age prediction network (using the two-Inception-block feature extractor)
class AgePredictionNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AgePredictionNet, self).__init__()
        # Replace with feature extractor that has only two Inception blocks
        self.feature_extractor = SimpleInception3DNet()

        # Output is (B, 512, 3, 3, 3)
        self.flattened_size = 512 * 3 * 3 * 3

        # Fully-connected part
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128 + 1, 1)  # Concatenate sex_feature => +1
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sex_feature):
        """
        x: 3D input, shape (B, 3, 64, 64, 64)
        sex_feature: tensor of shape (B,), representing sex feature
        """
        # 1) Extract 3D features
        x = self.feature_extractor(x)  # -> (B, 512, 3, 3, 3)

        # 2) Flatten
        x = x.view(x.size(0), -1)      # -> (B, 512*3*3*3)

        # 3) Fully-connected + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 4) Concatenate sex feature
        sex_feature = sex_feature.unsqueeze(1)  # (B,1)
        x = torch.cat((x, sex_feature), dim=1)  # (B, 128 + 1)

        # 5) Regression
        x = F.relu(self.fc2(x))  # (B, 1)
        return x



# class AgePredictionNet(nn.Module):
#     def __init__(self, dropout_rate=0.5):
#         super(AgePredictionNet, self).__init__()
#         # Image feature extraction module
#         self.feature_extractor = VisualFeatureExtractor()
#
#         # Fully-connected layers
#         self.flattened_size = 512 * 4 * 4 * 4  # Adjust according to VisualFeatureExtractor output size
#         self.fc1 = nn.Linear(self.flattened_size, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, x, sex_feature):
#         # Image feature extraction
#         x = self.feature_extractor(x)
#
#         # Flatten image features
#         x = x.view(x.size(0), -1)  # Flatten the feature map
#
#         # Fully-connected layers
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))  # ReLU in the last layer to ensure non-negative output
#
#         return x

    # def forward(self, x, sex_feature):
    #     # Image feature extraction
    #     x = self.feature_extractor(x)
    #
    #     # Sex feature processing
    #     sex_feature = F.relu(self.sex_fc(sex_feature.unsqueeze(1)))  # Expand dims and map
    #
    #     # Concatenate image and sex features
    #     x = x.view(x.size(0), -1)  # Flatten image features
    #     x = torch.cat((x, sex_feature), dim=1)  # Merge features
    #
    #     # Fully-connected layers
    #     if self.fc1 is None:
    #         flattened_size = x.size(1)
    #         self.fc1 = nn.Linear(flattened_size, 128).to(x.device)
    #         self.fc2 = nn.Linear(128, 1).to(x.device)
    #
    #     x = F.relu(self.fc1(x))
    #     x = self.dropout(x)
    #     x = self.fc2(x)
    #
    #     return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedMSEMAELossWithConstraint(nn.Module):
    def __init__(self, alpha=0.5, min_age=0, max_age=120, penalty=1.0):
        """
        Combined loss of MSE and MAE with a range constraint on predictions.

        Args:
            alpha (float): Weight between MSE and MAE. Larger alpha increases MSE influence.
            min_age (float): Lower bound constraint for age.
            max_age (float): Upper bound constraint for age.
            penalty (float): Penalty weight for out-of-range predictions.
        """
        super(CombinedMSEMAELossWithConstraint, self).__init__()
        self.alpha = alpha
        self.min_age = min_age
        self.max_age = max_age
        self.penalty = penalty

    def forward(self, predictions, targets):
        # Base loss (weighted combination of MSE and MAE)
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        base_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss

        # Penalty loss for out-of-range predictions
        lower_bound_penalty = torch.relu(self.min_age - predictions)  # Penalty for below minimum
        upper_bound_penalty = torch.relu(predictions - self.max_age)  # Penalty for above maximum
        penalty_loss = torch.mean(lower_bound_penalty + upper_bound_penalty)

        # Total loss = base loss + constraint penalty
        total_loss = base_loss + self.penalty * penalty_loss

        return total_loss




# Function to evaluate the model on the validation set and print CCID, gt, predicted age, and calculate MAE

def evaluate_model(valid_loader, model, criterion):
    """
    Evaluate model performance using CombinedMSEMAELossWithConstraint.

    Args:
        valid_loader (DataLoader): Validation DataLoader.
        model (torch.nn.Module): Model to be evaluated.
        criterion (nn.Module): Custom loss CombinedMSEMAELossWithConstraint.

    Returns:
        tuple: Average validation loss (val_loss) and mean absolute error (MAE).
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(valid_loader, desc="Validation")):
            inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)

            # Model prediction
            outputs = model(inputs, sex_labels)

            # Compute CombinedMSEMAELossWithConstraint
            loss = criterion(outputs, real_ages.unsqueeze(1))
            val_loss += loss.item()

            # Store ground-truth and predictions
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

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgePredictionNet().to(device)
criterion = CombinedMSEMAELossWithConstraint(alpha=0.5, min_age=0, max_age=120, penalty=1.0)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50)


# Initialize the best validation loss to a high value
best_val_loss = float('inf')  # Initialize to a large number

# Create a log file and write headers
log_file = 'experiments/genderins.txt'
with open(log_file, 'w') as f:
    f.write("Epoch,Training Loss,Validation Loss,Training MAE,Validation MAE\n")

# Training loop with validation, model saving, and MAE output
for epoch in range(5000):  # Number of epochs
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_labels_train = []
    all_predictions_train = []

    # Training loop
    # Training loop with tqdm progress bar
    for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
        inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs, sex_labels)  # Adjust model to accept sex_labels
        loss = criterion(outputs, real_ages.unsqueeze(1))  # Use real_ages to compute loss
        loss.backward()
        optimizer.step()



        running_loss += loss.item()

        # Store labels and predictions for MAE calculation
        all_labels_train.extend(real_ages.tolist())  # Convert to list and append
        all_predictions_train.extend(outputs.squeeze(1).tolist())  # Convert to list and append

    # Calculate Training MAE
    mae_train = mean_absolute_error(all_labels_train, all_predictions_train)
    print(f"Epoch {epoch + 1}, Training MAE: {mae_train:.4f}")

    # Evaluate on the validation set, print MAE, and print CCID, gt age, and predicted age
    val_loss, mae_val = evaluate_model(valid_loader, model, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'experiments/genderins.pth')
        print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

    with open(log_file, 'a') as f:
        f.write(f"{epoch + 1},{running_loss / len(train_loader):.4f},{val_loss:.4f},{mae_train:.4f},{mae_val:.4f}\n")

    print(
        f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {mae_val:.4f}')

print('Finished Training')
