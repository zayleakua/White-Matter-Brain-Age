"""
White Matter Brain Age — 3D ResNet Training Pipeline (FA/MD/MO + Sex) with AdamW, CosineAnnealing, and Range-Constrained MSE/MAE Loss

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
import numpy as np  # Numerical computing and array operations
import nibabel as nib  # Load NIfTI medical images
import torch  # PyTorch core library
from torch.utils.data import Dataset  # Custom dataset class
import random  # Random number generation
from scipy.ndimage import rotate  # For image rotation
from torch.optim.lr_scheduler import StepLR

# >>> Added imports to connect modules (only change made for splitting):
from dataset import ImageDataset
from model import AgePredictionNet
from losses import CombinedMSEMAELossWithConstraint
# <<<

# Load CSV file with CCID and age labels
csv_file = 'data3.csv'
df = pd.read_csv(csv_file)
df = df[['CCID', 'Age', 'Sex']]

# Set the folder containing subfolders named after CCIDs
image_base_dir = 'output'

# Split data into train and test sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Default behavior: normalization and augmentation enabled for training
train_dataset = ImageDataset(train_df, image_base_dir)
valid_dataset = ImageDataset(valid_df, image_base_dir, augment=False)  # Disable augmentation for validation

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate_model(valid_loader, model, criterion):
    """
    Evaluate model performance using CombinedMSEMAELossWithConstraint.
    """
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(valid_loader, desc="Validation")):
            inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)
            outputs = model(inputs, sex_labels)
            loss = criterion(outputs, real_ages.unsqueeze(1))
            val_loss += loss.item()
            all_labels.extend(real_ages.tolist())
            all_predictions.extend(outputs.squeeze(1).tolist())

    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"Validation MAE: {mae:.4f}")
    return val_loss / len(valid_loader), mae

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AgePredictionNet().to(device)
criterion = CombinedMSEMAELossWithConstraint(alpha=0.5, min_age=0, max_age=120, penalty=1.0)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_val_loss = float('inf')

log_file = 'experiments/gender.txt'
with open(log_file, 'w') as f:
    f.write("Epoch,Training Loss,Validation Loss,Training MAE,Validation MAE\n")

for epoch in range(5000):  # Number of epochs
    model.train()
    running_loss = 0.0
    all_labels_train = []
    all_predictions_train = []

    for i, (inputs, sex_labels, real_ages) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
        inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, sex_labels)
        loss = criterion(outputs, real_ages.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_labels_train.extend(real_ages.tolist())
        all_predictions_train.extend(outputs.squeeze(1).tolist())

    mae_train = mean_absolute_error(all_labels_train, all_predictions_train)
    print(f"Epoch {epoch + 1}, Training MAE: {mae_train:.4f}")

    val_loss, mae_val = evaluate_model(valid_loader, model, criterion)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'experiments/gender.pth')
        print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

    with open(log_file, 'a') as f:
        f.write(f"{epoch + 1},{running_loss / len(train_loader):.4f},{val_loss:.4f},{mae_train:.4f},{mae_val:.4f}\n")

    print(
        f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {mae_val:.4f}')

print('Finished Training')
