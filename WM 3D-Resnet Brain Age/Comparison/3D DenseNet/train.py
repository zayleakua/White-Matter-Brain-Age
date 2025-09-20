import os, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ImageDataset
from model import AgePredictionNet
from losses import CombinedMSEMAELossWithConstraint

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def evaluate_model(valid_loader, model, criterion, device):
    """
    Evaluate model performance using CombinedMSEMAELossWithConstraint.
    Returns validation loss and MAE.
    """
    model.eval()
    val_loss = 0.0
    all_labels, all_predictions = [], []
    with torch.no_grad():
        for inputs, sex_labels, real_ages in tqdm(valid_loader, desc="Validation"):
            inputs, sex_labels, real_ages = inputs.to(device), sex_labels.to(device), real_ages.to(device)
            outputs = model(inputs, sex_labels)
            loss = criterion(outputs, real_ages.unsqueeze(1))
            val_loss += loss.item()
            all_labels.extend(real_ages.tolist())
            all_predictions.extend(outputs.squeeze(1).tolist())
    mae = mean_absolute_error(all_labels, all_predictions)
    print(f"Validation MAE: {mae:.4f}")
    return val_loss / len(valid_loader), mae

if __name__ == "__main__":
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CSV
    df = pd.read_csv('data3.csv')[['CCID','Age','Sex']]
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    image_base_dir = 'output'
    train_loader = DataLoader(ImageDataset(train_df, image_base_dir), batch_size=16, shuffle=True)
    valid_loader = DataLoader(ImageDataset(valid_df, image_base_dir, augment=False), batch_size=16)

    model = AgePredictionNet().to(device)
    criterion = CombinedMSEMAELossWithConstraint()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float('inf')
    os.makedirs('experiments', exist_ok=True)
    log_file = 'experiments/genderdense.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch,Training Loss,Validation Loss,Training MAE,Validation MAE\n")

    for epoch in range(5000):
        model.train()
        running_loss, all_labels_train, all_predictions_train = 0.0, [], []
        for inputs, sex_labels, real_ages in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
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
        print(f"Epoch {epoch+1}, Training MAE: {mae_train:.4f}")

        val_loss, mae_val = evaluate_model(valid_loader, model, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/genderdense.pth')
            print(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")

        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{running_loss/len(train_loader):.4f},{val_loss:.4f},{mae_train:.4f},{mae_val:.4f}\n")

    print("Finished Training")
