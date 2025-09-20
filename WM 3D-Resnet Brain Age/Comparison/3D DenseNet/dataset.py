import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate

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
        means = images.mean(axis=(1, 2, 3), keepdims=True)
        stds = images.std(axis=(1, 2, 3), keepdims=True)
        return (images - means) / (stds + 1e-8)

    def _augment_images(self, images):
        """
        Apply random 3D augmentations: flips, rotation, and random crop.
        """
        if random.random() > 0.5:
            images = np.flip(images, axis=1)
        if random.random() > 0.5:
            images = np.flip(images, axis=2)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            for i in range(images.shape[0]):
                images[i] = rotate(images[i], angle, axes=(1, 2), reshape=False, mode='nearest')

        crop_size = 100
        h, w, d = images.shape[1:]
        crop_size_h, crop_size_w, crop_size_d = min(crop_size, h), min(crop_size, w), min(crop_size, d)

        start_h = random.randint(0, h - crop_size_h) if h > crop_size_h else 0
        start_w = random.randint(0, w - crop_size_w) if w > crop_size_w else 0
        start_d = random.randint(0, d - crop_size_d) if d > crop_size_d else 0

        return images[:, start_h:start_h+crop_size_h, start_w:start_w+crop_size_w, start_d:start_d+crop_size_d]
