"""
Multimodal 3D dMRI (FA/MD/MO) Dataset & Augmentation for White Matter Brain Age (with Sex Metadata)

@author: Puzhen & Ruijia
"""

import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate

class ImageDataset(Dataset):
    def __init__(self, dataframe, image_base_dir, transform=None, augment=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'CCID', 'Age', and 'Sex'.
            image_base_dir (str): Path to the base directory containing image folders.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool, optional): If True, apply data augmentation (default: True).
        """
        self.dataframe = dataframe
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        ccid = self.dataframe.iloc[idx]['CCID']
        sex = self.dataframe.iloc[idx]['Sex']      # Binary feature 0/1
        real_age = self.dataframe.iloc[idx]['Age'] # Ground-truth age

        ccid_folder = os.path.join(self.image_base_dir, ccid)
        img_files = ['FA.nii', 'MD.nii', 'MO.nii']

        images = []
        for img_file in img_files:
            img_path = os.path.join(ccid_folder, img_file)
            img = nib.load(img_path).get_fdata()
            images.append(img)

        images = np.stack(images, axis=0)  # (3, H, W, D)
        images = self._normalize_images(images)

        if self.augment:
            images = self._augment_images(images)

        images = images.copy()  # avoid negative strides

        return (
            torch.tensor(images, dtype=torch.float32),
            torch.tensor(sex, dtype=torch.float32),
            torch.tensor(real_age, dtype=torch.float32)
        )

    def _normalize_images(self, images):
        means = images.mean(axis=(1, 2, 3), keepdims=True)
        stds = images.std(axis=(1, 2, 3), keepdims=True)
        return (images - means) / (stds + 1e-8)

    def _augment_images(self, images):
        if random.random() > 0.5:
            images = np.flip(images, axis=1)  # height
        if random.random() > 0.5:
            images = np.flip(images, axis=2)  # width

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            for i in range(images.shape[0]):
                images[i] = rotate(images[i], angle, axes=(1, 2), reshape=False, mode='nearest')

        crop_size = 100
        h, w, d = images.shape[1:]
        ch, cw, cd = min(crop_size, h), min(crop_size, w), min(crop_size, d)

        sh = random.randint(0, h - ch) if h > ch else 0
        sw = random.randint(0, w - cw) if w > cw else 0
        sd = random.randint(0, d - cd) if d > cd else 0

        images = images[:, sh:sh+ch, sw:sw+cw, sd:sd+cd]
        return images