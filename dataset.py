import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom


class SkullDataset(Dataset):
    def __init__(self, ct_dir, mask_dir, file_list):
        self.ct_dir = ct_dir
        self.mask_dir = mask_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        ct_path = os.path.join(self.ct_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        ct_img = sitk.ReadImage(ct_path)
        mask_img = sitk.ReadImage(mask_path)

        ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        mask = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        
        
        

        target_size = (64, 128, 128)
        if ct.shape != target_size:
            zoom_factors = [t/s for t, s in zip(target_size, ct.shape)]
            ct = zoom(ct, zoom_factors, order=1)
            mask = zoom(mask, zoom_factors, order=0)

        ct_min, ct_max = ct.min(), ct.max()
        ct = (ct - ct_min) / (ct_max - ct_min + 1e-7)
        

        mask = (mask > 0.5).astype(np.float32)

        return torch.from_numpy(ct).unsqueeze(0), torch.from_numpy(mask)
