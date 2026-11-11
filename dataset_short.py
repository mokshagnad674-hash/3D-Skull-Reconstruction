import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import zoom
import os

class SkullDataset(Dataset):
    def __init__(self, ct_dir, mask_dir, files):
        self.ct_dir, self.mask_dir, self.files = ct_dir, mask_dir, files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.ct_dir, f))).astype('float32')
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.mask_dir, f))).astype('float32')
        
        # Resize
        if ct.shape != (64, 128, 128):
            z = [64/ct.shape[0], 128/ct.shape[1], 128/ct.shape[2]]
            ct, mask = zoom(ct, z, order=1), zoom(mask, z, order=0)
        
        # Normalize
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-7)
        mask = (mask > 0.5).astype('float32')
        
        return torch.from_numpy(ct).unsqueeze(0), torch.from_numpy(mask)