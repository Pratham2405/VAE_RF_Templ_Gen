# This module deals with the preparation of the dataset by Dataset. Later to be used by the training script to load the dataset by DataLoader.
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class C6DDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu")  # {'c6d': ..., 'mask': ..., 'params': ...}
        c6d = data['c6d']   # [N, 40, 40, 4]
        mask = data['mask'] # [N, 40, 40]

        # channels-last -> channels-first
        c6d = c6d.permute(0, 3, 1, 2).contiguous()  # [N, 4, 40, 40]

        self.x = c6d.float()
        self.mask = mask.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx]

# Usage (in your training script):
# data = torch.load(".../2erlA_c6d_with_mask.pt")
# dataset = C6DDataset(data['c6d'], data['mask'])
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
