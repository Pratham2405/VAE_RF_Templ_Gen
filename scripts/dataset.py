# This module deals with the preparation of the dataset by Dataset. Later to be used by the training script to load the dataset by DataLoader.
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader

class C6DDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu") 
        c6d = data['c6d']   # [N, 40, 40, 4]
        mask = data['mask'] # [N, 40, 40]

        c6d = c6d.permute(0, 3, 1, 2).contiguous()  # [N, 4, 40, 40]

        self.x = c6d.float()
        self.mask = mask.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx]
