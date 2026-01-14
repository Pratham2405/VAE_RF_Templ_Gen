#Module for implementing dataloader which loads the data and prepares it into training-ready bathc_sizes with shuffling.
from torch.utils.data import DataLoader
from dataset import C6DDataset

def build_train_loader(path, batch_size, num_workers, pin_memory):
    ds = C6DDataset(path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
