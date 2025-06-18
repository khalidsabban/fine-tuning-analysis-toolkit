import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class DummyDataset(Dataset):
    """
    A minimal dataset returning (text, label) pairs for smoke-testing.
    """
    def __init__(self, data_file=None):
        # For now, ignore data_file and use hardcoded examples
        self.samples = [
            ("This is positive.", 1),
            ("This is negative.", 0),
            ("Another positive example.", 1),
            ("Another negative example.", 0)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return {"text": text, "label": torch.tensor(label, dtype=torch.long)}

class DummyDataModule(pl.LightningDataModule):
    """
    A LightningDataModule wrapping DummyDataset for train/val.
    """
    def __init__(self, batch_size: int = 2, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Called on every GPU separately (if using ddp)
        self.train_dataset = DummyDataset()
        self.val_dataset = DummyDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    