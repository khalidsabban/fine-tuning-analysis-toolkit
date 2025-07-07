from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch

class HFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for any 🤗 Dataset, with an arbitrary split/subset.
    """
    def __init__(
        self,
        dataset_name: str = "imdb",
        split: str = "train[:1%]",
        text_field: str = "text",
        label_field: str = "label",
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split       = split
        self.text_field  = text_field
        self.label_field = label_field
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed

    def setup(self, stage=None):
        # load the split, e.g. "train[:1%]" or "train"
        self.ds = load_dataset(path=self.dataset_name, split=self.split)
        # optionally map or filter here

    def _collate(self, batch):
        texts  = [ex[self.text_field]  for ex in batch]
        labels = torch.tensor([ex[self.label_field] for ex in batch], dtype=torch.long)
        return {"text": texts, "label": labels}

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate,
        )

    # if you want a validation split, you can add val_dataloader() similarly
