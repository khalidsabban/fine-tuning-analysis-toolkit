from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from collections import Counter

class HFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for any 🤗 Dataset, with an arbitrary split/subset.
    """
    def __init__(
        self,
        dataset_name: str = "imdb",
        split: str = "train[:10%]",
        text_field: str = "text",
        label_field: str = "label",
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
        val_split_ratio: float = 0.1,  # Add validation split
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.label_field = label_field
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_split_ratio = val_split_ratio

    def setup(self, stage=None):
        # Load the split, e.g. "train[:25%]" or "train"
        self.ds = load_dataset(path=self.dataset_name, split=self.split)
        
        # 🔍 DEBUG: Check dataset properties
        print(f"\n🔍 === DATASET DEBUG INFO ===")
        print(f"Dataset size: {len(self.ds)}")
        print(f"Text field: {self.text_field}")
        print(f"Label field: {self.label_field}")
        
        # Check first few examples
        print("\n📝 Sample examples:")
        for i in range(min(3, len(self.ds))):
            example = self.ds[i]
            text_preview = str(example[self.text_field])[:100] + "..." if len(str(example[self.text_field])) > 100 else str(example[self.text_field])
            print(f"  Example {i}: Text='{text_preview}', Label={example[self.label_field]}")
        
        # 🚨 CRITICAL: Check label distribution
        all_labels = [ex[self.label_field] for ex in self.ds]
        label_counts = Counter(all_labels)
        print(f"\n📊 Label distribution: {dict(label_counts)}")
        
        # Check for problematic label distributions
        if len(label_counts) == 1:
            print("🚨 CRITICAL ERROR: All labels are the same value!")
            print("   This will cause loss to be 0. Check your dataset split.")
        elif min(label_counts.values()) / max(label_counts.values()) < 0.1:
            print("⚠️  Warning: Very imbalanced labels detected")
            print(f"   Ratio: {min(label_counts.values())}/{max(label_counts.values())} = {min(label_counts.values())/max(label_counts.values()):.3f}")
        
        # Check label values are in expected range
        unique_labels = set(all_labels)
        expected_labels = {0, 1}  # For binary classification
        if unique_labels != expected_labels:
            print(f"⚠️  Warning: Unexpected label values: {unique_labels}")
            print(f"   Expected: {expected_labels}")
        
        # Split into train/validation
        if self.val_split_ratio > 0:
            split_idx = int(len(self.ds) * (1 - self.val_split_ratio))
            self.train_ds = self.ds.select(range(split_idx))
            self.val_ds = self.ds.select(range(split_idx, len(self.ds)))
            
            print(f"\n📊 Data splits:")
            print(f"  Training: {len(self.train_ds)} samples")
            print(f"  Validation: {len(self.val_ds)} samples")
        else:
            self.train_ds = self.ds
            self.val_ds = None
            print(f"\n📊 Training samples: {len(self.train_ds)}")

    def _collate(self, batch):
        texts = [str(ex[self.text_field]) for ex in batch]  # Ensure strings
        labels = torch.tensor([int(ex[self.label_field]) for ex in batch], dtype=torch.long)  # Ensure int conversion
        
        # 🔍 DEBUG: Print batch info occasionally
        if hasattr(self, '_debug_batch_count'):
            self._debug_batch_count += 1
        else:
            self._debug_batch_count = 1
            
        if self._debug_batch_count <= 2:  # Debug first 2 batches
            print(f"\n🔍 Batch {self._debug_batch_count} debug:")
            print(f"  Batch size: {len(texts)}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Label tensor dtype: {labels.dtype}")
            print(f"  Text lengths: {[len(t) for t in texts]}")
        
        return {"text": texts, "label": labels}

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate,
        )
    