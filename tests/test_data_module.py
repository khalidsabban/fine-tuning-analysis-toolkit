import pytest
import torch
from toolkit.modules.data_module import DummyDataModule

@ pytest.mark.parametrize("batch_size", [1, 2])
def test_dummy_data_module_loads_non_empty_batch(batch_size):
    dm = DummyDataModule(batch_size=batch_size)
    # lightning calls setup internally, but call explicitly here
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # Assert we got a dict with "text" and "label"
    assert isinstance(batch, dict)
    assert "text" in batch and "label" in batch
    # Text batch should be non-empty list of strings
    assert len(batch["text"]) > 0
    # Label tensor should have size == batch_size
    labels = batch["label"]
    assert labels.shape[0] == batch_size
    assert labels.dtype == torch.long
    