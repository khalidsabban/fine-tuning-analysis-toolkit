import pytorch_lightning as pl
from toolkit.modules.trainer_module import TrainerModule
from toolkit.modules.data_module import DummyDataModule

def test_trainer_one_step(tmp_path):
    dm = DummyDataModule(batch_size=2)
    dm.setup()
    model = TrainerModule(
        base_model_name="NousResearch/Llama-2-7b-hf",
        num_labels=2,
        lora_rank=4,
        learning_rate=1e-3
    )
    trainer = pl.Trainer(
        default_root_dir=str(tmp_path),
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    trainer.fit(model, dm)
    