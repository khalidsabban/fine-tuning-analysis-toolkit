print("▶ Starting run_experiment entrypoint")

import hydra
from omegaconf import DictConfig
from toolkit.modules.carbon_tracker import CarbonTracker
from toolkit.modules.data_module import DummyDataModule
from toolkit.modules.trainer_module import TrainerModule
import pytorch_lightning as pl

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    # Start carbon
    carbon = CarbonTracker(
        project_name=cfg.carbon.tracker.project_name,
        output_dir=cfg.carbon.tracker.output_dir,
    )
    carbon.start()

    # 1) Spin up DataModule
    dm = DummyDataModule(batch_size=cfg.training.batch_size)
    dm.setup()

    # 2) Spin up LightningModule
    model = TrainerModule(
        base_model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lora_rank=cfg.model.lora_rank,
        learning_rate=cfg.training.learning_rate,
    )

    # 3) Trainer: one step (or max_steps)
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps, # e.g. set max_steps=1 in your config
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, dm)

    # 4) do a forward pass on a tiny batch and print the logits
    sample = ["Hello world!", "How are you?"]
    logits = model(sample)
    print("-> Classification logits:\n", logits)

    carbon.stop()

if __name__ == "__main__":
    main()
    