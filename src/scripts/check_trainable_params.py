# src/toolkit/scripts/check_trainable_params.py

import hydra
from omegaconf import DictConfig
from toolkit.modules.trainer_module import TrainerModule

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """
    Counts how many trainable parameters your PEFT-wrapped model has.
    """

    # Note: config.yaml uses `model.name`, not `model.base_model_name`
    model = TrainerModule(
        base_model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lora_rank=cfg.model.lora_rank,
        learning_rate=cfg.training.learning_rate,  # from `training.learning_rate`
    )

    total = sum(
        p.numel() for p in model.adapter.model.parameters() if p.requires_grad
    )
    print(f"🔍 Trainable parameters detected: {total:,}")
    if total == 0:
        raise RuntimeError(
            "❌ No trainable parameters found! Did you subclass ModelAdapter from nn.Module?"
        )

if __name__ == "__main__":
    main()
