#!/usr/bin/env python3
# File: src/scripts/run_experiment.py

print("▶ Starting run_experiment entrypoint")

import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.nn.functional as F

from toolkit.modules.carbon_tracker import CarbonTracker
from toolkit.modules.data_module import HFDataModule
from toolkit.modules.trainer_module import TrainerModule


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Entry point for fine-tuning GPT-2 (LoRA) on a HuggingFace dataset.
    """

    # 1) Start carbon tracking
    carbon = CarbonTracker(
        project_name=cfg.carbon.tracker.project_name,
        output_dir=cfg.carbon.tracker.output_dir,
    )
    carbon.start()

    # 2) Prepare the HF dataset DataModule
    dm = HFDataModule(
        dataset_name=cfg.data.dataset_name,
        split=cfg.data.split,
        text_field=cfg.data.text_field,
        label_field=cfg.data.label_field,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
    )
    dm.setup()

    # 3) Initialize the TrainerModule (LoRA-wrapped model)
    model = TrainerModule(
        base_model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lora_rank=cfg.model.lora_rank,
        learning_rate=cfg.training.learning_rate,
    )

    # 4) Run one (or more) training steps
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    # 5) Evaluate *before* training
    sample_texts = [str(s) for s in cfg.eval.samples]
    print("\n=== EVALUATION BEFORE TRAINING ===")
    logits_pre = model(sample_texts)
    probs_pre = F.softmax(logits_pre, dim=-1)
    preds_pre = torch.argmax(probs_pre, dim=-1)
    print("Probs: ", probs_pre)
    print("Preds: ", preds_pre.tolist())

    # 6) Run the training
    trainer.fit(model, dm)

    # 7) Evaluate *after* training
    print("\n=== EVALUATION AFTER TRAINING ===")

    # Forward pass on sample texts
    # Assuming 'sample_texts' is a batch of tokenized text inputs
    logits_post = model(sample_texts)

    # Logits shape: [batch_size, num_labels]

    # Convert logits to probabilities
    probs_post = F.softmax(logits_post, dim=-1)

    # Get predicted class indices
    preds_post = torch.argmax(probs_post, dim=-1)

    # Print results
    print("\n\n\n→ Probabilities on eval sample:\n", probs_post)
    print("→ Predicted classes:", preds_post.tolist())
    print("\n\n\n")

    

    # 8) Stop carbon tracking (writes CO₂ log file)
    carbon.stop()


if __name__ == "__main__":
    main()
