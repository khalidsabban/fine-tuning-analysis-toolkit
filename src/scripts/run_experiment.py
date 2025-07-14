#!/usr/bin/env python3
# File: src/scripts/run_experiment.py

print("▶ Starting QLoRA experiment")

import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.nn.functional as F
import os

from toolkit.modules.carbon_tracker import CarbonTracker
from toolkit.modules.data_module import HFDataModule
from toolkit.modules.trainer_module import TrainerModule

# Set environment variables for optimal performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point for fine-tuning GPT-2 with QLoRA on a HuggingFace dataset.
    """
    print("🔧 Initializing QLoRA experiment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"🎯 CUDA device: {torch.cuda.get_device_name()}")
        print(f"💾 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Initial GPU memory usage: {initial_memory:.2f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")

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
        max_length=cfg.data.get('max_length', 512),
    )
    dm.setup()

    # 3) Initialize the TrainerModule with QLoRA
    model = TrainerModule(
        base_model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        lora_rank=cfg.model.lora_rank,
        learning_rate=cfg.training.learning_rate,
        gradient_checkpointing=cfg.training.get('gradient_checkpointing', True),
        use_qlora=cfg.model.get('use_qlora', True),
        quantization_config=cfg.model.get('quantization_config', 'nf4'),
        max_length=cfg.data.get('max_length', 512),
    )

    # Check memory after model loading
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Memory after model loading: {model_memory:.2f} GB")

    # 4) Create trainer with QLoRA-optimized settings
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        precision=16 if cfg.training.get('use_mixed_precision', True) else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.training.get('gradient_accumulation_steps', 1),
        deterministic=False,
        benchmark=True,
        log_every_n_steps=10,
    )

    # 5) Evaluate *before* training
    sample_texts = [str(s) for s in cfg.eval.samples]
    print("\n🔍 === EVALUATION BEFORE TRAINING ===")
    
    model.eval()
    with torch.no_grad():
        logits_pre = model(sample_texts)
        probs_pre = F.softmax(logits_pre, dim=-1)
        preds_pre = torch.argmax(probs_pre, dim=-1)
        
        print("📊 Pre-training probabilities:")
        for i, (text, prob, pred) in enumerate(zip(sample_texts, probs_pre, preds_pre)):
            print(f"  Text: '{text[:50]}...'")
            print(f"  Probs: {prob.tolist()}")
            print(f"  Pred: {pred.item()}")
            print()

    # 6) Run the training
    print("\n🚀 === STARTING TRAINING ===")
    if torch.cuda.is_available():
        pre_train_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Memory before training: {pre_train_memory:.2f} GB")

    model.train()
    
    try:
        trainer.fit(model, dm)
        print("✅ Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM Error: {e}")
        print("💡 Suggestions:")
        print("   - Reduce batch_size further")
        print("   - Try fp4 quantization instead of nf4")
        print("   - Reduce max_length")
        print("   - Increase gradient_accumulation_steps")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise

    # 7) Evaluate *after* training
    print("\n🔍 === EVALUATION AFTER TRAINING ===")
    
    if torch.cuda.is_available():
        post_train_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Memory after training: {post_train_memory:.2f} GB")

    model.eval()
    with torch.no_grad():
        logits_post = model(sample_texts)
        probs_post = F.softmax(logits_post, dim=-1)
        preds_post = torch.argmax(probs_post, dim=-1)

        print("📊 Post-training results:")
        for i, (text, prob_pre, prob_post, pred_pre, pred_post) in enumerate(
            zip(sample_texts, probs_pre, probs_post, preds_pre, preds_post)
        ):
            print(f"  Text: '{text[:50]}...'")
            print(f"  Before: {prob_pre.tolist()} → {pred_pre.item()}")
            print(f"  After:  {prob_post.tolist()} → {pred_post.item()}")
            print(f"  Change: {'✅' if pred_pre != pred_post else '➖'}")
            print()

    # 8) Stop carbon tracking
    carbon.stop()

    # Final memory report
    if torch.cuda.is_available():
        final_memory = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 === MEMORY REPORT ===")
        print(f"Final memory usage: {final_memory:.2f} GB")
        print(f"Peak memory usage: {peak_memory:.2f} GB")
        print(f"Memory efficiency: {peak_memory/16:.1%} of GPU capacity")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

    print("\n🎉 Experiment completed!")


if __name__ == "__main__":
    main()
    