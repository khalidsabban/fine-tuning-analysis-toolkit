#!/usr/bin/env python3
# File: src/scripts/run_experiment.py

print("▶ Starting Llama-2 QLoRA experiment")

import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import warnings

from toolkit.modules.carbon_tracker import CarbonTracker
from toolkit.modules.data_module import HFDataModule
from toolkit.modules.trainer_module import TrainerModule

# Set environment variables for optimal performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point for fine-tuning Llama-2 with QLoRA on a HuggingFace dataset.
    """
    print("🔧 Initializing Llama-2 QLoRA experiment...")
    print(f"🎯 Model: {cfg.model.name}")
    print(f"📊 LoRA rank: {cfg.model.lora_rank}")
    print(f"🔢 Quantization: {cfg.model.quantization_config}")
    
    # Check CUDA availability and memory
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 CUDA device: {device_name}")
        print(f"💾 Total GPU memory: {total_memory:.2f} GB")
        
        # For Llama-2-7B, we need at least 8GB GPU memory with QLoRA
        if total_memory < 8:
            print("⚠️  Warning: GPU memory may be insufficient for Llama-2-7B")
            print("   Consider using a smaller model or reducing batch size/sequence length")
        
        # Clear cache and set memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Initial GPU memory usage: {initial_memory:.2f} GB")
    else:
        print("❌ CUDA not available - Llama-2-7B requires GPU!")
        print("   This model is too large to run on CPU effectively.")
        return

    # 1) Start carbon tracking
    carbon = CarbonTracker(
        project_name=cfg.carbon.tracker.project_name,
        output_dir=cfg.carbon.tracker.output_dir,
    )
    carbon.start()

    try:
        # 2) Prepare the HF dataset DataModule
        print("📚 Loading dataset...")
        dm = HFDataModule(
            dataset_name=cfg.data.dataset_name,
            split=cfg.data.split,
            text_field=cfg.data.text_field,
            label_field=cfg.data.label_field,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
        )
        dm.setup()

        # 3) Initialize the TrainerModule with QLoRA
        print("🤖 Loading Llama-2 model...")
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
            
            if model_memory > total_memory * 0.8:
                print("⚠️  High memory usage detected. Consider:")
                print("   - Reducing max_length")
                print("   - Using fp4 quantization instead of nf4")
                print("   - Reducing batch_size further")

        # 4) Create trainer with optimized settings for Llama-2
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
            log_every_n_steps=5,  # More frequent logging for monitoring
            detect_anomaly=False,  # Disable for performance
        )

        # 5) Evaluate *before* training
        sample_texts = [str(s) for s in cfg.eval.samples]
        print("\n🔍 === EVALUATION BEFORE TRAINING ===")
        
        model.eval()
        with torch.no_grad():
            try:
                logits_pre = model(sample_texts)
                probs_pre = F.softmax(logits_pre, dim=-1)
                preds_pre = torch.argmax(probs_pre, dim=-1)
                
                print("📊 Pre-training results:")
                for i, (text, prob, pred) in enumerate(zip(sample_texts, probs_pre, preds_pre)):
                    print(f"  Text: '{text[:50]}...'")
                    print(f"  Probs: [neg: {prob[0]:.3f}, pos: {prob[1]:.3f}]")
                    print(f"  Prediction: {'Positive' if pred.item() == 1 else 'Negative'}")
                    print()
            except Exception as e:
                print(f"⚠️  Pre-training evaluation failed: {e}")
                logits_pre, probs_pre, preds_pre = None, None, None

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
            print("\n💡 Optimization suggestions:")
            print("   1. Reduce batch_size from 4 to 2 or 1")
            print("   2. Increase gradient_accumulation_steps to maintain effective batch size")
            print("   3. Reduce max_length from 512 to 256 or 128")
            print("   4. Try fp4 quantization instead of nf4")
            print("   5. Reduce lora_rank from 8 to 4")
            print("   6. Disable gradient checkpointing if memory is critical")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            if "token" in str(e).lower():
                print("💡 This might be an authentication error.")
                print("   Make sure you have access to the Llama-2 model on HuggingFace.")
                print("   You may need to:")
                print("   1. Accept the license agreement on the model page")
                print("   2. Set up your HuggingFace token")
            raise

        # 7) Evaluate *after* training
        print("\n🔍 === EVALUATION AFTER TRAINING ===")
        
        if torch.cuda.is_available():
            post_train_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Memory after training: {post_train_memory:.2f} GB")

        model.eval()
        with torch.no_grad():
            try:
                logits_post = model(sample_texts)
                probs_post = F.softmax(logits_post, dim=-1)
                preds_post = torch.argmax(probs_post, dim=-1)

                print("📊 Post-training results:")
                for i, (text, prob_post, pred_post) in enumerate(zip(sample_texts, probs_post, preds_post)):
                    print(f"  Text: '{text[:50]}...'")
                    print(f"  Probs: [neg: {prob_post[0]:.3f}, pos: {prob_post[1]:.3f}]")
                    print(f"  Prediction: {'Positive' if pred_post.item() == 1 else 'Negative'}")
                    
                    # Show change if we have pre-training results
                    if probs_pre is not None and preds_pre is not None:
                        prob_pre = probs_pre[i]
                        pred_pre = preds_pre[i]
                        print(f"  Before: [neg: {prob_pre[0]:.3f}, pos: {prob_pre[1]:.3f}] → {'Positive' if pred_pre.item() == 1 else 'Negative'}")
                        print(f"  Change: {'✅ Changed' if pred_pre != pred_post else '➖ Same'}")
                    print()
            except Exception as e:
                print(f"⚠️  Post-training evaluation failed: {e}")

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise
    
    finally:
        # 8) Stop carbon tracking
        carbon.stop()

        # Final memory report
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n📊 === MEMORY REPORT ===")
            print(f"Final memory usage: {final_memory:.2f} GB")
            print(f"Peak memory usage: {peak_memory:.2f} GB")
            print(f"Memory efficiency: {peak_memory/total_memory:.1%} of GPU capacity")
            
            # Cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n🎉 Llama-2 QLoRA experiment completed!")


if __name__ == "__main__":
    main()
    