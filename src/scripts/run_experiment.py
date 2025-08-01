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
    Entry point for fine-tuning Llama-2 with QLoRA on classification or QA tasks.
    """
    # Add memory management at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Set memory fraction to leave some headroom
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    task_type = cfg.task.type
    print(f"\n🔧 Initializing Llama-2 QLoRA experiment for {task_type}...")
    print(f"🎯 Model: {cfg.model.name}")
    print(f"📊 LoRA rank: {cfg.model.lora_rank}")
    print(f"🔢 Quantization: {cfg.model.quantization_config}")
    print(f"📝 Task: {task_type}")
    
    # Check CUDA availability and memory
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 CUDA device: {device_name}")
        print(f"💾 Total GPU memory: {total_memory:.2f} GB")
        
        if total_memory < 8:
            print("⚠️  Warning: GPU memory may be insufficient for Llama-2-7B")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"📊 Initial GPU memory usage: {initial_memory:.2f} GB")
    else:
        print("❌ CUDA not available - Llama-2-7B requires GPU!")
        return

    # 1) Start carbon tracking
    carbon = CarbonTracker(
        project_name=cfg.carbon.tracker.project_name,
        output_dir=cfg.carbon.tracker.output_dir,
    )
    carbon.start()

    try:
        # 2) Initialize the TrainerModule FIRST (to get tokenizer)
        print("🤖 Loading Llama-2 model...")
        model = TrainerModule(
            base_model_name=cfg.model.name,
            task_type=task_type,
            num_labels=cfg.model.get('num_labels', 2) if task_type == "classification" else None,
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
                print("⚠️  High memory usage detected. Consider optimizations.")

        # 3) Prepare the HF dataset DataModule
        print("📚 Loading dataset...")
        
        if task_type == "classification":
            dm = HFDataModule(
                task_type=task_type,
                dataset_name=cfg.data.dataset_name,
                split=cfg.data.split,
                text_field=cfg.data.text_field,
                label_field=cfg.data.label_field,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.data.num_workers,
                max_length=cfg.data.get('max_length', 512),
                val_split_ratio=cfg.data.get('val_split_ratio', 0.1),
            )
        elif task_type == "question_answering":
            dm = HFDataModule(
                task_type=task_type,
                dataset_name=cfg.data.dataset_name,
                split=cfg.data.split,
                question_field=cfg.data.question_field,
                context_field=cfg.data.context_field,
                answers_field=cfg.data.answers_field,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.data.num_workers,
                max_length=cfg.data.get('max_length', 512),
                val_split_ratio=cfg.data.get('val_split_ratio', 0.1),
                max_answer_length=cfg.qa.get('max_answer_length', 30),
                doc_stride=cfg.qa.get('doc_stride', 128),
                tokenizer=model.adapter.tokenizer,  # FIXED: Pass tokenizer immediately
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Setup data module
        dm.setup()
        
        # FIXED: Validate QA data quality
        if task_type == "question_answering" and hasattr(dm, 'validate_qa_data'):
            dm.validate_qa_data(num_samples=5)

        # 4) Create trainer with better settings for QA
        trainer = pl.Trainer(
            max_steps=cfg.training.max_steps,
            max_epochs=cfg.training.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            precision=16 if cfg.training.get('use_mixed_precision', True) else 32,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm',
            accumulate_grad_batches=cfg.training.get('gradient_accumulation_steps', 1),
            deterministic=False,
            benchmark=True,
            log_every_n_steps=5,
            detect_anomaly=False,
            limit_val_batches=0.1 if task_type == "question_answering" else 0,  # FIXED: Enable some validation for QA
            num_sanity_val_steps=0,
            # FIXED: Add callbacks for better monitoring
            callbacks=[
                pl.callbacks.LearningRateMonitor(logging_interval='step'),
            ] if task_type == "question_answering" else [],
        )

        # 5) Evaluate *before* training
        print(f"\n🔍 === EVALUATION BEFORE TRAINING ({task_type.upper()}) ===")
        
        model.eval()
        with torch.no_grad():
            try:
                if task_type == "classification":
                    sample_texts = [str(s) for s in cfg.eval.get('classification_samples', cfg.eval.get('samples', []))]
                    if sample_texts:
                        logits_pre = model(sample_texts)
                        probs_pre = F.softmax(logits_pre, dim=-1)
                        preds_pre = torch.argmax(probs_pre, dim=-1)
                        
                        print("📊 Pre-training classification results:")
                        for i, (text, prob, pred) in enumerate(zip(sample_texts, probs_pre, preds_pre)):
                            print(f"  Text: '{text[:50]}...'")
                            print(f"  Probs: [neg: {prob[0]:.3f}, pos: {prob[1]:.3f}]")
                            print(f"  Prediction: {'Positive' if pred.item() == 1 else 'Negative'}")
                            print()
                
                elif task_type == "question_answering":
                    qa_samples = cfg.eval.get('qa_samples', [])
                    if qa_samples:
                        questions = [sample['question'] for sample in qa_samples]
                        contexts = [sample['context'] for sample in qa_samples]
                        
                        qa_inputs = {
                            'questions': questions,
                            'contexts': contexts
                        }
                        
                        predicted_answers = model.adapter.extract_answer(qa_inputs)
                        
                        print("📊 Pre-training QA results:")
                        for question, context, pred_answer in zip(questions, contexts, predicted_answers):
                            print(f"  Question: '{question}'")
                            print(f"  Context: '{context[:100]}...'")
                            print(f"  Predicted Answer: '{pred_answer}'")
                            print()
                
            except Exception as e:
                print(f"⚠️  Pre-training evaluation failed: {e}")

        # 6) Run the training
        print(f"\n🚀 === STARTING {task_type.upper()} TRAINING ===")
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
            print("   1. Reduce batch_size (already at 1)")
            print("   2. Increase gradient_accumulation_steps")
            print("   3. Reduce max_length")
            print("   4. Reduce lora_rank further")
            print("   5. Use cpu offloading")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        except Exception as e:
            print(f"❌ Training error: {e}")
            if "token" in str(e).lower():
                print("💡 This might be an authentication error.")
                print("   Make sure you have access to the Llama-2 model on HuggingFace.")
                print("   You may need to run: huggingface-cli login")
            raise

        # 7) Evaluate *after* training
        print(f"\n🔍 === EVALUATION AFTER TRAINING ({task_type.upper()}) ===")
        
        if torch.cuda.is_available():
            post_train_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 Memory after training: {post_train_memory:.2f} GB")

        model.eval()
        with torch.no_grad():
            try:
                if task_type == "classification":
                    sample_texts = [str(s) for s in cfg.eval.get('classification_samples', cfg.eval.get('samples', []))]
                    if sample_texts:
                        logits_post = model(sample_texts)
                        probs_post = F.softmax(logits_post, dim=-1)
                        preds_post = torch.argmax(probs_post, dim=-1)

                        print("📊 Post-training classification results:")
                        for i, (text, prob_post, pred_post) in enumerate(zip(sample_texts, probs_post, preds_post)):
                            print(f"  Text: '{text[:50]}...'")
                            print(f"  Probs: [neg: {prob_post[0]:.3f}, pos: {prob_post[1]:.3f}]")
                            print(f"  Prediction: {'Positive' if pred_post.item() == 1 else 'Negative'}")
                            print()

                elif task_type == "question_answering":
                    qa_samples = cfg.eval.get('qa_samples', [])
                    if qa_samples:
                        questions = [sample['question'] for sample in qa_samples]
                        contexts = [sample['context'] for sample in qa_samples]
                        
                        qa_inputs = {
                            'questions': questions,
                            'contexts': contexts
                        }
                        
                        predicted_answers = model.adapter.extract_answer(qa_inputs)
                        
                        print("📊 Post-training QA results:")
                        for question, context, pred_answer in zip(questions, contexts, predicted_answers):
                            print(f"  Question: '{question}'")
                            print(f"  Context: '{context[:100]}...'")
                            print(f"  Predicted Answer: '{pred_answer}'")
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

    print(f"\n🎉 Llama-2 QLoRA {task_type} experiment completed!")


if __name__ == "__main__":
    main()
    