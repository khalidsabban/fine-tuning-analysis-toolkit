import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from toolkit.modules.model_adapter import ModelAdapter
from typing import Dict, List, Union

class TrainerModule(pl.LightningModule):
    """
    Lightning module wrapping ModelAdapter with support for both classification and QA tasks.
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-hf",
        task_type: str = "classification",  # "classification" or "question_answering"
        num_labels: int = 2,  # Only used for classification
        lora_rank: int = 16,
        learning_rate: float = 1e-5, 
        gradient_checkpointing: bool = True,
        use_qlora: bool = True,
        quantization_config: str = "nf4",
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.task_type = task_type
        self.adapter = ModelAdapter(
            base_model_name=base_model_name,
            task_type=task_type,
            num_labels=num_labels,
            lora_rank=lora_rank,
            use_qlora=use_qlora,
            quantization_config=quantization_config,
            max_length=max_length,
        )
        
        self.learning_rate = learning_rate
        self.use_qlora = use_qlora
        self.step_count = 0
        
        print(f"📊 Model memory footprint: {self.adapter.get_memory_footprint()}")
        
        # Note: Gradient checkpointing is already enabled in ModelAdapter
        print("✅ Model initialized with gradient checkpointing")

    def forward(self, inputs):
        """Forward pass that handles both task types"""
        return self.adapter(inputs)

    def training_step(self, batch, batch_idx):
        if self.task_type == "classification":
            return self._training_step_classification(batch, batch_idx)
        elif self.task_type == "question_answering":
            return self._training_step_qa(batch, batch_idx)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _training_step_classification(self, batch, batch_idx):
        """Training step for classification tasks"""
        texts = batch["text"]
        labels = batch["label"]
        
        # Debug info for first few steps
        if self.step_count < 5:
            print(f"\n🔍 === CLASSIFICATION TRAINING STEP {self.step_count} DEBUG ===")
            print(f"Batch size: {len(texts)}")
            print(f"Labels: {labels.tolist()}")
            print(f"Sample texts: {[t[:50] + '...' for t in texts[:2]]}")
        
        # Forward pass
        logits = self(texts)
        
        if self.step_count < 5:
            print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        # Ensure labels are on the same device as logits
        labels = labels.to(logits.device)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        
        # Debug loss
        if self.step_count < 10:
            print(f"\n🚨 CLASSIFICATION LOSS DEBUG - Step {self.step_count}:")
            print(f"  Raw loss: {loss.item()}")
            unique_labels_in_batch = torch.unique(labels)
            if len(unique_labels_in_batch) == 1:
                print(f"  🚨 PROBLEM: All labels in batch are {unique_labels_in_batch[0].item()}")
        
        self.step_count += 1
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def _training_step_qa(self, batch, batch_idx):
        """Training step for QA tasks"""
        questions = batch["questions"]
        contexts = batch["contexts"]
        answers = batch["answers"]

        # Get batch size
        batch_size = len(questions)
        
        # Debug info for first few steps
        if self.step_count < 5:
            print(f"\n🔍 === QA TRAINING STEP {self.step_count} DEBUG ===")
            print(f"Batch size: {len(questions)}")
            print(f"Sample question: '{questions[0][:50]}...'")
            print(f"Sample context: '{contexts[0][:100]}...'")
            print(f"Sample answer: '{answers[0].get('text', 'N/A') if isinstance(answers[0], dict) else str(answers[0])}'")
        
        # Prepare inputs for model
        qa_inputs = {
            'questions': questions,
            'contexts': contexts
        }
        
        # Forward pass
        outputs = self(qa_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        if self.step_count < 5:
            print(f"Start logits shape: {start_logits.shape}")
            print(f"End logits shape: {end_logits.shape}")
        
        # Calculate positions for loss (simplified - using first answer)
        # In practice, you'd want more sophisticated position calculation
        start_positions = torch.zeros(len(questions), dtype=torch.long, device=start_logits.device)
        end_positions = torch.zeros(len(questions), dtype=torch.long, device=end_logits.device)
        
        # Try to find actual answer positions in context (simplified approach)
        for i, (question, context, answer_data) in enumerate(zip(questions, contexts, answers)):
            if isinstance(answer_data, dict) and 'text' in answer_data:
                answer_text = answer_data['text']
                answer_start_char = answer_data.get('answer_start', 0)
            else:
                answer_text = str(answer_data)
                answer_start_char = 0
            
            # This is a simplified position calculation
            # In practice, you'd want to use proper tokenization alignment
            if answer_text and answer_text in context:
                context_start_pos = context.find(answer_text)
                if context_start_pos != -1:
                    # Rough token position estimation (this should be more precise)
                    estimated_start = min(context_start_pos // 4, start_logits.size(1) - 1)
                    estimated_end = min(estimated_start + len(answer_text) // 4, end_logits.size(1) - 1)
                    start_positions[i] = estimated_start
                    end_positions[i] = max(estimated_start, estimated_end)
        
        # Calculate loss
        start_loss = F.cross_entropy(start_logits, start_positions)
        end_loss = F.cross_entropy(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        # Debug loss
        if self.step_count < 10:
            print(f"\n🚨 QA LOSS DEBUG - Step {self.step_count}:")
            print(f"  Start loss: {start_loss.item()}")
            print(f"  End loss: {end_loss.item()}")
            print(f"  Total loss: {total_loss.item()}")
            print(f"  Start positions: {start_positions[:3].tolist()}")
            print(f"  End positions: {end_positions[:3].tolist()}")
        
        self.step_count += 1
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("start_loss", start_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("end_loss", end_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
            
        if self.task_type == "classification":
            return self._validation_step_classification(batch, batch_idx)
        elif self.task_type == "question_answering":
            return self._validation_step_qa(batch, batch_idx)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _validation_step_classification(self, batch, batch_idx):
        """Validation step for classification"""
        texts = batch["text"]
        labels = batch["label"]
        
        logits = self(texts)
        labels = labels.to(logits.device)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return loss

    def _validation_step_qa(self, batch, batch_idx):
        """Validation step for QA"""
        questions = batch["questions"]
        contexts = batch["contexts"]
        answers = batch["answers"]
        
        # Get the batch size
        batch_size = len(questions)
        
        qa_inputs = {
            'questions': questions,
            'contexts': contexts
        }
        
        outputs = self(qa_inputs)
        
        # Calculate loss (simplified)
        start_positions = torch.zeros(len(questions), dtype=torch.long, device=outputs.start_logits.device)
        end_positions = torch.zeros(len(questions), dtype=torch.long, device=outputs.end_logits.device)
        
        start_loss = F.cross_entropy(outputs.start_logits, start_positions)
        end_loss = F.cross_entropy(outputs.end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        # Calculate approximate metrics
        predicted_answers = self.adapter.extract_answer(qa_inputs)
        
        # Simple EM calculation for validation
        exact_matches = []
        for pred, answer_data in zip(predicted_answers, answers):
            if isinstance(answer_data, dict) and 'text' in answer_data:
                gt_text = answer_data['text']
            else:
                gt_text = str(answer_data)
            
            em = float(pred.lower().strip() == gt_text.lower().strip())
            exact_matches.append(em)
        
        avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        
        # FIXED: Add batch_size parameter to all log calls
        self.log("val_loss", total_loss, prog_bar=True, batch_size=batch_size)
        self.log("val_em", avg_em, prog_bar=True, batch_size=batch_size)
        
        return total_loss

    def configure_optimizers(self):
        # Get trainable parameters
        trainable_params = [p for p in self.adapter.model.parameters() if p.requires_grad]
        
        print(f"\n🔍 === OPTIMIZER DEBUG ===")
        print(f"Task type: {self.task_type}")
        print(f"Total trainable parameters: {len(trainable_params)}")
        
        if trainable_params:
            total_params = sum(p.numel() for p in trainable_params)
            print(f"Total trainable parameter count: {total_params:,}")
        else:
            print("🚨 CRITICAL ERROR: No trainable parameters found!")
            dummy_param = torch.tensor([0.0], requires_grad=True)
            return torch.optim.AdamW([dummy_param], lr=self.learning_rate)

        # Try to use 8-bit optimizer if available
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.95),
                eps=1e-6,
            )
            print("✅ Using 8-bit PagedAdamW optimizer")
        except ImportError:
            # Fallback to regular AdamW
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.95),
                eps=1e-6,
            )
            print("ℹ️ Using standard AdamW optimizer (install bitsandbytes for memory efficiency)")

        # Configure scheduler
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            max_steps = self.trainer.estimated_stepping_batches
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps
        else:
            max_steps = 1000
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=self.learning_rate * 0.01,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_start(self):
        """Called when training starts"""
        model_name = f"{'QLoRA' if self.use_qlora else 'LoRA'} Llama-2 ({self.task_type})"
        print(f"🚀 Training started with {model_name}")
        
        if torch.cuda.is_available():
            print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 End of epoch - Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
            
    def on_train_end(self):
        """Called when training ends"""
        print(f"\n🏁 === {self.task_type.upper()} TRAINING COMPLETED ===")
        print(f"Total steps completed: {self.trainer.global_step}")
        print(f"Final epoch: {self.trainer.current_epoch}")
        