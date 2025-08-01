import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from toolkit.modules.model_adapter import ModelAdapter
from typing import Dict, List, Union
import math

class TrainerModule(pl.LightningModule):
    """
    Lightning module wrapping ModelAdapter with support for both classification and QA tasks.
    
    MAJOR FIXES FOR QA:
    - Proper handling of tokenized batch data
    - Correct position-based loss calculation
    - Better error handling and validation
    - Fixed device management
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-hf",
        task_type: str = "classification",  # "classification" or "question_answering"
        num_labels: int = 2,  # Only used for classification
        lora_rank: int = 16,
        learning_rate: float = 5e-5,  # FIXED: Increased for QA tasks
        gradient_checkpointing: bool = True,
        use_qlora: bool = True,
        quantization_config: str = "fp4",  # CHANGED: from nf4 to fp4 for speed
        max_length: int = 512,  # INCREASED: was 384
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
        
        # NEW: Track training statistics for QA
        self.qa_training_stats = {
            'total_batches': 0,
            'valid_batches': 0,
            'empty_answer_batches': 0,
            'position_errors': 0
        }
        
        print(f"📊 Model memory footprint: {self.adapter.get_memory_footprint()}")
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
        """Training step for classification tasks - unchanged"""
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
        """
        COMPLETELY REWRITTEN: Training step for QA tasks with proper position handling.
        
        Now works with the processed batch data from the fixed collate function.
        """
        # batch is now a list of processed samples from _collate_qa
        batch_size = len(batch)
        
        # Update statistics
        self.qa_training_stats['total_batches'] += 1
        
        # Separate valid and invalid samples
        valid_samples = [sample for sample in batch if sample.get('is_valid', False)]
        invalid_count = batch_size - len(valid_samples)
        
        if len(valid_samples) == 0:
            # All samples are invalid - skip this batch
            self.qa_training_stats['empty_answer_batches'] += 1
            print(f"⚠️  Batch {batch_idx}: All samples invalid, skipping...")
            
            # Return a proper dummy loss (CHANGED THIS PART)
            dummy_input = torch.randn(1, 10, requires_grad=True, device=self.device)
            dummy_loss = 0.001 * dummy_input.mean()
            
            self.log("train_loss", dummy_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
            return dummy_loss
        
        # Use only valid samples for training
        actual_batch_size = len(valid_samples)
        self.qa_training_stats['valid_batches'] += 1
        
        # Extract tensor data from valid samples
        try:
            # Pad sequences to same length within batch
            max_len_in_batch = max(len(sample['input_ids']) for sample in valid_samples)
            
            input_ids_list = []
            attention_mask_list = []
            start_positions = []
            end_positions = []
            
            for sample in valid_samples:
                # Pad to max length in batch
                input_ids = sample['input_ids'] + [self.adapter.tokenizer.pad_token_id] * (max_len_in_batch - len(sample['input_ids']))
                attention_mask = sample['attention_mask'] + [0] * (max_len_in_batch - len(sample['attention_mask']))
                
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                start_positions.append(sample['start_position'])
                end_positions.append(sample['end_position'])
            
            # Convert to tensors
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=self.device)
            start_positions = torch.tensor(start_positions, dtype=torch.long, device=self.device)
            end_positions = torch.tensor(end_positions, dtype=torch.long, device=self.device)
            
            # Clamp positions to valid range
            seq_len = input_ids.size(1)
            start_positions = torch.clamp(start_positions, 0, seq_len - 1)
            end_positions = torch.clamp(end_positions, 0, seq_len - 1)
            
        except Exception as e:
            self.qa_training_stats['position_errors'] += 1
            print(f"❌ Error processing batch {batch_idx}: {e}")
            
            # Return dummy loss
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log("train_loss", dummy_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
            return dummy_loss
        
        # Debug info for first few steps
        if self.step_count < 5:
            print(f"\n🔍 === QA TRAINING STEP {self.step_count} DEBUG ===")
            print(f"Original batch size: {batch_size}")
            print(f"Valid samples: {actual_batch_size}")
            print(f"Invalid samples: {invalid_count}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Start positions: {start_positions[:3].tolist()}")
            print(f"End positions: {end_positions[:3].tolist()}")
            print(f"Sample question: '{valid_samples[0]['question'][:50]}...'")
            print(f"Sample answer: '{valid_samples[0]['answer']}'")
        
        # Forward pass through the model
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Get hidden states from the base model
        if hasattr(self.adapter.model, 'base_model'):
            # PEFT wrapped model
            base_model = self.adapter.model.base_model.model
        else:
            base_model = self.adapter.model.model
        
        # Get transformer outputs
        transformer_outputs = base_model.model(**model_inputs)
        sequence_output = transformer_outputs.last_hidden_state
        
        # Apply QA head to get start/end logits
        if hasattr(self.adapter.model, 'base_model'):
            qa_head = self.adapter.model.base_model.score
        else:
            qa_head = self.adapter.model.score
            
        start_logits, end_logits = qa_head(sequence_output)
        
        if self.step_count < 5:
            print(f"Start logits shape: {start_logits.shape}")
            print(f"End logits shape: {end_logits.shape}")
        
        # Calculate loss using correct positions
        start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=-1)
        end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=-1)
        total_loss = (start_loss + end_loss) / 2
        
        # Debug loss and predictions
        if self.step_count % 50 == 0:  # Every 50 steps
            with torch.no_grad():
                # Check if model is learning to predict answer positions
                pred_starts = torch.argmax(start_logits, dim=-1)
                pred_ends = torch.argmax(end_logits, dim=-1)
                
                print(f"\n🔍 Step {self.step_count} - Answer Position Analysis:")
                for i in range(min(2, actual_batch_size)):  # First 2 samples
                    sample = valid_samples[i]
                    print(f"\nSample {i}:")
                    print(f"  Question: '{sample['question'][:50]}...'")
                    print(f"  Answer: '{sample['answer']}'")
                    print(f"  True positions: [{start_positions[i].item()}, {end_positions[i].item()}]")
                    print(f"  Pred positions: [{pred_starts[i].item()}, {pred_ends[i].item()}]")
                    
                    # Decode predicted answer
                    if pred_ends[i] >= pred_starts[i]:
                        pred_tokens = input_ids[i][pred_starts[i]:pred_ends[i]+1]
                        pred_answer = self.adapter.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        print(f"  Predicted answer: '{pred_answer}'")
                    
                    # Show logit statistics
                    print(f"  Start logits - min: {start_logits[i].min():.2f}, max: {start_logits[i].max():.2f}, mean: {start_logits[i].mean():.2f}")
                    print(f"  End logits - min: {end_logits[i].min():.2f}, max: {end_logits[i].max():.2f}, mean: {end_logits[i].mean():.2f}")
        
        # Log metrics
        self.step_count += 1
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=actual_batch_size)
        self.log("start_loss", start_loss, on_step=True, on_epoch=True, batch_size=actual_batch_size)
        self.log("end_loss", end_loss, on_step=True, on_epoch=True, batch_size=actual_batch_size)
        self.log("valid_samples_ratio", actual_batch_size / batch_size, on_step=True, batch_size=batch_size)
        
        # Print statistics periodically
        if self.step_count % 100 == 0:
            self._print_training_statistics()
            
            # Check gradients
            total_grad_norm = 0.0
            param_count = 0
            
            for name, param in self.adapter.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_count += 1
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"\n📊 Gradient Stats at step {self.step_count}:")
            print(f"  Total gradient norm: {total_grad_norm:.4f}")
            print(f"  Parameters with gradients: {param_count}")
        
        return total_loss

    def _print_training_statistics(self):
        """NEW: Print training statistics for debugging"""
        stats = self.qa_training_stats
        if stats['total_batches'] > 0:
            valid_pct = (stats['valid_batches'] / stats['total_batches']) * 100
            empty_pct = (stats['empty_answer_batches'] / stats['total_batches']) * 100
            error_pct = (stats['position_errors'] / stats['total_batches']) * 100
            
            print(f"\n📊 === QA TRAINING STATISTICS (Step {self.step_count}) ===")
            print(f"  Total batches: {stats['total_batches']}")
            print(f"  Valid batches: {stats['valid_batches']} ({valid_pct:.1f}%)")
            print(f"  Empty batches: {stats['empty_answer_batches']} ({empty_pct:.1f}%)")
            print(f"  Error batches: {stats['position_errors']} ({error_pct:.1f}%)")
            
            if valid_pct < 80:
                print("  🚨 WARNING: Low valid batch percentage!")

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
        """Validation step for classification - unchanged"""
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
        """
        REWRITTEN: Validation step for QA with proper handling of processed batch data.
        """
        # batch is a list of processed samples
        batch_size = len(batch)
        
        # Filter valid samples
        valid_samples = [sample for sample in batch if sample.get('is_valid', False)]
        
        if len(valid_samples) == 0:
            # Return dummy loss for invalid batch
            dummy_loss = torch.tensor(0.0, device=self.device)
            self.log("val_loss", dummy_loss, prog_bar=True, batch_size=batch_size)
            self.log("val_em", 0.0, prog_bar=True, batch_size=batch_size)
            return dummy_loss
        
        try:
            # Prepare tensors similar to training step
            max_len_in_batch = max(len(sample['input_ids']) for sample in valid_samples)
            
            input_ids_list = []
            attention_mask_list = []
            start_positions = []
            end_positions = []
            
            for sample in valid_samples:
                # Pad sequences
                input_ids = sample['input_ids'] + [self.adapter.tokenizer.pad_token_id] * (max_len_in_batch - len(sample['input_ids']))
                attention_mask = sample['attention_mask'] + [0] * (max_len_in_batch - len(sample['attention_mask']))
                
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                start_positions.append(sample['start_position'])
                end_positions.append(sample['end_position'])
            
            # Convert to tensors
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=self.device)
            start_positions = torch.tensor(start_positions, dtype=torch.long, device=self.device)
            end_positions = torch.tensor(end_positions, dtype=torch.long, device=self.device)
            
            # Clamp positions
            seq_len = input_ids.size(1)
            start_positions = torch.clamp(start_positions, 0, seq_len - 1)
            end_positions = torch.clamp(end_positions, 0, seq_len - 1)
            
            # Forward pass
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Get hidden states
            if hasattr(self.adapter.model, 'base_model'):
                base_model = self.adapter.model.base_model.model
            else:
                base_model = self.adapter.model.model
            
            transformer_outputs = base_model.model(**model_inputs)
            sequence_output = transformer_outputs.last_hidden_state
            
            # Apply QA head
            if hasattr(self.adapter.model, 'base_model'):
                qa_head = self.adapter.model.base_model.score
            else:
                qa_head = self.adapter.model.score
                
            start_logits, end_logits = qa_head(sequence_output)
            
            # Calculate loss
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
            # Calculate approximate exact match for validation
            predicted_starts = torch.argmax(start_logits, dim=-1)
            predicted_ends = torch.argmax(end_logits, dim=-1)
            
            exact_matches = []
            for i in range(len(valid_samples)):
                pred_start = predicted_starts[i].item()
                pred_end = predicted_ends[i].item()
                true_start = start_positions[i].item()
                true_end = end_positions[i].item()
                
                # Simple position-based exact match
                em = float(pred_start == true_start and pred_end == true_end)
                exact_matches.append(em)
            
            avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
            
            # Log metrics
            self.log("val_loss", total_loss, prog_bar=True, batch_size=len(valid_samples))
            self.log("val_em", avg_em, prog_bar=True, batch_size=len(valid_samples))
            self.log("val_valid_samples", len(valid_samples), batch_size=batch_size)
            
            return total_loss
            
        except Exception as e:
            print(f"❌ Validation error in batch {batch_idx}: {e}")
            dummy_loss = torch.tensor(0.0, device=self.device)
            self.log("val_loss", dummy_loss, prog_bar=True, batch_size=batch_size)
            return dummy_loss

    def configure_optimizers(self):
        """Optimizer configuration - enhanced with better settings for QA"""
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
            # IMPROVED: Better optimizer settings for QA tasks
            optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),  # CHANGED: Better for QA tasks
                eps=1e-8,            # CHANGED: More stable
            )
            print("✅ Using 8-bit PagedAdamW optimizer")
        except ImportError:
            # Fallback to regular AdamW
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            print("ℹ️ Using standard AdamW optimizer (install bitsandbytes for memory efficiency)")

        # Configure scheduler with warmup for better QA training
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            max_steps = self.trainer.estimated_stepping_batches
        elif hasattr(self, 'trainer') and self.trainer.max_steps and self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps
        else:
            max_steps = 1000
        
        # NEW: Add warmup for more stable QA training
        warmup_steps = max(100, max_steps // 10)  # 10% warmup, minimum 100 steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate"""
        if self.step_count % 50 == 0:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True)
            
            if self.step_count % 200 == 0:
                print(f"\n📈 Current learning rate: {current_lr:.2e}")

    def on_train_start(self):
        """Called when training starts - enhanced with QA-specific info"""
        model_name = f"{'QLoRA' if self.use_qlora else 'LoRA'} Llama-2 ({self.task_type})"
        print(f"🚀 Training started with {model_name}")
        
        if torch.cuda.is_available():
            print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # NEW: QA-specific warnings
        if self.task_type == "question_answering":
            print(f"📏 Max sequence length: {self.adapter.max_length}")
            print(f"📚 Expected QA format: question + context with proper answer positions")
            print(f"⚠️  Monitor 'valid_samples_ratio' to ensure data quality")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch - enhanced"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 End of epoch - Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
        
        # NEW: Print QA statistics at epoch end
        if self.task_type == "question_answering":
            self._print_training_statistics()
            
    def on_train_end(self):
        """Called when training ends - enhanced with final statistics"""
        print(f"\n🏁 === {self.task_type.upper()} TRAINING COMPLETED ===")
        print(f"Total steps completed: {self.trainer.global_step}")
        print(f"Final epoch: {self.trainer.current_epoch}")
        
        # NEW: Final QA statistics
        if self.task_type == "question_answering":
            stats = self.qa_training_stats
            print(f"\n📊 === FINAL QA TRAINING STATISTICS ===")
            print(f"Total batches processed: {stats['total_batches']}")
            print(f"Valid batches: {stats['valid_batches']}")
            print(f"Empty answer batches: {stats['empty_answer_batches']}")
            print(f"Position error batches: {stats['position_errors']}")
            
            if stats['total_batches'] > 0:
                success_rate = (stats['valid_batches'] / stats['total_batches']) * 100
                print(f"Success rate: {success_rate:.1f}%")
                
                if success_rate < 70:
                    print("🚨 WARNING: Low success rate indicates data quality issues!")
                    print("💡 Recommendations:")
                    print("   - Increase max_length")
                    print("   - Check answer position calculation")
                    print("   - Validate dataset format")
                    