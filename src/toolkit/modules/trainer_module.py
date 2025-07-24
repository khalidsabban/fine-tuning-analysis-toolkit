import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from toolkit.modules.model_adapter import ModelAdapter

class TrainerModule(pl.LightningModule):
    """
    Lightning module wrapping ModelAdapter with QLoRA support for Llama-2.
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        num_labels: int = 2,
        lora_rank: int = 16,
        learning_rate: float = 5e-5,  # 🔧 FIXED: Reduced from 2e-4
        gradient_checkpointing: bool = True,
        use_qlora: bool = True,
        quantization_config: str = "nf4",
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.adapter = ModelAdapter(
            base_model_name=base_model_name,
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
        
        if gradient_checkpointing and hasattr(self.adapter.model, 'gradient_checkpointing_enable'):
            self.adapter.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")

    def forward(self, texts: list[str]):
        return self.adapter(texts)

    def training_step(self, batch, batch_idx):
        texts = batch["sentence"]
        labels = batch["label"]
        
        # Debug info for first few steps
        if self.step_count < 5:
            print(f"\n🔍 === TRAINING STEP {self.step_count} DEBUG ===")
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
        
        # 🔧 FIXED: Enhanced gradient debugging
        if self.step_count < 10:
            print(f"\n🚨 LOSS DEBUG - Step {self.step_count}:")
            print(f"  Raw loss: {loss.item()}")
            print(f"  Loss requires_grad: {loss.requires_grad}")
            
            # Check if all labels are the same
            unique_labels_in_batch = torch.unique(labels)
            if len(unique_labels_in_batch) == 1:
                print(f"  🚨 PROBLEM: All labels in batch are {unique_labels_in_batch[0].item()}")
            
            # Check logits distribution
            probs = F.softmax(logits, dim=-1)
            print(f"  Probabilities range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            print(f"  Mean probabilities: {probs.mean(dim=0)}")
            
            # 🔧 IMPROVED: Better gradient checking
            trainable_params = [p for p in self.adapter.model.parameters() if p.requires_grad]
            if trainable_params:
                print(f"  Trainable parameters found: {len(trainable_params)}")
                
                # Force a backward pass to check gradients
                if loss.requires_grad:
                    # Store current gradients (if any)
                    grad_norms_before = []
                    for p in trainable_params[:3]:  # Check first 3 params
                        if p.grad is not None:
                            grad_norms_before.append(p.grad.norm().item())
                        else:
                            grad_norms_before.append(0.0)
                    
                    print(f"  Gradient norms (first 3 params): {grad_norms_before}")
                    
                    # Check if gradients are flowing by examining parameter values
                    param_means = [p.data.mean().item() for p in trainable_params[:3]]
                    print(f"  Parameter means (first 3): {param_means}")
            else:
                print("  🚨 CRITICAL: No trainable parameters found!")
        
        self.step_count += 1
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Memory monitoring
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            self.log("memory_used_gb", memory_used)
            self.log("memory_reserved_gb", memory_reserved)
            
            if batch_idx % 100 == 0:
                print(f"Step {batch_idx}: Memory used: {memory_used:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        return loss

    def configure_optimizers(self):
        # 🔧 IMPROVED: Better optimizer configuration
        trainable_params = [p for p in self.adapter.model.parameters() if p.requires_grad]
        
        print(f"\n🔍 === OPTIMIZER DEBUG ===")
        print(f"Total trainable parameters: {len(trainable_params)}")
        
        if trainable_params:
            total_params = sum(p.numel() for p in trainable_params)
            print(f"Total trainable parameter count: {total_params:,}")
            print(f"First few parameter shapes: {[p.shape for p in trainable_params[:3]]}")
            
            # Check parameter initialization
            param_stats = []
            for i, p in enumerate(trainable_params[:3]):
                mean_val = p.data.mean().item()
                std_val = p.data.std().item()
                param_stats.append(f"Param {i}: mean={mean_val:.6f}, std={std_val:.6f}")
            print("Parameter statistics:")
            for stat in param_stats:
                print(f"  {stat}")
                
        else:
            print("🚨 CRITICAL ERROR: No trainable parameters found!")
            print("   This means LoRA adapters are not properly configured.")
            all_params = list(self.adapter.model.parameters())
            print(f"   Total parameters in model: {len(all_params)}")
            for i, p in enumerate(all_params[:5]):
                print(f"   Param {i}: shape={p.shape}, requires_grad={p.requires_grad}")
            
            # Return a dummy optimizer to prevent crashes
            dummy_param = torch.tensor([0.0], requires_grad=True)
            return torch.optim.AdamW([dummy_param], lr=self.learning_rate)

        # 🔧 FIXED: Improved optimizer settings for QLoRA
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=0.01,  # Small weight decay for stability
            betas=(0.9, 0.95),  # Better betas for transformer training
            eps=1e-6,  # Smaller epsilon for better numerical stability
        )

        # 🔧 FIXED: Better scheduler configuration
        if hasattr(self.trainer, 'estimated_stepping_batches') and self.trainer.estimated_stepping_batches:
            max_steps = self.trainer.estimated_stepping_batches
        elif self.trainer.max_steps and self.trainer.max_steps > 0:
            max_steps = self.trainer.max_steps
        else:
            # Fallback calculation
            max_steps = 1000
            
        print(f"Scheduler max_steps: {max_steps}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=self.learning_rate * 0.01,  # Lower minimum LR
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
        model_name = "QLoRA Llama-2" if self.use_qlora else "Standard LoRA Llama-2"
        print(f"🚀 Training started with {model_name}")
        
        if torch.cuda.is_available():
            print(f"🎯 Using GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        print(f"\n🔍 === TRAINING CONFIG DEBUG ===")
        print(f"Max steps: {self.trainer.max_steps}")
        print(f"Max epochs: {self.trainer.max_epochs}")
        print(f"Current epoch: {self.trainer.current_epoch}")
        print(f"Global step: {self.trainer.global_step}")
        
        # 🔧 ADDED: Verify model is in training mode
        print(f"Model training mode: {self.adapter.model.training}")

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
            
        texts = batch["sentence"]
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

    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            current_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"📊 End of epoch - Current: {current_memory:.2f}GB, Peak: {peak_memory:.2f}GB")
            
    def on_train_end(self):
        """Called when training ends"""
        print(f"\n🏁 === TRAINING COMPLETED ===")
        print(f"Total steps completed: {self.trainer.global_step}")
        print(f"Final epoch: {self.trainer.current_epoch}")
        
        # 🔧 ADDED: Final parameter check
        trainable_params = [p for p in self.adapter.model.parameters() if p.requires_grad]
        if trainable_params:
            final_grad_norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in trainable_params[:3]]
            print(f"Final gradient norms (first 3 params): {final_grad_norms}")
            