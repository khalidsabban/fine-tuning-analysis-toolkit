import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from toolkit.modules.model_adapter import ModelAdapter

class TrainerModule(pl.LightningModule):
    """
    Lightning module wrapping ModelAdapter with QLoRA support.
    """
    def __init__(
        self,
        base_model_name: str = "gpt2",
        num_labels: int = 2,
        lora_rank: int = 16,
        learning_rate: float = 2e-4,
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
        
        # Print memory footprint
        print(f"📊 Model memory footprint: {self.adapter.get_memory_footprint()}")
        
        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(self.adapter.model, 'gradient_checkpointing_enable'):
            self.adapter.model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")

    def forward(self, texts: list[str]):
        # For QLoRA, the model is already optimally placed
        """
        if not self.use_qlora:
            self.adapter.model.to(self.device) """

        # Force model to CUDA for evaluation (QLoRA models can have mixed devices)
        devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to GPU
        self.adapter.model.to(devices)
        
        # DEBUG: Print device info
        print(f"🔍 DEBUG: self.device = {self.device}")
        print(f"🔍 DEBUG: Model device = {next(self.adapter.model.parameters()).device}")
        
        # Tokenize inputs
        tokenized = self.adapter.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # DEBUG: Print tokenized tensor devices before moving
        print(f"🔍 DEBUG: Tokenized tensors before device move:")
        for k, v in tokenized.items():
            print(f"  {k}: {v.device}")
        
        # Move each tensor to GPU
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        # DEBUG: Print tokenized tensor devices after moving
        print(f"🔍 DEBUG: Tokenized tensors after device move:")
        for k, v in tokenized.items():
            print(f"  {k}: {v.device}")
    
        # Forward pass through the model
        outputs = self.adapter.model(**tokenized)
        
        # Return logits
        return outputs.logits

    def training_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Log memory usage periodically
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            self.log("memory_used_gb", memory_used)
            self.log("memory_reserved_gb", memory_reserved)
        
        return loss

    def configure_optimizers(self):
        # For QLoRA, we typically use AdamW with specific settings
        if self.use_qlora:
            # Only optimize the trainable parameters (LoRA adapters)
            trainable_params = [p for p in self.adapter.model.parameters() if p.requires_grad]

            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=0.0,  # No weight decay for QLoRA
                betas=(0.9, 0.999),
                eps=1e-8,
            )

            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_steps,
                eta_min=self.learning_rate * 0.1,
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            # Standard optimizer for regular LoRA
            return torch.optim.AdamW(
                self.adapter.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01,
            )

    def on_train_start(self):
        """Called when training starts"""
        print("🚀 Training started with QLoRA" if self.use_qlora else "🚀 Training started with standard LoRA")

    def validation_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return loss
