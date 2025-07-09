import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from toolkit.modules.model_adapter import ModelAdapter

class TrainerModule(pl.LightningModule):
    """
    Lightning module wrapping ModelAdapter to train one step.
    """
    def __init__(
        self,
        base_model_name: str = "sshleifer/tiny-gpt2",
        num_labels: int = 2,
        lora_rank: int = 4,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.adapter = ModelAdapter(
            base_model_name=base_model_name,
            num_labels=num_labels,
            lora_rank=lora_rank
        )
        self.learning_rate = learning_rate

    def forward(self, texts: list[str]):
        # Move the underlying model to the correct device, not the adapter wrapper
        self.adapter.model.to(self.device)
        
        # Forward pass
        logits = self.adapter(texts)
        
        # Ensure logits are on the correct device
        logits = logits.to(self.device)
        
        return logits

    def training_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]

        # Debug: Check initial device states
        #print(f"Original labels device: {labels.device}")
        #print(f"Model device (self.device): {self.device}")
        #print(f"Next model parameter device: {next(self.adapter.model.parameters()).device}")

        # CRITICAL FIX: Move labels to the same device as the model
        labels = labels.to(self.device)
        #print(f"Labels device after move: {labels.device}")

        # Forward pass
        logits = self(texts)
        #print(f"Logits device: {logits.device}")
        #print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.adapter.model.parameters(),
            lr=self.learning_rate
        )

    def on_train_start(self):
        """Called when training starts - move model to correct device"""
        #print(f"Training started. Moving model to device: {self.device}")
        
        # Move only the underlying PyTorch model, not the adapter wrapper
        self.adapter.model.to(self.device)
        
        # Debug device placement
        try:
            actual_device = next(self.adapter.model.parameters()).device
            #print(f"Model parameters are on device: {actual_device}")
        except:
            print("Could not access model parameters")

    def on_train_epoch_start(self):
        """Called at the start of each epoch"""
        #print(f"Epoch starting. Model device: {self.device}")
        # Move the underlying model to correct device
        self.adapter.model.to(self.device)
        
        # Debug: Check if model is actually on the right device
        try:
            actual_device = next(self.adapter.model.parameters()).device
            #print(f"Adapter model actual device after move: {actual_device}")
        except:
            print("Could not check adapter model device")
