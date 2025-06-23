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
        return self.adapter(texts)

    def training_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        logits = self(texts)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.adapter.model.parameters(),
            lr=self.learning_rate
        )
    