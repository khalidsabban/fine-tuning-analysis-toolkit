from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

class ModelAdapter:
    """
    Wraps a HuggingFace model with LoRA via PEFT.
    """
    def __init__(
        self,
        base_model_name: str = "sshleifer/tiny-gpt2",
        num_labels: int = 2,
        lora_rank: int = 4,
        use_safetensors: bool = True,
    ):
        # Load tokenizer and base model (safetensors to avoid torch.load vuln)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_safetensors=use_safetensors,
        )
        
        # Set pad_token to eos_token for GPT-2 models that don't have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            use_safetensors=use_safetensors,
        )
        
        # Set the model's pad_token_id to match the tokenizer
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        # Configure LoRA for GPT2: target conv layers c_attn and c_proj
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],
            bias="none",
            task_type="SEQ_CLS",
        )
        # Wrap model
        self.model = get_peft_model(base_model, config)

    def __call__(self, texts: list[str]):
        # Tokenize and forward
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = self.model(**inputs)
        return outputs.logits
    