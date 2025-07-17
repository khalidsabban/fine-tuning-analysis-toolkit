# model_adapter.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn as nn

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with optional QLoRA via PEFT and BitsAndBytesConfig.
    """
    def __init__(
        self,
        base_model_name: str = "gpt2",
        num_labels: int = 2,
        lora_rank: int = 16,
        use_qlora: bool = True,
        use_safetensors: bool = True,
        quantization_config: str = "nf4",  # now accepted from your Hydra config
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        self.use_qlora = use_qlora

        # Determine a single-device map so everything stays on cuda:0 (or CPU)
        device_map = (
            {"": f"cuda:{torch.cuda.current_device()}"} 
            if torch.cuda.is_available() 
            else {"": "cpu"}
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
        )

        if use_qlora:
            # Build 4-bit quantization config using the passed string
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            # Load the base model in 4-bit onto one device
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Prepare for k-bit training (QLoRA)
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            # Standard (fp16) load onto one device
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

        # Make sure pad token is defined
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Configure LoRA adapter
        if use_qlora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )
        else:
            lora_config = LoraConfig(
                r=max(1, lora_rank // 2),
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )

        # Attach LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # Log summary
        if use_qlora:
            print(f"✅ Using QLoRA with 4-bit quantization ({quantization_config})")
        else:
            print("⚠️  Using standard LoRA (fp16) — higher memory usage")

    def __call__(self, texts: list[str]):
        # Tokenize (remains on CPU so HF's device_map can stream shards)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Move tokens to the model's device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward
        outputs = self.model(**inputs)
        return outputs.logits

    def get_memory_footprint(self):
        """Estimate memory usage in MB."""
        total = sum(p.numel() for p in self.model.parameters())
        if self.use_qlora:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            base = total - trainable
            mb = (base * 0.5 + trainable * 4) / (1024**2)
        else:
            mb = (total * 2) / (1024**2)
        return f"~{mb:.1f} MB"
