# model_adapter.py

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    LlamaForSequenceClassification,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with optional QLoRA via PEFT and BitsAndBytesConfig,
    ensuring the entire model and inputs stay on the same device.
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        num_labels: int = 2,
        lora_rank: int = 16,
        learning_rate: float = 2e-4,
        use_qlora: bool = True,
        use_safetensors: bool = True,
        quantization_config: str = "nf4",
        max_length: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        self.use_qlora   = use_qlora
        self.base_model_name = base_model_name

        # Decide the single device we load onto.
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Load tokenizer - Llama models need special handling
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            use_fast=False,  # Llama tokenizers work better with use_fast=False
            trust_remote_code=True
        )
        
        # Set pad token for Llama (it doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build the model (quantized or fp16) with a one‐to‐one device_map
        device_str = str(self.device)
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                quantization_config=bnb_config,
                device_map={"": device_str},
                trust_remote_code=True,
                use_auth_token=True,  # Required for Llama models
            )
            # Prep for k‑bit QLoRA training
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                torch_dtype=torch.float16,
                device_map={"": device_str},
                trust_remote_code=True,
                use_auth_token=True,  # Required for Llama models
            )

        # Sync the model's pad_token_id
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Configure LoRA for Llama-2 (different target modules than GPT-2)
        if use_qlora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )
        else:
            lora_config = LoraConfig(
                r=max(1, lora_rank // 2),
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )

        # Attach PEFT LoRA and move everything to self.device
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        self.model.print_trainable_parameters()

        # Debug prints
        if use_qlora:
            print(f"✅ Loaded Llama-2 QLoRA 4‑bit ({quantization_config}) on {device_str}")
        else:
            print(f"⚠️  Loaded Llama-2 fp16 model on {device_str}")

    def __call__(self, texts: list[str]):
        # Tokenize on CPU, then move tensors to self.device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward on the unified device
        outputs = self.model(**inputs)
        return outputs.logits

    def get_memory_footprint(self) -> str:
        """Estimate memory usage in MB."""
        total = sum(p.numel() for p in self.model.parameters())
        if self.use_qlora:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            base = total - trainable
            # 4-bit quantization uses ~0.5 bytes per parameter for base model
            mb = (base * 0.5 + trainable * 4) / (1024**2)
        else:
            mb = (total * 2) / (1024**2)
        return f"~{mb:.1f} MB"
    