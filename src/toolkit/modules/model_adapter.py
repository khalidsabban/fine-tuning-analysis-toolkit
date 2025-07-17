# model_adapter.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn as nn

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with QLoRA via PEFT and BitsAndBytesConfig.
    """
    def __init__(
        self,
        base_model_name: str = "gpt2",
        num_labels: int = 2,
        lora_rank: int = 16,         # Can use higher rank with QLoRA
        use_qlora: bool = True,
        use_safetensors: bool = True,
        max_length: int = 512,       # Can handle longer sequences with QLoRA
    ):
        super().__init__()
        self.max_length = max_length
        self.use_qlora = use_qlora

        # Determine device map: load entire model onto a single device
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
            # 4-bit quantization settings
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            # Load model with 4-bit quantization onto one device
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            # Prepare for k-bit training
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            # Standard (fp16) loading onto one device
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                torch_dtype=torch.float16,
                device_map=device_map,
            )

        # Ensure pad token is defined
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Configure LoRA adapter
        if use_qlora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,                 # Higher alpha OK with QLoRA
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )
        else:
            lora_config = LoraConfig(
                r=lora_rank // 2,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )

        # Attach LoRA to the base model
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # Log info
        if use_qlora:
            print("✅ Using QLoRA with 4-bit quantization")
            print(f"📊 Model loaded in 4-bit with compute dtype: {torch.float16}")
        else:
            print("⚠️  Using standard LoRA (higher memory usage)")

    def __call__(self, texts: list[str]):
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        return outputs.logits

    def get_memory_footprint(self):
        """Get approximate memory footprint of the model"""
        if hasattr(self.model, "get_memory_footprint"):
            return self.model.get_memory_footprint()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            if self.use_qlora:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                base_params = total_params - trainable
                # 0.5 bytes per base-parameter in 4-bit + 4 bytes per LoRA-parameter
                memory_mb = (base_params * 0.5 + trainable * 4) / (1024 * 1024)
            else:
                # 2 bytes per parameter in fp16
                memory_mb = (total_params * 2) / (1024 * 1024)
            return f"~{memory_mb:.1f} MB"
