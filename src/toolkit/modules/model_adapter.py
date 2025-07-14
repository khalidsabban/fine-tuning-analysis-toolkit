from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
import torch

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with QLoRA via PEFT and BitsAndBytesConfig.
    """
    def __init__(
        self,
        base_model_name: str = "gpt2",
        num_labels: int = 2,
        lora_rank: int = 16,  # Can use higher rank with QLoRA
        use_safetensors: bool = True,
        max_length: int = 512,  # Can handle longer sequences with QLoRA
        use_qlora: bool = True,
        quantization_config: str = "nf4",  # or "fp4"
    ):
        super().__init__()
        self.max_length = max_length
        self.use_qlora = use_qlora
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_safetensors=use_safetensors,
        )
        
        # Set pad_token to eos_token for GPT-2 models that don't have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for QLoRA
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config,  # "nf4" or "fp4"
                bnb_4bit_use_double_quant=True,  # Nested quantization for additional memory savings
                bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16 for speed
            )
            
            # Load model with 4-bit quantization
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                quantization_config=bnb_config,
                device_map="auto",  # Automatically distribute across available devices
                trust_remote_code=True,
            )
            
            # Prepare model for k-bit training
            base_model = prepare_model_for_kbit_training(base_model)
            
        else:
            # Standard loading without quantization
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                use_safetensors=use_safetensors,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        
        # Set the model's pad_token_id to match the tokenizer
        base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Configure LoRA - can use higher rank with QLoRA due to memory savings
        if use_qlora:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,  # Can keep higher alpha with QLoRA
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )
        else:
            # More conservative settings for standard LoRA
            config = LoraConfig(
                r=lora_rank // 2,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                bias="none",
                task_type="SEQ_CLS",
                lora_dropout=0.1,
            )
        
        # Apply LoRA adapter
        self.model = get_peft_model(base_model, config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Print quantization info
        if use_qlora:
            print(f"✅ Using QLoRA with {quantization_config} quantization")
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
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        return outputs.logits

    def get_memory_footprint(self):
        """Get approximate memory footprint of the model"""
        if hasattr(self.model, 'get_memory_footprint'):
            return self.model.get_memory_footprint()
        else:
            # Approximate calculation
            total_params = sum(p.numel() for p in self.model.parameters())
            if self.use_qlora:
                # 4-bit quantization uses ~0.5 bytes per parameter for base model
                # + full precision for LoRA adapters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                base_params = total_params - trainable_params
                memory_mb = (base_params * 0.5 + trainable_params * 4) / (1024 * 1024)
            else:
                # Standard 16-bit uses ~2 bytes per parameter
                memory_mb = (total_params * 2) / (1024 * 1024)
            return f"~{memory_mb:.1f} MB"
        