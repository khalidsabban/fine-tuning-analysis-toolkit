import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Union, Optional
import gc

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with optional QLoRA via PEFT and BitsAndBytesConfig,
    supporting both classification and question answering tasks.
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-chat-hf",
        task_type: str = "classification",  # "classification" or "question_answering"
        num_labels: int = 2,  # Only used for classification
        lora_rank: int = 16,
        learning_rate: float = 2e-4,
        use_qlora: bool = True,
        use_safetensors: bool = True,
        quantization_config: str = "nf4",
        max_length: int = 512,
    ):
        super().__init__()
        self.task_type = task_type
        self.max_length = max_length
        self.use_qlora = use_qlora
        self.base_model_name = base_model_name

        # Decide the single device we load onto
        self.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            use_fast=False,
            trust_remote_code=True
        )
        
        # Set pad token for Llama
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Build the model based on task type
        device_str = str(self.device)
        
        if use_qlora:
            # AGGRESSIVE MEMORY OPTIMIZATION
            # Clear any existing cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
            
            # Ultra-aggressive quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # or "fp4" for even more compression
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            # Load with EXTREME memory-efficient settings
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": 0},  # Everything on GPU 0
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
                "offload_folder": "offload",  # Enable CPU offloading
                "offload_state_dict": True,
            }
            
            print("📊 Loading model with 4-bit quantization...")
            
            if task_type == "classification":
                model_kwargs["num_labels"] = num_labels
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name, **model_kwargs
                )
            elif task_type == "question_answering":
                base_model = AutoModelForQuestionAnswering.from_pretrained(
                    base_model_name, **model_kwargs
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Print memory after loading
            if torch.cuda.is_available():
                print(f"📊 Memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # CRITICAL: Set model to eval mode first to save memory
            base_model.eval()
            
            # Enable gradient checkpointing BEFORE prepare_model_for_kbit_training
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                base_model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            
            # Prepare model for training with memory-efficient approach
            try:
                # Try to prepare model
                base_model = prepare_model_for_kbit_training(
                    base_model, 
                    use_gradient_checkpointing=True
                )
            except torch.cuda.OutOfMemoryError:
                # If OOM, try more aggressive approach
                print("⚠️ Initial preparation failed, trying memory-efficient approach...")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Manually prepare model to avoid the float32 conversion
                for name, param in base_model.named_parameters():
                    # Instead of converting to float32, keep in original dtype
                    param.requires_grad = False
                    
                # Only make specific layers trainable
                if hasattr(base_model, 'model'):
                    # For models with a 'model' attribute
                    for name, param in base_model.model.named_parameters():
                        if any(target in name for target in ["norm", "lm_head"]):
                            param.data = param.data.float()
                            param.requires_grad = True
            
            # Enable input gradients after preparation
            if hasattr(base_model, 'enable_input_require_grads'):
                base_model.enable_input_require_grads()
            
            # Clear cache again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"📊 Memory after preparation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
        else:
            # Non-QLoRA path
            raise NotImplementedError("Non-QLoRA path not implemented for memory efficiency")

        # Sync the model's pad_token_id
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Configure LoRA with MINIMAL settings
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,  # Lower alpha for memory efficiency
            target_modules=["q_proj", "v_proj"],  # Minimal modules
            bias="none",
            task_type="SEQ_CLS" if task_type == "classification" else "QUESTION_ANS",
            lora_dropout=0.1,  # Slightly higher dropout
            inference_mode=False,
        )

        # Attach PEFT LoRA with memory monitoring
        if torch.cuda.is_available():
            print(f"📊 Memory before PEFT: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        self.model = get_peft_model(base_model, lora_config)
        
        # Set model to training mode
        self.model.train()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"📊 Memory after PEFT: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        self.model.print_trainable_parameters()

        print(f"✅ Loaded Llama-2 {task_type} model with {'QLoRA' if use_qlora else 'LoRA'} on {device_str}")

    def __call__(self, inputs: Union[List[str], Dict[str, List[str]]]):
        """
        Forward pass that handles both classification and QA inputs.
        
        Args:
            inputs: For classification: List[str] of texts
                   For QA: Dict with 'questions' and 'contexts' keys
        """
        self.model.to(self.device)
        
        if self.task_type == "classification":
            return self._forward_classification(inputs)
        elif self.task_type == "question_answering":
            return self._forward_qa(inputs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _forward_classification(self, texts: List[str]):
        """Forward pass for classification tasks"""
        # Process in smaller chunks if needed
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use gradient checkpointing context if available
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(**inputs)
        
        if outputs.logits.device != self.device:
            outputs.logits = outputs.logits.to(self.device)
            
        return outputs.logits

    def _forward_qa(self, qa_inputs: Dict[str, List[str]]):
        """Forward pass for question answering tasks"""
        questions = qa_inputs['questions']
        contexts = qa_inputs['contexts']
        
        # Tokenize question + context pairs
        inputs = self.tokenizer(
            questions,
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,  # Llama doesn't use token_type_ids
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use gradient checkpointing context if available
        with torch.cuda.amp.autocast(enabled=True):
            outputs = self.model(**inputs)
        
        # QA models return start_logits and end_logits
        if outputs.start_logits.device != self.device:
            outputs.start_logits = outputs.start_logits.to(self.device)
        if outputs.end_logits.device != self.device:
            outputs.end_logits = outputs.end_logits.to(self.device)
            
        return outputs

    def extract_answer(self, qa_inputs: Dict[str, List[str]], max_answer_length: int = 30):
        """
        Extract answers from QA model outputs.
        
        Args:
            qa_inputs: Dict with 'questions' and 'contexts'
            max_answer_length: Maximum answer length in tokens
            
        Returns:
            List of extracted answer texts
        """
        if self.task_type != "question_answering":
            raise ValueError("extract_answer only available for QA tasks")
            
        questions = qa_inputs['questions']
        contexts = qa_inputs['contexts']
        
        # Get model outputs
        outputs = self(qa_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Tokenize to get input_ids for answer extraction
        tokenized = self.tokenizer(
            questions,
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        
        input_ids = tokenized['input_ids']
        offset_mapping = tokenized.get('offset_mapping')
        
        answers = []
        
        for i in range(len(questions)):
            # Get best start and end positions
            start_scores = start_logits[i]
            end_scores = end_logits[i]
            
            # Find the best valid start/end pair
            best_score = float('-inf')
            best_start = 0
            best_end = 0
            
            for start_idx in range(len(start_scores)):
                for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_scores))):
                    score = start_scores[start_idx] + end_scores[end_idx]
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx
            
            # Extract answer tokens
            answer_tokens = input_ids[i][best_start:best_end + 1]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Clean up the answer
            answer_text = answer_text.strip()
            answers.append(answer_text)
        
        return answers

    def get_memory_footprint(self) -> str:
        """Estimate memory usage in MB."""
        total = sum(p.numel() for p in self.model.parameters())
        if self.use_qlora:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            base = total - trainable
            mb = (base * 0.5 + trainable * 4) / (1024**2)
        else:
            mb = (total * 2) / (1024**2)
        return f"~{mb:.1f} MB"
    