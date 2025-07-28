import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict, Union, Optional
import gc

class QAHead(nn.Module):
    """Custom QA head that's more memory efficient"""
    def __init__(self, hidden_size):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 outputs: start and end
        
    def forward(self, hidden_states):
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits

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
            # Clear any existing cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # More aggressive quantization config for memory efficiency
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quantization_config,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # Load with memory-efficient settings
            model_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": 0},
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }
            
            # ALWAYS load as classification model first (uses less memory)
            model_kwargs["num_labels"] = 2  # Dummy value
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, **model_kwargs
            )
            
            # Enable gradient checkpointing BEFORE prepare_model_for_kbit_training
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                base_model.gradient_checkpointing_enable()
            
            # Prepare model for training
            base_model = prepare_model_for_kbit_training(base_model)
            
            # Enable input gradients after preparation
            if hasattr(base_model, 'enable_input_require_grads'):
                base_model.enable_input_require_grads()
                
            # Now modify for QA if needed
            if task_type == "question_answering":
                # Replace the classification head with a QA head
                hidden_size = base_model.config.hidden_size
                base_model.score = QAHead(hidden_size)
                # Make sure the new head is trainable
                for param in base_model.score.parameters():
                    param.requires_grad = True
                print("✅ Replaced classification head with QA head")
                
        else:
            # Non-QLoRA path
            raise NotImplementedError("Non-QLoRA path not implemented")

        # Sync the model's pad_token_id
        base_model.config.pad_token_id = self.tokenizer.pad_token_id

        # Configure LoRA for Llama-2 with more conservative settings
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="SEQ_CLS",  # Always use SEQ_CLS to avoid issues
            lora_dropout=0.05,
        )

        # Attach PEFT LoRA with memory monitoring
        if torch.cuda.is_available():
            print(f"📊 Memory before PEFT: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
        self.model = get_peft_model(base_model, lora_config)
        
        # Clear cache after model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
            return_token_type_ids=False,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # For QA, we need to get the hidden states directly
        # First, get the base model (without the classification/QA head)
        if hasattr(self.model, 'base_model'):
            # PEFT wrapped model
            base_model = self.model.base_model.model
        else:
            base_model = self.model.model
        
        # Get the transformer outputs directly
        transformer_outputs = base_model.model(**inputs, output_hidden_states=True)
        sequence_output = transformer_outputs.last_hidden_state
        
        # Apply our custom QA head
        if hasattr(self.model, 'base_model'):
            # For PEFT models
            qa_head = self.model.base_model.score
        else:
            qa_head = self.model.score
            
        start_logits, end_logits = qa_head(sequence_output)
        
        # Create a simple namespace to match expected output format
        class QAOutput:
            def __init__(self, start_logits, end_logits):
                self.start_logits = start_logits
                self.end_logits = end_logits
        
        return QAOutput(start_logits, end_logits)

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
        
        input_ids = tokenized['input_ids'].to(self.device)
        
        answers = []
        
        for i in range(len(questions)):
            # Get best start and end positions
            start_scores = start_logits[i]
            end_scores = end_logits[i]
            
            # Find the best valid start/end pair
            best_score = float('-inf')
            best_start = 0
            best_end = 0
            
            # Get top k start and end indices
            start_top_k = torch.topk(start_scores, k=min(20, len(start_scores)))
            end_top_k = torch.topk(end_scores, k=min(20, len(end_scores)))
            
            for start_idx in start_top_k.indices:
                for end_idx in end_top_k.indices:
                    if start_idx <= end_idx and end_idx - start_idx < max_answer_length:
                        score = start_scores[start_idx] + end_scores[end_idx]
                        if score > best_score:
                            best_score = score
                            best_start = start_idx.item()
                            best_end = end_idx.item()
            
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
    