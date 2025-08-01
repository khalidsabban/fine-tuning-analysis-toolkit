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
    """
    Custom QA head that's more memory efficient.
    
    ENHANCED: Better initialization and more stable forward pass.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 outputs: start and end
        
        # NEW: Better initialization for QA tasks
        torch.nn.init.normal_(self.qa_outputs.weight, std=0.02)
        torch.nn.init.zeros_(self.qa_outputs.bias)
        
    def forward(self, hidden_states):
        """
        Forward pass with enhanced stability.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            start_logits: [batch_size, seq_len]
            end_logits: [batch_size, seq_len]
        """
        logits = self.qa_outputs(hidden_states)  # [batch_size, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)  # Each: [batch_size, seq_len, 1]
        start_logits = start_logits.squeeze(-1).contiguous()  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1).contiguous()    # [batch_size, seq_len]
        
        return start_logits, end_logits

class ModelAdapter(nn.Module):
    """
    Wraps a HuggingFace model with optional QLoRA via PEFT and BitsAndBytesConfig,
    supporting both classification and question answering tasks.
    
    MAJOR FIXES FOR QA:
    - Simplified forward pass
    - Better answer extraction
    - Improved memory management
    - Fixed device handling
    """
    def __init__(
        self,
        base_model_name: str = "NousResearch/Llama-2-7b-hf",
        task_type: str = "classification",  # "classification" or "question_answering"
        num_labels: int = 2,  # Only used for classification
        lora_rank: int = 16,
        learning_rate: float = 2e-4,
        use_qlora: bool = True,
        use_safetensors: bool = True,
        quantization_config: str = "fp4",  # fp4 is faster than nf4
        max_length: int = 512,  # INCREASED: from 384
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

        # Load tokenizer with enhanced settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, 
            use_fast=True,  # CHANGED: Use fast tokenizer for better offset mapping
            trust_remote_code=True
        )
        
        # Set pad token for Llama
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # NEW: Verify tokenizer supports required features for QA
        if task_type == "question_answering":
            try:
                # Test if tokenizer supports offset mapping
                test_result = self.tokenizer("test", return_offsets_mapping=True)
                print("✅ Tokenizer supports offset mapping for QA")
            except Exception as e:
                print(f"⚠️  Tokenizer may not support offset mapping: {e}")

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

        # Configure LoRA for Llama-2 with QA-optimized settings
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # EXPANDED: More modules for QA
            bias="none",
            task_type="SEQ_CLS",  # Keep as SEQ_CLS to avoid PEFT issues
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

    def get_memory_footprint(self):
        """Get model memory footprint"""
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        return "N/A (CPU)"

    def __call__(self, inputs: Union[List[str], Dict[str, torch.Tensor]]):
        """
        ENHANCED: Forward pass that handles both classification and QA inputs.
        
        Args:
            inputs: For classification: List[str] of texts
                   For QA: Dict with tensor inputs (input_ids, attention_mask)
        """
        self.model.to(self.device)
        
        if self.task_type == "classification":
            return self._forward_classification(inputs)
        elif self.task_type == "question_answering":
            return self._forward_qa(inputs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _forward_classification(self, texts: List[str]):
        """Forward pass for classification tasks - unchanged"""
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

    def _forward_qa(self, model_inputs: Dict[str, torch.Tensor]):
        """
        COMPLETELY REWRITTEN: Simplified forward pass for QA tasks.
        
        Args:
            model_inputs: Dict with 'input_ids' and 'attention_mask' tensors
            
        Returns:
            QAOutput with start_logits and end_logits
        """
        # Ensure inputs are on correct device
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        # Get hidden states from the base model
        if hasattr(self.model, 'base_model'):
            # PEFT wrapped model
            base_model = self.model.base_model.model
        else:
            base_model = self.model.model
        
        # Forward through transformer
        transformer_outputs = base_model.model(**model_inputs)
        sequence_output = transformer_outputs.last_hidden_state
        
        # Apply QA head
        if hasattr(self.model, 'base_model'):
            qa_head = self.model.base_model.score
        else:
            qa_head = self.model.score
            
        start_logits, end_logits = qa_head(sequence_output)
        
        # Create output object
        class QAOutput:
            def __init__(self, start_logits, end_logits):
                self.start_logits = start_logits
                self.end_logits = end_logits
        
        return QAOutput(start_logits, end_logits)

    def extract_answer(self, qa_inputs: Dict[str, List[str]], max_answer_length: int = 30):
        """
        FIXED: Extract answers from QA model outputs with better logic.
        
        Args:
            qa_inputs: Dict with 'questions' and 'contexts' lists
            max_answer_length: Maximum answer length in tokens
            
        Returns:
            List of extracted answer texts
        """
        if self.task_type != "question_answering":
            raise ValueError("extract_answer only available for QA tasks")
            
        questions = qa_inputs['questions']
        contexts = qa_inputs['contexts']
        
        # Tokenize inputs
        tokenized = self.tokenizer(
            questions,
            contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # Move to device
        model_inputs = {k: v.to(self.device) for k, v in tokenized.items()}
        
        # Get model outputs
        outputs = self(model_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        input_ids = model_inputs['input_ids']
        
        answers = []
        
        for i in range(len(questions)):
            # Find where context starts (after [SEP] token)
            sep_token_id = self.tokenizer.sep_token_id
            input_id_list = input_ids[i].tolist()
            
            # Find first [SEP] token (after question)
            context_start_idx = None
            sep_count = 0
            for idx, token_id in enumerate(input_id_list):
                if token_id == sep_token_id:
                    sep_count += 1
                    if sep_count == 1:  # First [SEP] after question
                        context_start_idx = idx + 1
                        break
            
            if context_start_idx is None:
                # Fallback: estimate based on question length
                question_tokens = self.tokenizer(questions[i], add_special_tokens=False)
                context_start_idx = len(question_tokens['input_ids']) + 2  # +2 for [CLS] and [SEP]
            
            # Only consider logits from context part
            valid_start = min(context_start_idx, len(start_logits[i]) - 1)
            valid_end = len(start_logits[i])
            
            # Get logits only from context portion
            context_start_logits = start_logits[i, valid_start:valid_end]
            context_end_logits = end_logits[i, valid_start:valid_end]
            
            if len(context_start_logits) == 0:
                # No context portion found
                answers.append("")
                continue
            
            # Find best valid answer span
            best_score = float('-inf')
            best_start = valid_start
            best_end = valid_start
            
            # Get top-k start positions
            k = min(10, len(context_start_logits))
            start_scores, start_indices = torch.topk(context_start_logits, k=k)
            
            for j, start_offset in enumerate(start_indices):
                start_idx = start_offset.item() + valid_start
                
                # Look for best end position after this start
                max_end_idx = min(start_idx + max_answer_length, valid_end)
                if max_end_idx > start_idx:
                    end_scores_slice = end_logits[i, start_idx:max_end_idx]
                    if len(end_scores_slice) > 0:
                        end_offset = torch.argmax(end_scores_slice)
                        end_idx = start_idx + end_offset.item()
                        
                        # Calculate combined score
                        score = start_scores[j] + end_scores_slice[end_offset]
                        
                        if score > best_score:
                            best_score = score
                            best_start = start_idx
                            best_end = end_idx
            
            # Extract answer tokens
            if best_end >= best_start and best_start < len(input_ids[i]) and best_end < len(input_ids[i]):
                answer_tokens = input_ids[i][best_start:best_end + 1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Clean the answer
                answer_text = self._clean_extracted_answer(answer_text)
                
                # Validate answer doesn't contain question words
                question_indicators = ['what', 'who', 'when', 'where', 'why', 'how', 'which', '?']
                if any(indicator in answer_text.lower() for indicator in question_indicators):
                    # Try alternative extraction
                    # Get the highest scoring span that doesn't contain question words
                    alternative_found = False
                    
                    for j in range(min(20, len(context_start_logits))):
                        alt_start = torch.topk(context_start_logits, k=j+1)[1][-1].item() + valid_start
                        
                        for length in range(1, min(max_answer_length, valid_end - alt_start)):
                            alt_end = alt_start + length
                            if alt_end < len(input_ids[i]):
                                alt_tokens = input_ids[i][alt_start:alt_end + 1]
                                alt_text = self.tokenizer.decode(alt_tokens, skip_special_tokens=True).strip()
                                
                                if alt_text and not any(ind in alt_text.lower() for ind in question_indicators):
                                    answer_text = alt_text
                                    alternative_found = True
                                    break
                        
                        if alternative_found:
                            break
            else:
                answer_text = ""
            
            answers.append(answer_text)
        
        return answers

    def _clean_extracted_answer(self, answer_text: str) -> str:
        """
        NEW: Clean extracted answer text.
        
        Args:
            answer_text: Raw extracted answer
            
        Returns:
            Cleaned answer text
        """
        # Remove extra whitespace
        answer_text = ' '.join(answer_text.split())
        
        # Remove common artifacts
        answer_text = answer_text.strip('.,;:!? ')
        
        # Remove incomplete words at the end
        if answer_text and answer_text[-1] not in '.!?' and len(answer_text.split()) > 1:
            words = answer_text.split()
            if len(words[-1]) < 3 and not words[-1].isdigit():
                answer_text = ' '.join(words[:-1])
        
        return answer_text
    