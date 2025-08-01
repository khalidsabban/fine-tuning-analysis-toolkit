from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from collections import Counter
from typing import Dict, List, Optional, Union

class HFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for any 🤗 Dataset, supporting both classification and QA tasks.
    
    MAJOR FIXES FOR QA:
    - Proper tokenization with offset mapping
    - Correct answer position calculation
    - Validation of answer spans
    - Better handling of truncated contexts
    """
    def __init__(
        self,
        # Task configuration
        task_type: str = "classification",  # "classification" or "question_answering"
        
        # Common parameters
        dataset_name: str = "sst2",
        split: str = "train",
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
        val_split_ratio: float = 0.1,
        max_length: int = 512,  # INCREASED: was 384, now 512 for better QA performance
        
        # Classification parameters
        text_field: str = "sentence",
        label_field: str = "label",
        
        # QA parameters
        question_field: str = "question",
        context_field: str = "context",
        answers_field: str = "answers",
        max_answer_length: int = 30,
        doc_stride: int = 128,
        
        # NEW: Tokenizer parameter for QA position calculation
        tokenizer = None,
    ):
        super().__init__()
        self.task_type = task_type
        
        # Common
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_split_ratio = val_split_ratio
        self.max_length = max_length
        
        # Classification
        self.text_field = text_field
        self.label_field = label_field
        
        # QA
        self.question_field = question_field
        self.context_field = context_field
        self.answers_field = answers_field
        self.max_answer_length = max_answer_length
        self.doc_stride = doc_stride
        
        # NEW: Store tokenizer for proper QA processing
        self.tokenizer = tokenizer
        
        # NEW: Track statistics for debugging
        self.qa_stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'truncated_answers': 0,
            'empty_contexts': 0,
            'position_errors': 0,
        }

    def setup(self, stage=None):
        # Load the dataset
        self.ds = load_dataset(path=self.dataset_name, split=self.split)
        
        print(f"\n🔍 === DATASET DEBUG INFO ===")
        print(f"Task type: {self.task_type}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Dataset size: {len(self.ds)}")
        
        if self.task_type == "classification":
            self._setup_classification()
        elif self.task_type == "question_answering":
            self._setup_qa()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Split into train/validation
        if self.val_split_ratio > 0:
            split_idx = int(len(self.ds) * (1 - self.val_split_ratio))
            self.train_ds = self.ds.select(range(split_idx))
            self.val_ds = self.ds.select(range(split_idx, len(self.ds)))
            
            print(f"\n📊 Data splits:")
            print(f"  Training: {len(self.train_ds)} samples")
            print(f"  Validation: {len(self.val_ds)} samples")
        else:
            self.train_ds = self.ds
            self.val_ds = None
            print(f"\n📊 Training samples: {len(self.train_ds)}")

    def _setup_classification(self):
        """Setup for classification tasks - unchanged"""
        print(f"Text field: {self.text_field}")
        print(f"Label field: {self.label_field}")
        
        # Check first few examples
        print("\n📝 Sample examples:")
        for i in range(min(3, len(self.ds))):
            example = self.ds[i]
            text_preview = str(example[self.text_field])[:100] + "..." if len(str(example[self.text_field])) > 100 else str(example[self.text_field])
            print(f"  Example {i}: Text='{text_preview}', Label={example[self.label_field]}")
        
        # Check label distribution
        all_labels = [ex[self.label_field] for ex in self.ds]
        label_counts = Counter(all_labels)
        print(f"\n📊 Label distribution: {dict(label_counts)}")
        
        if len(label_counts) == 1:
            print("🚨 CRITICAL ERROR: All labels are the same value!")
        elif min(label_counts.values()) / max(label_counts.values()) < 0.1:
            print("⚠️  Warning: Very imbalanced labels detected")

    def _setup_qa(self):
        """Setup for question answering tasks - enhanced with validation"""
        print(f"Question field: {self.question_field}")
        print(f"Context field: {self.context_field}")
        print(f"Answers field: {self.answers_field}")
        print(f"Max length: {self.max_length}")  # NEW: Show max length
        
        # NEW: Validate that tokenizer is available for QA
        if self.tokenizer is None:
            print("⚠️  Warning: No tokenizer provided. Position calculation will be approximate.")
        
        # Check first few examples and validate data quality
        print("\n📝 Sample QA examples:")
        context_lengths = []
        
        for i in range(min(3, len(self.ds))):
            example = self.ds[i]
            question = str(example[self.question_field])[:100] + "..." if len(str(example[self.question_field])) > 100 else str(example[self.question_field])
            
            # Handle context field - might not exist in some datasets
            context = ""
            if self.context_field in example and example[self.context_field]:
                context = str(example[self.context_field])
                context_lengths.append(len(context))
                context_preview = context[:150] + "..." if len(context) > 150 else context
            else:
                context_preview = "No context provided"
            
            # Handle different answer formats
            answers = example[self.answers_field]
            if isinstance(answers, dict) and 'text' in answers:
                answer_texts = answers['text'][:3]  # Show first 3 answers
                answer_starts = answers.get('answer_start', [])
            elif isinstance(answers, list):
                answer_texts = [str(a) for a in answers[:3]]
                answer_starts = []
            else:
                answer_texts = [str(answers)]
                answer_starts = []
                
            print(f"  Example {i}:")
            print(f"    Question: '{question}'")
            print(f"    Context: '{context_preview}'")
            print(f"    Answers: {answer_texts}")
            if answer_starts:
                print(f"    Answer starts: {answer_starts[:3]}")
            
            # NEW: Check if answer exists in context
            if context and answer_texts and answer_texts[0]:
                if answer_texts[0] in context:
                    print(f"    ✅ Answer found in context")
                else:
                    print(f"    ❌ Answer NOT found in context")
            print()
        
        # NEW: Analyze context lengths
        if context_lengths:
            avg_length = sum(context_lengths) / len(context_lengths)
            max_length_sample = max(context_lengths)
            print(f"\n📊 Context length analysis:")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Max length in sample: {max_length_sample} characters")
            print(f"  Estimated tokens (÷4): ~{avg_length/4:.0f} avg, ~{max_length_sample/4:.0f} max")
            
            if max_length_sample > self.max_length * 4:  # Rough character-to-token ratio
                print(f"  ⚠️  Warning: Some contexts may be truncated with max_length={self.max_length}")

    def _find_answer_positions(self, question: str, context: str, answer_text: str, 
                             answer_start_char: int = 0) -> tuple:
        """
        FIXED: Find proper token positions for answer spans using tokenizer.
        
        Args:
            question: The question text
            context: The context text  
            answer_text: The ground truth answer text
            answer_start_char: Character start position of answer in context
            
        Returns:
            (start_token_idx, end_token_idx, is_valid) tuple
        """
        if not self.tokenizer or not answer_text or not context:
            return 0, 0, False
        
        try:
            # First tokenize question alone to understand structure
            question_tokens = self.tokenizer(question, add_special_tokens=False)
            question_length = len(question_tokens['input_ids'])
            
            # Tokenize question and context together with offset mapping
            tokenized = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_offsets_mapping=True,
                add_special_tokens=True
            )
            
            offset_mapping = tokenized['offset_mapping']
            
            # Find where context starts in the tokenized sequence
            # Look for the [SEP] token after question
            sep_token_id = self.tokenizer.sep_token_id
            input_ids = tokenized['input_ids']
            
            context_start_token = None
            for idx, token_id in enumerate(input_ids):
                if token_id == sep_token_id:
                    # Context typically starts after first [SEP]
                    context_start_token = idx + 1
                    break
            
            if context_start_token is None:
                # Fallback: estimate based on question length
                context_start_token = question_length + 2  # +2 for [CLS] and [SEP]
            
            # The answer_start_char is relative to the context only
            # We need to find the token that corresponds to this character position
            
            start_token_idx = None
            end_token_idx = None
            
            # Search for answer positions only in the context part
            for idx in range(context_start_token, len(offset_mapping)):
                token_start, token_end = offset_mapping[idx]
                
                if token_start is None or token_end is None:
                    continue  # Skip special tokens
                
                # offset_mapping for context tokens are relative to the full input
                # We need to adjust for the fact that answer_start_char is relative to context only
                
                # Check if this token overlaps with answer start
                if start_token_idx is None:
                    # The offset includes the question + separator, so we need to check differently
                    # We'll use a heuristic: if the answer text appears in the decoded tokens
                    if idx < len(input_ids):
                        # Decode tokens from this position
                        test_tokens = input_ids[idx:min(idx + len(answer_text.split()), len(input_ids))]
                        decoded = self.tokenizer.decode(test_tokens, skip_special_tokens=True).strip()
                        
                        if answer_text.lower() in decoded.lower():
                            start_token_idx = idx
                            # Find end position
                            for end_idx in range(idx, min(idx + self.max_answer_length, len(input_ids))):
                                end_test_tokens = input_ids[idx:end_idx+1]
                                end_decoded = self.tokenizer.decode(end_test_tokens, skip_special_tokens=True).strip()
                                
                                if answer_text.lower() == end_decoded.lower():
                                    end_token_idx = end_idx
                                    break
                                elif len(end_decoded) > len(answer_text) * 1.5:
                                    # Stop if we've gone too far
                                    break
                            
                            if end_token_idx is None and start_token_idx is not None:
                                # Estimate end position
                                end_token_idx = start_token_idx + max(1, len(answer_text.split()) - 1)
            
            # Validate positions
            if start_token_idx is not None and end_token_idx is not None:
                if start_token_idx >= context_start_token and end_token_idx >= start_token_idx:
                    if end_token_idx - start_token_idx <= self.max_answer_length:
                        return start_token_idx, end_token_idx, True
            
            # If we couldn't find exact match, try character-based approach
            if start_token_idx is None:
                # This is a fallback - not as accurate but better than nothing
                char_to_token_ratio = len(input_ids) / len(question + " " + context)
                estimated_start = context_start_token + int(answer_start_char * char_to_token_ratio)
                estimated_end = estimated_start + max(1, int(len(answer_text) * char_to_token_ratio))
                
                if estimated_start < len(input_ids) and estimated_end < len(input_ids):
                    return estimated_start, estimated_end, False  # Mark as not valid but usable
            
            self.qa_stats['position_errors'] += 1
            return 0, 0, False
            
        except Exception as e:
            print(f"⚠️  Error finding positions for answer '{answer_text}': {e}")
            self.qa_stats['position_errors'] += 1
            return 0, 0, False

    def validate_qa_data(self, num_samples=5):
        """FIXED: Validate QA data quality"""
        print("\n🔍 === QA DATA VALIDATION ===")
        
        if not hasattr(self, 'train_ds'):
            print("⚠️  Dataset not set up yet")
            return
        
        for i in range(min(num_samples, len(self.train_ds))):
            example = self.train_ds[i]
            
            question = example[self.question_field]
            context = example[self.context_field] if self.context_field in example else ""
            answers = example[self.answers_field]
            
            # Extract answer text and position
            if isinstance(answers, dict):
                answer_text = answers['text'][0] if answers.get('text') else ""
                answer_start = answers['answer_start'][0] if answers.get('answer_start') else 0
            else:
                answer_text = str(answers)
                answer_start = context.find(answer_text) if context else -1
            
            print(f"\nSample {i}:")
            print(f"  Question: {question}")
            print(f"  Answer: '{answer_text}'")
            print(f"  Answer start: {answer_start}")
            
            # Verify answer is in context
            if answer_start >= 0 and context:
                extracted = context[answer_start:answer_start + len(answer_text)]
                match = extracted == answer_text
                print(f"  Answer in context: {'✅' if match else '❌'}")
                if not match:
                    print(f"    Expected: '{answer_text}'")
                    print(f"    Found: '{extracted}'")
            else:
                print(f"  Answer in context: ❌ (not found)")
            
            # Tokenize and check positions
            if self.tokenizer:
                start_pos, end_pos, is_valid = self._find_answer_positions(
                    question, context, answer_text, answer_start
                )
                print(f"  Token positions: [{start_pos}, {end_pos}] - Valid: {'✅' if is_valid else '❌'}")
                
                # Decode the tokens at those positions to verify
                if is_valid:
                    tokenized = self.tokenizer(question, context, truncation=True, max_length=self.max_length)
                    if start_pos < len(tokenized['input_ids']) and end_pos < len(tokenized['input_ids']):
                        answer_tokens = tokenized['input_ids'][start_pos:end_pos+1]
                        decoded_answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                        print(f"  Decoded from tokens: '{decoded_answer}'")

    def _collate_classification(self, batch):
        """Collate function for classification tasks - unchanged"""
        texts = [str(ex[self.text_field]) for ex in batch]
        labels = torch.tensor([int(ex[self.label_field]) for ex in batch], dtype=torch.long)
        
        # Debug info
        if hasattr(self, '_debug_batch_count'):
            self._debug_batch_count += 1
        else:
            self._debug_batch_count = 1
            
        if self._debug_batch_count <= 2:
            print(f"\n🔍 Classification Batch {self._debug_batch_count} debug:")
            print(f"  Batch size: {len(texts)}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Text lengths: {[len(t) for t in texts]}")
        
        return {"text": texts, "label": labels}

    def _collate_qa(self, batch):
        """
        FIXED: Proper collate function for QA tasks with correct position calculation.
        
        This is the core fix that addresses the main training issues.
        """
        # Extract raw data from batch
        questions = [str(ex[self.question_field]) for ex in batch]
        contexts = []
        raw_answers = []
        
        # Handle context field - might be empty or missing
        for ex in batch:
            if self.context_field in ex and ex[self.context_field]:
                contexts.append(str(ex[self.context_field]))
            else:
                contexts.append("")  # Empty context for datasets without it
        
        # Extract answer information
        for ex in batch:
            answers = ex[self.answers_field]
            if isinstance(answers, dict):
                # SQuAD format: {"text": [...], "answer_start": [...]}
                answer_texts = answers.get('text', [])
                answer_starts = answers.get('answer_start', [])
                if answer_texts and answer_starts:
                    raw_answers.append({
                        'text': answer_texts[0] if answer_texts else "",
                        'answer_start': answer_starts[0] if answer_starts else 0
                    })
                else:
                    raw_answers.append({'text': "", 'answer_start': 0})
            else:
                # Simple format - try to find answer in context
                answer_text = str(answers)
                answer_start = 0
                # Try to find the answer position in context
                if answer_text and contexts[-1]:
                    found_pos = contexts[-1].find(answer_text)
                    if found_pos != -1:
                        answer_start = found_pos
                
                raw_answers.append({
                    'text': answer_text,
                    'answer_start': answer_start
                })
        
        # Process each sample and calculate proper token positions
        processed_samples = []
        valid_count = 0
        
        for i, (question, context, answer_data) in enumerate(zip(questions, contexts, raw_answers)):
            # Update statistics
            self.qa_stats['total_samples'] += 1
            
            if not context:
                self.qa_stats['empty_contexts'] += 1
            
            # Tokenize without truncation first to check length
            if self.tokenizer:
                full_tokenized = self.tokenizer(
                    question,
                    context,
                    add_special_tokens=True,
                    return_offsets_mapping=False,
                    padding=False
                )
                
                # Check if we'll need to truncate
                full_length = len(full_tokenized['input_ids'])
                will_truncate = full_length > self.max_length
                
                if will_truncate:
                    # Check if answer will be preserved after truncation
                    answer_char_end = answer_data['answer_start'] + len(answer_data['text'])
                    # Rough estimate: if answer is in the latter half of context, it might be truncated
                    context_char_start = len(question) + 10  # Approximate
                    relative_answer_pos = (answer_data['answer_start'] - context_char_start) / len(context) if context else 0
                    
                    if relative_answer_pos > 0.7:  # Answer is in latter part of context
                        self.qa_stats['truncated_answers'] += 1
                
                # Tokenize with proper settings for position finding
                tokenized = self.tokenizer(
                    question,
                    context,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True
                )
                
                # Find answer token positions
                answer_text = answer_data['text']
                char_start = answer_data['answer_start']
                
                start_token_idx, end_token_idx, is_valid = self._find_answer_positions(
                    question, context, answer_text, char_start
                )
                
                if is_valid:
                    valid_count += 1
                    self.qa_stats['valid_samples'] += 1
                
                # Create processed sample
                processed_sample = {
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'start_position': start_token_idx,
                    'end_position': end_token_idx,
                    'question': question,
                    'context': context,
                    'answer': answer_text,
                    'is_valid': is_valid,
                    'original_length': full_length,
                    'was_truncated': will_truncate
                }
                
            else:
                # Fallback when no tokenizer available - create dummy data
                print("⚠️  No tokenizer available - using dummy positions")
                processed_sample = {
                    'input_ids': [],
                    'attention_mask': [],
                    'start_position': 0,
                    'end_position': 0,
                    'question': question,
                    'context': context,
                    'answer': answer_data['text'],
                    'is_valid': False,
                    'original_length': 0,
                    'was_truncated': False
                }
            
            processed_samples.append(processed_sample)
        
        # Debug info
        if hasattr(self, '_debug_qa_batch_count'):
            self._debug_qa_batch_count += 1
        else:
            self._debug_qa_batch_count = 1
            
        if self._debug_qa_batch_count <= 2:
            print(f"\n🔍 QA Batch {self._debug_qa_batch_count} debug:")
            print(f"  Batch size: {len(processed_samples)}")
            print(f"  Valid samples: {valid_count}/{len(processed_samples)}")
            print(f"  Question lengths: {[len(q) for q in questions]}")
            print(f"  Context lengths: {[len(c) for c in contexts]}")
            print(f"  Sample answer: '{processed_samples[0]['answer'] if processed_samples else 'N/A'}'")
            
            if processed_samples and processed_samples[0]['is_valid']:
                sample = processed_samples[0]
                print(f"  Token positions: {sample['start_position']}-{sample['end_position']}")
                print(f"  Sequence length: {len(sample['input_ids'])}")
                print(f"  Was truncated: {sample['was_truncated']}")
        
        # Print statistics every 10 batches
        if hasattr(self, '_debug_qa_batch_count') and self._debug_qa_batch_count % 10 == 0:
            self._print_qa_statistics()
        
        return processed_samples

    def _print_qa_statistics(self):
        """NEW: Print QA processing statistics for debugging"""
        stats = self.qa_stats
        if stats['total_samples'] > 0:
            valid_pct = (stats['valid_samples'] / stats['total_samples']) * 100
            truncated_pct = (stats['truncated_answers'] / stats['total_samples']) * 100
            empty_pct = (stats['empty_contexts'] / stats['total_samples']) * 100
            
            print(f"\n📊 === QA PROCESSING STATISTICS ===")
            print(f"  Total samples processed: {stats['total_samples']}")
            print(f"  Valid samples: {stats['valid_samples']} ({valid_pct:.1f}%)")
            print(f"  Truncated answers: {stats['truncated_answers']} ({truncated_pct:.1f}%)")
            print(f"  Empty contexts: {stats['empty_contexts']} ({empty_pct:.1f}%)")
            
            if valid_pct < 50:
                print("  🚨 WARNING: Less than 50% of samples are valid!")
                print("  💡 Consider: increasing max_length, checking data quality")
            elif truncated_pct > 20:
                print("  ⚠️  Warning: High truncation rate detected")
                print("  💡 Consider: increasing max_length or using sliding window")

    def _collate(self, batch):
        """Main collate function that dispatches based on task type"""
        if self.task_type == "classification":
            return self._collate_classification(batch)
        elif self.task_type == "question_answering":
            return self._collate_qa(batch)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate,
        )
    
    def set_tokenizer(self, tokenizer):
        """NEW: Set tokenizer after initialization for proper QA processing"""
        self.tokenizer = tokenizer
        print(f"✅ Tokenizer set for QA position calculation")
        