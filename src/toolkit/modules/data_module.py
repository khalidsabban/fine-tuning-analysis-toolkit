from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from collections import Counter
from typing import Dict, List, Optional, Union

class HFDataModule(pl.LightningDataModule):
    """
    LightningDataModule for any 🤗 Dataset, supporting both classification and QA tasks.
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
        max_length: int = 512,
        
        # Classification parameters
        text_field: str = "sentence",
        label_field: str = "label",
        
        # QA parameters
        question_field: str = "question",
        context_field: str = "context",
        answers_field: str = "answers",
        max_answer_length: int = 30,
        doc_stride: int = 128,
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
        """Setup for classification tasks"""
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
        """Setup for question answering tasks"""
        print(f"Question field: {self.question_field}")
        print(f"Context field: {self.context_field}")
        print(f"Answers field: {self.answers_field}")
        
        # Check first few examples
        print("\n📝 Sample QA examples:")
        for i in range(min(3, len(self.ds))):
            example = self.ds[i]
            question = str(example[self.question_field])[:100] + "..." if len(str(example[self.question_field])) > 100 else str(example[self.question_field])
            
            # Handle context field - might not exist in some datasets
            context = ""
            if self.context_field in example and example[self.context_field]:
                context = str(example[self.context_field])[:150] + "..." if len(str(example[self.context_field])) > 150 else str(example[self.context_field])
            else:
                context = "No context provided"
            
            # Handle different answer formats
            answers = example[self.answers_field]
            if isinstance(answers, dict) and 'text' in answers:
                answer_texts = answers['text'][:3]  # Show first 3 answers
            elif isinstance(answers, list):
                answer_texts = [str(a) for a in answers[:3]]
            else:
                answer_texts = [str(answers)]
                
            print(f"  Example {i}:")
            print(f"    Question: '{question}'")
            print(f"    Context: '{context}'")
            print(f"    Answers: {answer_texts}")
            print()

    def _collate_classification(self, batch):
        """Collate function for classification tasks"""
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
        """Collate function for QA tasks"""
        questions = [str(ex[self.question_field]) for ex in batch]
        
        # Handle context field - might be empty or missing in some datasets like SQL
        contexts = []
        for ex in batch:
            if self.context_field in ex and ex[self.context_field]:
                contexts.append(str(ex[self.context_field]))
            else:
                contexts.append("")  # Empty context for datasets without it
        
        # Extract answer information
        answers_batch = []
        for ex in batch:
            answers = ex[self.answers_field]
            if isinstance(answers, dict):
                # SQuAD format: {"text": [...], "answer_start": [...]}
                answer_texts = answers.get('text', [])
                answer_starts = answers.get('answer_start', [])
                if answer_texts and answer_starts:
                    answers_batch.append({
                        'text': answer_texts[0] if answer_texts else "",
                        'answer_start': answer_starts[0] if answer_starts else 0
                    })
                else:
                    answers_batch.append({'text': "", 'answer_start': 0})
            else:
                # Simple format (like SQL datasets) - just the text answer
                answers_batch.append({
                    'text': str(answers), 
                    'answer_start': 0
                })
        
        # Debug info
        if hasattr(self, '_debug_qa_batch_count'):
            self._debug_qa_batch_count += 1
        else:
            self._debug_qa_batch_count = 1
            
        if self._debug_qa_batch_count <= 2:
            print(f"\n🔍 QA Batch {self._debug_qa_batch_count} debug:")
            print(f"  Batch size: {len(questions)}")
            print(f"  Question lengths: {[len(q) for q in questions]}")
            print(f"  Context lengths: {[len(c) for c in contexts]}")
            print(f"  Sample answer: '{answers_batch[0]['text'] if answers_batch else 'N/A'}'")
        
        return {
            "questions": questions,
            "contexts": contexts,
            "answers": answers_batch
        }

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
    