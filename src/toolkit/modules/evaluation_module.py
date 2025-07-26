import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm
import string
import re
from collections import Counter

class EvaluationModule:
    """
    Handles model evaluation with support for both classification and QA tasks.
    """
    
    def __init__(self, model_adapter, device: Optional[torch.device] = None):
        self.model_adapter = model_adapter
        self.device = device or self.model_adapter.device
        self.task_type = model_adapter.task_type
        
        # Ensure model is on the correct device
        if hasattr(self.model_adapter, 'model'):
            self.model_adapter.model.to(self.device)
        
        print(f"🔍 Evaluation module initialized for {self.task_type} on device: {self.device}")

    def evaluate_batch(self, batch_data) -> Dict[str, float]:
        """
        Evaluate a single batch of data.
        
        Args:
            batch_data: Batch data (format depends on task type)
            
        Returns:
            Dictionary with batch metrics
        """
        self.model_adapter.model.eval()
        
        with torch.no_grad():
            if self.task_type == "classification":
                return self._evaluate_classification_batch(batch_data)
            elif self.task_type == "question_answering":
                return self._evaluate_qa_batch(batch_data)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

    def _evaluate_classification_batch(self, batch_data) -> Dict[str, float]:
        """Evaluate classification batch"""
        texts = batch_data["text"]
        labels = batch_data["label"].to(self.device)
        
        # Get model predictions
        logits = self.model_adapter(texts)
        
        if logits.device != self.device:
            logits = logits.to(self.device)
        
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        
        # Calculate loss
        loss = F.cross_entropy(logits, labels)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Move to CPU for sklearn metrics
        predictions_cpu = predictions.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(labels_cpu, predictions_cpu)
        f1 = f1_score(labels_cpu, predictions_cpu, average='weighted', zero_division=0)
        precision = precision_score(labels_cpu, predictions_cpu, average='weighted', zero_division=0)
        recall = recall_score(labels_cpu, predictions_cpu, average='weighted', zero_division=0)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': predictions_cpu,
            'labels': labels_cpu
        }

    def _evaluate_qa_batch(self, batch_data) -> Dict[str, float]:
        """Evaluate QA batch"""
        questions = batch_data["questions"]
        contexts = batch_data["contexts"]
        ground_truth_answers = batch_data["answers"]
        
        # Prepare input for model
        qa_inputs = {
            'questions': questions,
            'contexts': contexts
        }
        
        # Get model outputs
        outputs = self.model_adapter(qa_inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Extract predicted answers
        predicted_answers = self.model_adapter.extract_answer(qa_inputs)
        
        # Extract ground truth answer texts
        gt_texts = []
        for answer_data in ground_truth_answers:
            if isinstance(answer_data, dict) and 'text' in answer_data:
                gt_texts.append(answer_data['text'])
            else:
                gt_texts.append(str(answer_data))
        
        # Calculate QA metrics
        exact_matches = []
        f1_scores = []
        
        for pred, gt in zip(predicted_answers, gt_texts):
            # Exact Match
            em = self._compute_exact_match(pred, gt)
            exact_matches.append(em)
            
            # F1 Score
            f1 = self._compute_f1(pred, gt)
            f1_scores.append(f1)
        
        # Calculate average loss (approximation using start/end positions)
        # For QA, we don't have ground truth positions, so we use 0 as placeholder
        dummy_labels = torch.zeros(start_logits.size(0), dtype=torch.long, device=start_logits.device)
        start_loss = F.cross_entropy(start_logits, dummy_labels, reduction='mean')
        end_loss = F.cross_entropy(end_logits, dummy_labels, reduction='mean')
        total_loss = (start_loss + end_loss) / 2
        
        return {
            'loss': total_loss.item(),
            'exact_match': np.mean(exact_matches),
            'f1_score': np.mean(f1_scores),
            'predictions': predicted_answers,
            'ground_truth': gt_texts
        }

    def _normalize_answer(self, s: str) -> str:
        """Normalize answer text for comparison"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match score"""
        return float(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth"""
        pred_tokens = self._normalize_answer(prediction).split()
        gt_tokens = self._normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1

    def evaluate_dataset(self, dataloader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the entire dataset.
        
        Args:
            dataloader: PyTorch DataLoader
            max_batches: Optional limit on number of batches to evaluate
            
        Returns:
            Dictionary with aggregated metrics
        """
        print(f"🔍 Starting {self.task_type} evaluation on device: {self.device}")
        
        self.model_adapter.model.eval()
        self.model_adapter.model.to(self.device)
        
        all_results = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
            
            for batch_idx, batch in enumerate(progress_bar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                try:
                    # Evaluate this batch
                    batch_results = self.evaluate_batch(batch)
                    all_results.append(batch_results)
                    
                    # Update progress bar based on task type
                    if self.task_type == "classification":
                        progress_bar.set_postfix({
                            'loss': f"{batch_results['loss']:.4f}",
                            'acc': f"{batch_results['accuracy']:.3f}",
                            'f1': f"{batch_results['f1_score']:.3f}"
                        })
                    else:  # QA
                        progress_bar.set_postfix({
                            'loss': f"{batch_results['loss']:.4f}",
                            'EM': f"{batch_results['exact_match']:.3f}",
                            'F1': f"{batch_results['f1_score']:.3f}"
                        })
                    
                except Exception as e:
                    print(f"❌ Error evaluating batch {batch_idx}: {e}")
                    raise e
        
        # Calculate final aggregated metrics
        if self.task_type == "classification":
            return self._aggregate_classification_results(all_results)
        else:  # QA
            return self._aggregate_qa_results(all_results)

    def _aggregate_classification_results(self, all_results: List[Dict]) -> Dict[str, float]:
        """Aggregate classification results"""
        all_losses = [r['loss'] for r in all_results]
        all_predictions = []
        all_labels = []
        
        for r in all_results:
            all_predictions.extend(r['predictions'])
            all_labels.extend(r['labels'])
        
        final_metrics = {
            'avg_loss': np.mean(all_losses),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'num_samples': len(all_labels)
        }
        
        return final_metrics

    def _aggregate_qa_results(self, all_results: List[Dict]) -> Dict[str, float]:
        """Aggregate QA results"""
        all_losses = [r['loss'] for r in all_results]
        all_em_scores = []
        all_f1_scores = []
        all_predictions = []
        all_ground_truth = []
        
        for r in all_results:
            # For QA, we need to recalculate metrics from individual predictions
            predictions = r['predictions']
            ground_truth = r['ground_truth']
            
            for pred, gt in zip(predictions, ground_truth):
                em = self._compute_exact_match(pred, gt)
                f1 = self._compute_f1(pred, gt)
                all_em_scores.append(em)
                all_f1_scores.append(f1)
                all_predictions.append(pred)
                all_ground_truth.append(gt)
        
        final_metrics = {
            'avg_loss': np.mean(all_losses),
            'exact_match': np.mean(all_em_scores),
            'f1_score': np.mean(all_f1_scores),
            'num_samples': len(all_predictions)
        }
        
        return final_metrics

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'max_allocated_gb': 0.0}

    def print_evaluation_results(self, metrics: Dict[str, float]):
        """Pretty print evaluation results based on task type"""
        print("\n" + "="*50)
        if self.task_type == "classification":
            print("📊 CLASSIFICATION EVALUATION RESULTS")
            print("="*50)
            print(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"📏 F1 Score:  {metrics['f1_score']:.4f}")
            print(f"🔍 Precision: {metrics['precision']:.4f}")
            print(f"📈 Recall:    {metrics['recall']:.4f}")
            print(f"📉 Avg Loss:  {metrics['avg_loss']:.4f}")
            print(f"📊 Samples:   {metrics['num_samples']}")
        else:  # QA
            print("📊 QUESTION ANSWERING EVALUATION RESULTS")
            print("="*50)
            print(f"🎯 Exact Match: {metrics['exact_match']:.4f}")
            print(f"📏 F1 Score:    {metrics['f1_score']:.4f}")
            print(f"📉 Avg Loss:    {metrics['avg_loss']:.4f}")
            print(f"📊 Samples:     {metrics['num_samples']}")
        
        # Memory usage
        memory = self.get_memory_usage()
        if memory['allocated_gb'] > 0:
            print(f"💾 GPU Memory: {memory['allocated_gb']:.2f}GB allocated, {memory['reserved_gb']:.2f}GB reserved")
        
        print("="*50 + "\n")


def evaluate_model(model_adapter, test_dataloader, max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Convenience function to evaluate a model adapter.
    
    Args:
        model_adapter: ModelAdapter instance
        test_dataloader: DataLoader with test data
        max_batches: Optional limit on batches to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = EvaluationModule(model_adapter)
    metrics = evaluator.evaluate_dataset(test_dataloader, max_batches)
    evaluator.print_evaluation_results(metrics)
    return metrics
