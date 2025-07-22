# evaluation_module.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Any, Optional
import numpy as np
from tqdm import tqdm

class EvaluationModule:
    """
    Handles model evaluation with proper device management for QLoRA/LoRA models.
    Ensures all tensors are on the same device during evaluation.
    """
    
    def __init__(self, model_adapter, device: Optional[torch.device] = None):
        """
        Initialize evaluation module.
        
        Args:
            model_adapter: The ModelAdapter instance containing the model
            device: Target device (if None, uses model's device)
        """
        self.model_adapter = model_adapter
        self.device = device or self.model_adapter.device
        
        # Ensure model is on the correct device
        if hasattr(self.model_adapter, 'model'):
            self.model_adapter.model.to(self.device)
        
        print(f"🔍 Evaluation module initialized on device: {self.device}")

    def evaluate_batch(self, texts: List[str], labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate a single batch of data.
        
        Args:
            texts: List of input texts
            labels: Ground truth labels tensor
            
        Returns:
            Dictionary with batch metrics
        """
        # Ensure model is in eval mode
        self.model_adapter.model.eval()
        
        with torch.no_grad():
            try:
                # Get model predictions (ModelAdapter handles device placement for inputs)
                logits = self.model_adapter(texts)
                
                # Ensure labels are on the same device as logits
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
                
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"❌ Device mismatch error in evaluation: {e}")
                    print(f"Model device: {next(self.model_adapter.model.parameters()).device}")
                    print(f"Labels device: {labels.device}")
                    # Try to fix by moving everything to the model's device
                    model_device = next(self.model_adapter.model.parameters()).device
                    labels = labels.to(model_device)
                    logits = self.model_adapter(texts)
                    raise e
                else:
                    raise e

    def evaluate_dataset(self, dataloader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the entire dataset.
        
        Args:
            dataloader: PyTorch DataLoader with batches of {'text': [...], 'label': tensor}
            max_batches: Optional limit on number of batches to evaluate
            
        Returns:
            Dictionary with aggregated metrics
        """
        print(f"🔍 Starting evaluation on device: {self.device}")
        
        all_losses = []
        all_predictions = []
        all_labels = []
        
        # Ensure model is in eval mode
        self.model_adapter.model.eval()
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
            
            for batch_idx, batch in enumerate(progress_bar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                try:
                    texts = batch["text"]
                    labels = batch["label"]
                    
                    # Evaluate this batch
                    batch_results = self.evaluate_batch(texts, labels)
                    
                    # Collect results
                    all_losses.append(batch_results['loss'])
                    all_predictions.extend(batch_results['predictions'])
                    all_labels.extend(batch_results['labels'])
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{batch_results['loss']:.4f}",
                        'acc': f"{batch_results['accuracy']:.3f}",
                        'f1': f"{batch_results['f1_score']:.3f}"
                    })
                    
                except Exception as e:
                    print(f"❌ Error evaluating batch {batch_idx}: {e}")
                    print(f"Batch text types: {type(texts)}, length: {len(texts) if isinstance(texts, list) else 'N/A'}")
                    print(f"Batch label shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
                    print(f"Batch label device: {labels.device if hasattr(labels, 'device') else 'N/A'}")
                    raise e
        
        # Calculate final aggregated metrics
        final_metrics = {
            'avg_loss': np.mean(all_losses),
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'num_samples': len(all_labels)
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
        """Pretty print evaluation results."""
        print("\n" + "="*50)
        print("📊 EVALUATION RESULTS")
        print("="*50)
        print(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"📏 F1 Score:  {metrics['f1_score']:.4f}")
        print(f"🔍 Precision: {metrics['precision']:.4f}")
        print(f"📈 Recall:    {metrics['recall']:.4f}")
        print(f"📉 Avg Loss:  {metrics['avg_loss']:.4f}")
        print(f"📊 Samples:   {metrics['num_samples']}")
        
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
