# Add this to your engine.py where you do post-training evaluation

from toolkit.modules.evaluation_module import evaluate_model

def run_post_training_evaluation(trainer_module, test_dataloader):
    """
    Run evaluation after training with proper device management.
    """
    print("\n🔍 === EVALUATION AFTER TRAINING ===")
    
    try:
        # Get the trained model adapter from the trainer
        model_adapter = trainer_module.adapter
        
        # Ensure model is in eval mode and on correct device
        model_adapter.model.eval()
        
        # Run evaluation
        metrics = evaluate_model(
            model_adapter=model_adapter,
            test_dataloader=test_dataloader,
            max_batches=None  # Set to a number like 10 for quick testing
        )
        
        print("✅ Post-training evaluation completed successfully!")
        return metrics
        
    except Exception as e:
        print(f"⚠️ Post-training evaluation failed: {e}")
        
        # Debug information
        if hasattr(trainer_module, 'adapter') and hasattr(trainer_module.adapter, 'model'):
            model_device = next(trainer_module.adapter.model.parameters()).device
            print(f"🔧 Model device: {model_device}")
            print(f"🔧 Available devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        
        raise e

# Alternative: If you want to integrate directly into your existing evaluation call
def fix_device_evaluation(model_adapter, batch):
    """
    Quick fix for device mismatch in evaluation.
    Use this if you have existing evaluation code.
    """
    # Ensure model is in eval mode
    model_adapter.model.eval()
    
    with torch.no_grad():
        texts = batch["text"]
        labels = batch["label"]
        
        # Get model predictions (ModelAdapter handles input device placement)
        logits = model_adapter(texts)
        
        # CRITICAL: Move labels to same device as logits
        labels = labels.to(logits.device)
        
        # Now compute loss and metrics
        loss = F.cross_entropy(logits, labels)
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate accuracy
        accuracy = (predictions == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'predictions': predictions.cpu().numpy(),
            'labels': labels.cpu().numpy()
        }
    