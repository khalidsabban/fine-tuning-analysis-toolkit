#!/usr/bin/env python3
# File: src/scripts/run_qa_experiment.py

"""
Simple script to run QA experiments using the question_answering.yaml config
"""

import hydra
from omegaconf import DictConfig
from toolkit.engine import run_experiment_from_config

@hydra.main(config_path="../../config", config_name="question_answering", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run QA experiment using the unified engine
    """
    print("🤖 Starting Question Answering Experiment")
    print("="*50)
    
    # Verify this is a QA experiment
    if cfg.task.type != "question_answering":
        print(f"❌ Error: Expected QA task, got {cfg.task.type}")
        print("   Make sure you're using the question_answering.yaml config")
        return
    
    # Run the experiment
    try:
        results = run_experiment_from_config(cfg)
        
        print("\n📊 === EXPERIMENT SUMMARY ===")
        print(f"Task: {cfg.task.type}")
        print(f"Dataset: {cfg.data.dataset_name}")
        print(f"Model: {cfg.model.name}")
        print(f"Training Success: {results.get('training_success', False)}")
        
        if 'post_training' in results and 'validation_metrics' in results['post_training']:
            val_metrics = results['post_training']['validation_metrics']
            print(f"Final Exact Match: {val_metrics.get('exact_match', 'N/A'):.4f}")
            print(f"Final F1 Score: {val_metrics.get('f1_score', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
    