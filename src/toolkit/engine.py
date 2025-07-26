# src/toolkit/engine.py

import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from toolkit.modules.carbon_tracker import CarbonTracker
from toolkit.modules.data_module import HFDataModule
from toolkit.modules.trainer_module import TrainerModule
from toolkit.modules.evaluation_module import EvaluationModule


class ExperimentEngine:
    """
    Orchestrates the entire experimental workflow for both classification and QA tasks.
    Handles data loading, model initialization, training, and evaluation.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = config.task.type
        self.carbon_tracker = None
        self.data_module = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        print(f"🎯 Experiment Engine initialized for {self.task_type} task")
        
    def setup_carbon_tracking(self):
        """Initialize carbon tracking"""
        self.carbon_tracker = CarbonTracker(
            project_name=self.config.carbon.tracker.project_name,
            output_dir=self.config.carbon.tracker.output_dir,
        )
        print("🌱 Carbon tracking initialized")
        
    def setup_data(self):
        """Setup data module based on task type"""
        print("📚 Setting up data module...")
        
        common_params = {
            'task_type': self.task_type,
            'dataset_name': self.config.data.dataset_name,
            'split': self.config.data.split,
            'batch_size': self.config.training.batch_size,
            'num_workers': self.config.data.num_workers,
            'max_length': self.config.data.get('max_length', 512),
            'val_split_ratio': self.config.data.get('val_split_ratio', 0.1),
        }
        
        if self.task_type == "classification":
            task_params = {
                'text_field': self.config.data.text_field,
                'label_field': self.config.data.label_field,
            }
        elif self.task_type == "question_answering":
            task_params = {
                'question_field': self.config.data.question_field,
                'context_field': self.config.data.context_field,
                'answers_field': self.config.data.answers_field,
                'max_answer_length': self.config.qa.get('max_answer_length', 30),
                'doc_stride': self.config.qa.get('doc_stride', 128),
            }
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
        self.data_module = HFDataModule(**common_params, **task_params)
        self.data_module.setup()
        print(f"✅ Data module setup complete for {self.task_type}")
        
    def setup_model(self):
        """Setup model based on task type"""
        print("🤖 Setting up model...")
        
        model_params = {
            'base_model_name': self.config.model.name,
            'task_type': self.task_type,
            'lora_rank': self.config.model.lora_rank,
            'learning_rate': self.config.training.learning_rate,
            'gradient_checkpointing': self.config.training.get('gradient_checkpointing', True),
            'use_qlora': self.config.model.get('use_qlora', True),
            'quantization_config': self.config.model.get('quantization_config', 'nf4'),
            'max_length': self.config.data.get('max_length', 512),
        }
        
        if self.task_type == "classification":
            model_params['num_labels'] = self.config.model.get('num_labels', 2)
            
        self.model = TrainerModule(**model_params)
        print(f"✅ Model setup complete for {self.task_type}")
        
    def setup_trainer(self):
        """Setup PyTorch Lightning trainer"""
        print("⚡ Setting up trainer...")
        
        self.trainer = pl.Trainer(
            max_steps=self.config.training.max_steps,
            max_epochs=self.config.training.max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            precision=16 if self.config.training.get('use_mixed_precision', True) else 32,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm',
            accumulate_grad_batches=self.config.training.get('gradient_accumulation_steps', 1),
            deterministic=False,
            benchmark=True,
            log_every_n_steps=5,
            detect_anomaly=False,
        )
        print("✅ Trainer setup complete")
        
    def setup_evaluator(self):
        """Setup evaluation module"""
        self.evaluator = EvaluationModule(self.model.adapter)
        print(f"📊 Evaluator setup complete for {self.task_type}")
        
    def run_pre_training_evaluation(self) -> Dict[str, Any]:
        """Run evaluation before training"""
        print(f"\n🔍 === PRE-TRAINING EVALUATION ({self.task_type.upper()}) ===")
        
        self.model.eval()
        results = {}
        
        try:
            if self.task_type == "classification":
                results = self._evaluate_classification_samples(
                    self.config.eval.get('classification_samples', 
                                       self.config.eval.get('samples', []))
                )
            elif self.task_type == "question_answering":
                results = self._evaluate_qa_samples(
                    self.config.eval.get('qa_samples', [])
                )
        except Exception as e:
            print(f"⚠️  Pre-training evaluation failed: {e}")
            
        return results
        
    def run_training(self):
        """Execute the training process"""
        print(f"\n🚀 === STARTING {self.task_type.upper()} TRAINING ===")
        
        try:
            self.trainer.fit(self.model, self.data_module)
            print("✅ Training completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
            
    def run_post_training_evaluation(self) -> Dict[str, Any]:
        """Run evaluation after training"""
        print(f"\n🔍 === POST-TRAINING EVALUATION ({self.task_type.upper()}) ===")
        
        self.model.eval()
        results = {}
        
        try:
            if self.task_type == "classification":
                results = self._evaluate_classification_samples(
                    self.config.eval.get('classification_samples', 
                                       self.config.eval.get('samples', []))
                )
            elif self.task_type == "question_answering":
                results = self._evaluate_qa_samples(
                    self.config.eval.get('qa_samples', [])
                )
                
            # Also run full validation set evaluation if available
            if self.data_module.val_ds is not None:
                val_dataloader = self.data_module.val_dataloader()
                if val_dataloader:
                    val_metrics = self.evaluator.evaluate_dataset(val_dataloader, max_batches=10)
                    self.evaluator.print_evaluation_results(val_metrics)
                    results['validation_metrics'] = val_metrics
                    
        except Exception as e:
            print(f"⚠️  Post-training evaluation failed: {e}")
            
        return results
        
    def _evaluate_classification_samples(self, samples):
        """Evaluate classification samples"""
        if not samples:
            return {}
            
        import torch.nn.functional as F
        
        with torch.no_grad():
            logits = self.model(samples)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            print("📊 Classification results:")
            results = []
            for i, (text, prob, pred) in enumerate(zip(samples, probs, preds)):
                result = {
                    'text': text,
                    'probabilities': prob.cpu().tolist(),
                    'prediction': pred.item(),
                    'predicted_label': 'Positive' if pred.item() == 1 else 'Negative'
                }
                results.append(result)
                
                print(f"  Text: '{text[:50]}...'")
                print(f"  Probs: [neg: {prob[0]:.3f}, pos: {prob[1]:.3f}]")
                print(f"  Prediction: {result['predicted_label']}")
                print()
                
        return {'sample_results': results}
        
    def _evaluate_qa_samples(self, qa_samples):
        """Evaluate QA samples"""
        if not qa_samples:
            return {}
            
        questions = [sample['question'] for sample in qa_samples]
        contexts = [sample['context'] for sample in qa_samples]
        
        qa_inputs = {
            'questions': questions,
            'contexts': contexts
        }
        
        with torch.no_grad():
            predicted_answers = self.model.adapter.extract_answer(qa_inputs)
            
            print("📊 QA results:")
            results = []
            for question, context, pred_answer in zip(questions, contexts, predicted_answers):
                result = {
                    'question': question,
                    'context': context,
                    'predicted_answer': pred_answer
                }
                results.append(result)
                
                print(f"  Question: '{question}'")
                print(f"  Context: '{context[:100]}...'")
                print(f"  Predicted Answer: '{pred_answer}'")
                print()
                
        return {'sample_results': results}
        
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline"""
        print(f"🚀 Starting full {self.task_type} experiment pipeline")
        
        # Setup all components
        self.setup_carbon_tracking()
        self.carbon_tracker.start()
        
        experiment_results = {}
        
        try:
            self.setup_data()
            self.setup_model()
            self.setup_trainer()
            self.setup_evaluator()
            
            # Pre-training evaluation
            pre_results = self.run_pre_training_evaluation()
            experiment_results['pre_training'] = pre_results
            
            # Training
            training_success = self.run_training()
            experiment_results['training_success'] = training_success
            
            if training_success:
                # Post-training evaluation
                post_results = self.run_post_training_evaluation()
                experiment_results['post_training'] = post_results
                
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            experiment_results['error'] = str(e)
            raise
        finally:
            # Stop carbon tracking
            if self.carbon_tracker:
                self.carbon_tracker.stop()
                
        print(f"🎉 {self.task_type.capitalize()} experiment completed!")
        return experiment_results


def run_experiment_from_config(config: DictConfig) -> Dict[str, Any]:
    """
    Convenience function to run experiment from Hydra config
    """
    engine = ExperimentEngine(config)
    return engine.run_full_experiment()
