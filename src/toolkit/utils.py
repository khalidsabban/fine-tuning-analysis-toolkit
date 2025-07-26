# src/toolkit/utils.py

import random
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional
import os
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to {seed}")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    print(f"📝 Logging setup complete (level: {log_level})")


def get_model_info(model) -> Dict[str, Any]:
    """
    Get comprehensive model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    # Add device information
    if hasattr(model, 'device'):
        info['device'] = str(model.device)
    elif next(model.parameters(), None) is not None:
        info['device'] = str(next(model.parameters()).device)
    else:
        info['device'] = 'unknown'
    
    return info


def print_model_info(model, model_name: str = "Model"):
    """
    Pretty print model information
    """
    info = get_model_info(model)
    
    print(f"\n📊 === {model_name.upper()} INFO ===")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Trainable percentage: {info['trainable_percentage']:.2f}%")
    print(f"Estimated size: {info['model_size_mb']:.1f} MB")
    print(f"Device: {info['device']}")
    print("=" * 30)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information if available
    """
    info = {'available': torch.cuda.is_available()}
    
    if info['available']:
        info['device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name()
        
        props = torch.cuda.get_device_properties(0)
        info['total_memory_gb'] = props.total_memory / (1024 ** 3)
        info['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
        info['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        
    return info


def print_gpu_info():
    """
    Pretty print GPU information
    """
    info = get_gpu_info()
    
    print(f"\n🎯 === GPU INFO ===")
    if info['available']:
        print(f"CUDA Available: ✅")
        print(f"Device: {info['device_name']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Allocated: {info['memory_allocated_gb']:.2f} GB")
        print(f"Reserved: {info['memory_reserved_gb']:.2f} GB")
        print(f"Free: {info['total_memory_gb'] - info['memory_allocated_gb']:.2f} GB")
    else:
        print(f"CUDA Available: ❌")
    print("=" * 20)


def cleanup_memory():
    """
    Clean up GPU memory
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cleaned up")


def save_experiment_config(config: Dict[str, Any], save_path: str):
    """
    Save experiment configuration to file
    """
    import json
    from omegaconf import OmegaConf
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    if hasattr(config, '_content'):  # OmegaConf DictConfig
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"💾 Experiment config saved to {save_path}")


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from file
    """
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"📁 Experiment config loaded from {config_path}")
    return config


def format_time(seconds: float) -> str:
    """
    Format time duration in human readable format
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_experiment_name(config) -> str:
    """
    Generate a unique experiment name based on config
    """
    import datetime
    
    task_type = getattr(config, 'task', {}).get('type', 'unknown')
    model_name = getattr(config, 'model', {}).get('name', 'unknown').split('/')[-1]
    dataset_name = getattr(config, 'data', {}).get('dataset_name', 'unknown')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{task_type}_{model_name}_{dataset_name}_{timestamp}"
