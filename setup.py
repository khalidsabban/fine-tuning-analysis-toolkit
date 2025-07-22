#!/usr/bin/env python3
"""
Setup script for Llama-2 QLoRA fine-tuning
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_gpu():
    """Check if GPU is available and has sufficient memory"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎯 GPU detected: {gpu_name}")
            print(f"💾 GPU memory: {gpu_memory:.2f} GB")
            
            if gpu_memory < 8:
                print("⚠️  Warning: GPU memory may be insufficient for Llama-2-7B")
                print("   Recommended: At least 8GB GPU memory for QLoRA")
                return False
            else:
                print("✅ GPU memory is sufficient for Llama-2 QLoRA")
                return True
        else:
            print("❌ No GPU detected. Llama-2-7B requires GPU acceleration!")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet, will check GPU after installation")
        return None

def setup_huggingface():
    """Guide user through HuggingFace setup"""
    print("\n🤗 HuggingFace Setup")
    print("="*50)
    print("To use Llama-2, you need to:")
    print("1. Create a HuggingFace account at https://huggingface.co")
    print("2. Request access to Llama-2 at https://huggingface.co/NousResearch/Llama-2-7b-chat-hf")
    print("3. Get your access token from https://huggingface.co/settings/tokens")
    print("4. Run: huggingface-cli login")
    print("\nPress Enter when you've completed these steps...")
    input()

def create_directories():
    """Create necessary directories"""
    directories = [
        "src/toolkit/modules",
        "src/toolkit/scripts", 
        "src/scripts",
        "config",
        "carbon_logs",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def main():
    print("🚀 Setting up Llama-2 QLoRA Fine-tuning Environment")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create directories
    print("\n📁 Creating project structure...")
    create_directories()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️  Some packages failed to install. Try installing manually:")
        print("   pip install torch transformers peft bitsandbytes accelerate")
        
    # Check GPU after installation
    gpu_ok = check_gpu()
    if gpu_ok is False:
        print("\n⚠️  GPU Warning:")
        print("   Llama-2-7B requires significant GPU memory")
        print("   Consider using Google Colab with GPU runtime")
        print("   or a cloud instance with sufficient GPU memory")
    
    # HuggingFace setup
    setup_huggingface()
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "torch",
        "transformers", 
        "peft",
        "bitsandbytes",
        "datasets",
        "pytorch_lightning"
    ]
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Make sure you have HuggingFace access to Llama-2")
    print("2. Run: python src/toolkit/scripts/check_trainable_params.py")
    print("3. Run: python src/scripts/run_experiment.py")
    print("\nFor Google Colab users:")
    print("- Use GPU runtime (Runtime > Change runtime type > GPU)")
    print("- You may need to restart runtime after installing packages")

if __name__ == "__main__":
    main()
    