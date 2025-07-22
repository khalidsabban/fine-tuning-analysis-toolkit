# src/toolkit/scripts/check_trainable_params.py

import hydra
from omegaconf import DictConfig
from toolkit.modules.trainer_module import TrainerModule

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """
    Counts how many trainable parameters your PEFT-wrapped Llama-2 model has.
    """
    print(f"🔍 Checking trainable parameters for {cfg.model.name}")
    print(f"📊 LoRA rank: {cfg.model.lora_rank}")
    print(f"🔧 Using QLoRA: {cfg.model.get('use_qlora', True)}")

    try:
        # Note: config.yaml uses `model.name`, not `model.base_model_name`
        model = TrainerModule(
            base_model_name=cfg.model.name,
            num_labels=cfg.model.num_labels,
            lora_rank=cfg.model.lora_rank,
            learning_rate=cfg.training.learning_rate,
            use_qlora=cfg.model.get('use_qlora', True),
            quantization_config=cfg.model.get('quantization_config', 'nf4'),
        )

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.adapter.model.parameters())
        trainable_params = sum(p.numel() for p in model.adapter.model.parameters() if p.requires_grad)
        
        print(f"\n📈 === PARAMETER ANALYSIS ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Memory estimation
        if cfg.model.get('use_qlora', True):
            # 4-bit quantized base model + full precision LoRA adapters
            base_params = total_params - trainable_params
            estimated_memory_gb = (base_params * 0.5 + trainable_params * 4) / (1024**3)
            print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB (with 4-bit quantization)")
        else:
            # Full precision model
            estimated_memory_gb = (total_params * 2) / (1024**3)
            print(f"Estimated memory usage: {estimated_memory_gb:.2f} GB (full precision)")
        
        if trainable_params == 0:
            raise RuntimeError(
                "❌ No trainable parameters found! Check your LoRA configuration."
            )
        else:
            print(f"✅ Success! Found {trainable_params:,} trainable parameters")
            
        # Print target modules for verification
        print(f"\n🎯 LoRA target modules:")
        if hasattr(model.adapter.model, 'peft_config'):
            for name, config in model.adapter.model.peft_config.items():
                print(f"   {config.target_modules}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "token" in str(e).lower():
            print("\n💡 Authentication error detected:")
            print("   1. Make sure you have access to the Llama-2 model")
            print("   2. Accept the license agreement on HuggingFace")
            print("   3. Set up your HuggingFace token with: huggingface-cli login")
        raise

if __name__ == "__main__":
    main()
    