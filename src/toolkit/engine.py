# File: src/toolkit/engine.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Dump full config for inspection
    print("\n=== Hydra Config ===")
    print(OmegaConf.to_yaml(cfg))

    # Check for expected keys
    if 'model' in cfg and 'name' in cfg.model:
        print(f"Model name: {cfg.model.name}")
    else:
        print("Key 'model.name' not found in config.")

    if 'training' in cfg and 'learning_rate' in cfg.training:
        print(f"Training learning rate: {cfg.training.learning_rate}")
    else:
        print("Key 'training.learning_rate' not found in config.")

if __name__ == "__main__":
    main()
