# File: src/toolkit/engine.py

# Suppress NumPy warnings (e.g., “mean of empty slice”) throughout the pipeline
import numpy as np
np.seterr(all="ignore")


import hydra
from omegaconf import DictConfig, OmegaConf
from toolkit.modules.carbon_tracker import CarbonTracker

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Start CO2 tracking
    carbon = CarbonTracker(project_name=cfg.get("carbon", {}).get("project_name", "exp"))
    carbon.start()

    # Check for expected keys
    if 'model' in cfg and 'name' in cfg.model:
        print(f"Model name: {cfg.model.name}")
    else:
        print("Key 'model.name' not found in config.")

    if 'training' in cfg and 'learning_rate' in cfg.training:
        print(f"Training learning rate: {cfg.training.learning_rate}")
    else:
        print("Key 'training.learning_rate' not found in config.")

    print("\n=== Hydra Config ===")
    print(OmegaConf.to_yaml(cfg))

    # (Training / evaluation pipeline here)

    # Stop CO2 tracking
    carbon.stop()

if __name__ == "__main__":
    main()
