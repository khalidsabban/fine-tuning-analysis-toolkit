# fine-tuning-analysis-toolkit

A batteries-included research toolkit for fine-tuning and evaluating language models with LoRA/QLoRA, Hydra-powered configuration, PyTorch Lightning training, MLflow experiment tracking, and CodeCarbon energy accounting. Comes with a tiny dashboard, clean CLI scripts, and unit tests.

> If you're skimming: **Install**, **Run a quick experiment**, **Track with MLflow**, **See your carbon footprint**, **Open the dashboard**. Tweak everything via `config/`.

---

## Features
- **Config-first** with **Hydra/ΩmegaConf** (`config/`) for reproducible experiments and easy overrides.
- **LoRA/QLoRA** adapters via a pluggable `model_adapter` (PEFT-style wrappers).
- **PyTorch Lightning** trainer wrapper for clean training loops and callbacks.
- **Evaluation** module with Accuracy, F1, and **Exact Match** (EM) + resource stats.
- **CodeCarbon** integration for energy/CO₂ tracking, logged per run.
- **MLflow** tracking for params, metrics, artifacts, and run comparison.
- **Scripts** for: run experiment, analyze results, and launch a simple dashboard.
- **Docker** and **docker-compose** for consistent local/dev runs (incl. MLflow server).
- **Unit tests** for key modules.

---

## Project Structure
your-toolkit/
├── .vscode/
│ ├── settings.json
│ ├── launch.json
│ └── extensions.json
├── config/
│ ├── config.yaml # base config (classification or QA)
│ ├── question_answering.yaml # QA-focused overrides
│ ├── lora.yaml # (optional) LoRA presets
│ ├── qlora.yaml # (optional) QLoRA presets
│ └── carbon.yaml # CodeCarbon settings
├── src/
│ ├── toolkit/
│ │ ├── init.py
│ │ ├── engine.py
│ │ ├── modules/
│ │ │ ├── data_module.py
│ │ │ ├── model_adapter.py
│ │ │ ├── trainer_module.py
│ │ │ ├── evaluation_module.py
│ │ │ └── carbon_tracker.py
│ │ └── utils.py
│ └── scripts/
│ ├── run_experiment.py
│ ├── run_qa_experiment.py
│ ├── analyze_results.py
│ └── launch_dashboard.py
├── experiments/
│ ├── mlflow/
│ └── logs/
├── dashboards/
│ ├── dashboard_app.py
│ └── public/
├── notebooks/
│ └── tutorial.ipynb
├── tests/
│ ├── test_data_module.py
│ ├── test_model_adapter.py
│ └── test_carbon_tracker.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml / setup.py
└── README.md


---

## Getting Started

### Prerequisites
- Python **3.10+**
- Optional: CUDA + compatible PyTorch for GPU training

### Install
```bash
# clone
git clone <your-repo-url> your-toolkit && cd your-toolkit

# env + deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# (optional) editable install to enable `python -m toolkit.scripts.*`
pip install -e .
