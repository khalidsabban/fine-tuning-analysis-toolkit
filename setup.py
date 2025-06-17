from setuptools import setup, find_packages

setup(
    name="fine_tuning_analysis_toolkit",
    version="0.1.0",
    description="A reusable toolkit for fine-tuning LLMs (LoRA & QLoRA) with integrated energy/carbon tracking and experiment analysis.",
    author="Khalid Sabban",
    author_email="khalidsabban@outlook.com",
    url="https://github.com/khalidsbn/fine-tuning-analysis-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Configuration & CLI
        "hydra-core>=1.2",
        "omegaconf>=2.3",
        # Deep learning & adapters
        "torch>=2.0",
        "transformers>=4.30",
        "peft>=0.5",
        "accelerate>=0.20",
        "bitsandbytes>=0.39",
        # Training framework
        "pytorch-lightning>=1.9",
        # Experiment tracking & logging
        "mlflow>=2.3",
        "codecarbon>=2.0",
        # Data processing & metrics
        "datasets>=2.11",
        "scikit-learn>=1.2",
        "pandas>=2.0",
        # Dashboard & analysis (optional)
        "streamlit>=1.20",
        "plotly>=5.13",
        # Utilities
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "loguru>=0.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3",
            "pytest-cov>=4.0",
            "black>=23.3",
            "isort>=5.12",
            "flake8>=6.0",
            "mypy>=0.961",
            "pre-commit>=2.22",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-experiment=toolkit.scripts.run_experiment:main",
            "analyze-results=toolkit.scripts.analyze_results:main",
            "launch-dashboard=toolkit.scripts.launch_dashboard:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
