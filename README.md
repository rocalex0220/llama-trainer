# Llama Trainer

Production-ready repository for training Llama models (including Llama 4 Maverick) on custom language corpora using PyTorch, Transformers, Accelerate, and PEFT.

## What is included

- Training orchestration and configuration for multiple dataset types
- Dataset preparation with support for HuggingFace datasets and local text files
- Model loading, fine-tuning, and checkpointing
- Experiment tracking with Weights & Biases
- Language-specific preprocessing (Chinese corpus cleaning included)
- GitHub Actions CI for linting and tests
- Comprehensive validation and monitoring scripts
- Package metadata and Python module structure

## Quick start

1. Create a Python environment with Python 3.11+
2. Install runtime dependencies: `pip install -e .[train]`
3. Validate your setup: `python scripts/validate_training_env.py`
4. Configure `configs/train.yaml`
5. Run training: `python -m llama_trainer.cli --config configs/train.yaml`

## Usage Examples

### Chinese Training on Llama 4 Maverick

See [TRAINING_GUIDE_CHINESE.md](TRAINING_GUIDE_CHINESE.md) for complete setup instructions.

```bash
# 1. Prepare your corpus
python scripts/prepare_chinese_corpus.py your_corpus.txt -o texts.txt

# 2. Validate environment
python scripts/validate_training_env.py

# 3. Configure training
# Edit configs/train.yaml with your settings

# 4. Start training
python -m llama_trainer.cli --config configs/train.yaml
```

## Repository structure

- `src/llama_trainer`: core training package
  - `config.py`: configuration management
  - `dataset.py`: dataset loading (HuggingFace and local text files)
  - `modeling.py`: model utilities
  - `trainer.py`: training loop with Accelerate and mixed precision
  - `cli.py`: command-line entry point
- `configs/`: example YAML configuration files
- `tests/`: unit and integration tests
- `.github/workflows/`: CI definitions
- `scripts/`: helper scripts for corpus preparation and validation
