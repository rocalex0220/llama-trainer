import argparse
from pathlib import Path

from .config import TrainingConfig
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama Trainer entry point")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.load(args.config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
