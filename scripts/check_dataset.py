from pathlib import Path

from llama_trainer.dataset import load_dataset, validate_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check a Hugging Face dataset for training readiness")
    parser.add_argument("dataset", type=str, help="Dataset name or path")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    validate_dataset(dataset)
    print("Dataset is valid for training.")


if __name__ == "__main__":
    main()
