from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset as hf_load_dataset, Dataset
from loguru import logger


def load_local_text_dataset(file_path: str, split_ratio: float = 0.9) -> tuple[Dataset, Dataset]:
    """Load a text file and split into train/eval datasets."""
    logger.info("Loading local text file: %s", file_path)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Split into chunks (e.g., paragraphs separated by blank lines or fixed size)
    texts = []
    current_text = []
    
    for line in lines:
        line = line.strip()
        if line:
            current_text.append(line)
        elif current_text:
            texts.append(" ".join(current_text))
            current_text = []
    
    if current_text:
        texts.append(" ".join(current_text))
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Split
    split_dataset = dataset.train_test_split(train_size=split_ratio, seed=42)
    return split_dataset["train"], split_dataset["test"]


def load_dataset(dataset_name_or_path: str, dataset_type: str = "huggingface", split: str = "train") -> Any:
    """Load a dataset from HuggingFace or local path."""
    if dataset_type == "local_text":
        train_ds, eval_ds = load_local_text_dataset(dataset_name_or_path)
        return train_ds if split == "train" else eval_ds
    else:
        logger.info("Loading dataset: %s", dataset_name_or_path)
        return hf_load_dataset(dataset_name_or_path, split=split)


def validate_dataset(dataset: Any, required_columns: Optional[list[str]] = None) -> None:
    """Validate that the dataset contains the expected fields."""
    if required_columns is None:
        required_columns = ["text"]

    first_item = dataset[0]
    missing = [col for col in required_columns if col not in first_item]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
