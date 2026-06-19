import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class TrainingConfig:
    model_name_or_path: str
    dataset_name_or_path: str
    output_dir: Path
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 2048
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    wandb_project: Optional[str] = None
    push_to_hub: bool = False
    report_to: Optional[str] = None
    dataset_type: str = "huggingface"
    language: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def save(self, path: Path) -> None:
        data = dataclasses.asdict(self)
        if isinstance(data.get("output_dir"), Path):
            data["output_dir"] = str(data["output_dir"])

        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream)
