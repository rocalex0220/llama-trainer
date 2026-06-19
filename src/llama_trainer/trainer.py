import math
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate import Accelerator
from datasets import Dataset
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .modeling import build_model
from .dataset import load_dataset, validate_dataset
from .config import TrainingConfig


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        precision = "bf16" if config.bf16 else ("fp16" if config.fp16 else "no")
        self.accelerator = Accelerator(
            mixed_precision=precision,
            log_with=config.report_to,
            project_dir=str(config.output_dir),
        )

        self.device = self.accelerator.device
        self.model, self.tokenizer = build_model(config.model_name_or_path, device=self.device)

    def prepare_dataset(self) -> Dataset:
        dataset = load_dataset(
            self.config.dataset_name_or_path,
            dataset_type=self.config.dataset_type
        )
        validate_dataset(dataset, required_columns=["text"])
        
        # Tokenize
        def tokenize_function(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset

    def get_dataloader(self, dataset: Dataset, train: bool = True) -> DataLoader:
        batch_size = self.config.train_batch_size if train else self.config.eval_batch_size
        return DataLoader(dataset, batch_size=batch_size, shuffle=train)

    def setup_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def train(self) -> None:
        dataset = self.prepare_dataset()
        dataloader = self.get_dataloader(dataset)
        optimizer = self.setup_optimizer()

        self.model, optimizer, dataloader = self.accelerator.prepare(self.model, optimizer, dataloader)
        num_update_steps_per_epoch = math.ceil(len(dataloader) / self.config.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_train_epochs * num_update_steps_per_epoch,
        )

        self.model.train()
        global_step = 0
        for epoch in range(self.config.num_train_epochs):
            for step, batch in enumerate(dataloader):
                outputs = self.model(**batch)
                loss = outputs.loss
                self.accelerator.backward(loss)

                if step % self.config.gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.logging_steps == 0:
                        self.accelerator.log({"train/loss": loss.item()}, step=global_step)

                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)

                    if global_step % self.config.eval_steps == 0:
                        self.evaluate()

        self.save_checkpoint(global_step)

    def evaluate(self) -> None:
        eval_dataset = load_dataset(
            self.config.dataset_name_or_path,
            dataset_type=self.config.dataset_type,
            split="eval"
        )
        
        # Tokenize
        def tokenize_function(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        dataloader = self.get_dataloader(eval_dataset, train=False)
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(**batch)
                losses.append(outputs.loss.item())

        self.model.train()
        average_loss = sum(losses) / len(losses) if losses else 0.0
        self.accelerator.log({"eval/loss": average_loss})
        logger.info("Evaluation loss: %s", average_loss)

    def save_checkpoint(self, step: int) -> None:
        output_dir = self.config.output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir / f"checkpoint-{step}")
