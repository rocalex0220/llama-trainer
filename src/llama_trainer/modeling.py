from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def build_model(model_name_or_path: str, device: Optional[str] = None, use_auth_token: bool = True):
    """Build model and tokenizer. Requires HF authentication for gated models."""
    logger.info("Building model from: %s", model_name_or_path)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_auth_token=use_auth_token if use_auth_token is not True else None,
            device_map="auto" if device is None else device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            use_auth_token=use_auth_token if use_auth_token is not True else None,
        )
    except Exception as e:
        if "403" in str(e) or "Access to this model" in str(e):
            raise PermissionError(
                f"Cannot access {model_name_or_path}. This is a gated model.\n"
                f"1. Accept the license: https://huggingface.co/{model_name_or_path}\n"
                f"2. Authenticate: python scripts/authenticate_huggingface.py\n"
                f"Error: {e}"
            ) from e
        raise
    
    if device:
        model.to(device)
    return model, tokenizer


def save_checkpoint(model: torch.nn.Module, output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint-{step}"
    logger.info("Saving checkpoint to %s", path)
    model.save_pretrained(path)


def load_checkpoint(path: Path, device: Optional[str] = None):
    logger.info("Loading checkpoint from %s", path)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if device:
        model.to(device)
    return model, tokenizer
