from pathlib import Path

from llama_trainer.config import TrainingConfig


def test_training_config_load_save(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config = TrainingConfig(
        model_name_or_path="meta-llama/Llama-2-7b",
        dataset_name_or_path="wikitext",
        output_dir=tmp_path / "outputs",
    )
    config.save(config_file)

    loaded = TrainingConfig.load(config_file)
    assert loaded.model_name_or_path == config.model_name_or_path
    assert loaded.dataset_name_or_path == config.dataset_name_or_path
    assert loaded.output_dir == config.output_dir
