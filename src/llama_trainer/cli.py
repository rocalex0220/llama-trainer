import argparse
from pathlib import Path

from .config import TrainingConfig
from .serve import serve_model
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama Trainer entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")

    serve_parser = subparsers.add_parser("serve", help="Serve a trained model with vLLM")
    serve_parser.add_argument("--model-dir", type=Path, required=True, help="Path to a trained model directory")
    serve_parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    serve_parser.add_argument("--port", type=int, default=8000, help="HTTP server port")
    serve_parser.add_argument("--device", default="cuda", help="Device for vLLM engine")
    serve_parser.add_argument("--temperature", type=float, default=0.1, help="Default sampling temperature")
    serve_parser.add_argument("--top-p", type=float, default=0.95, help="Default sampling top-p")
    serve_parser.add_argument("--top-k", type=int, default=50, help="Default sampling top-k")
    serve_parser.add_argument("--max-new-tokens", type=int, default=256, help="Default max new tokens")
    serve_parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Default repetition penalty")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        config = TrainingConfig.load(args.config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(config)
        trainer.train()
    elif args.command == "serve":
        serve_model(
            model_dir=args.model_dir,
            host=args.host,
            port=args.port,
            device=args.device,
            default_temperature=args.temperature,
            default_top_p=args.top_p,
            default_top_k=args.top_k,
            default_max_new_tokens=args.max_new_tokens,
            default_repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
