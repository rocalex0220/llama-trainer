from pathlib import Path
from typing import Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def _import_vllm_dependencies() -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        msg = (
            "vLLM serve support requires a working vLLM installation with native extensions. "
            "On Windows, vLLM is often not available or may not provide the compiled `_C` backend. "
            "Use a Linux/CUDA environment or WSL if possible, and install the optional dependencies with: pip install -e .[serve]"
        )
        raise ImportError(msg) from exc

    return FastAPI, BaseModel, LLM, SamplingParams


def build_vllm_app(
    model_dir: Path,
    device: Optional[str] = None,
    default_temperature: float = 0.1,
    default_top_p: float = 0.95,
    default_top_k: int = 50,
    default_max_new_tokens: int = 256,
    default_repetition_penalty: float = 1.0,
) -> Any:
    FastAPI, BaseModel, LLM, SamplingParams = _import_vllm_dependencies()

    model_path = str(model_dir)
    device = device or "cuda"
    logger.info("Loading vLLM model from %s on device %s", model_path, device)

    llm = LLM(model=model_path, tokenizer=model_path, engine=device)
    app = FastAPI(title="llama-trainer vLLM server")

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = default_max_new_tokens
        temperature: float = default_temperature
        top_p: float = default_top_p
        top_k: int = default_top_k
        repetition_penalty: float = default_repetition_penalty
        stop: Optional[List[str]] = None

    class GenerateResponse(BaseModel):
        text: str

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop,
        )

        outputs = llm.generate([request.prompt], sampling_params=sampling_params)
        output = next(outputs)
        return GenerateResponse(text=output.text)

    return app


def serve_model(
    model_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: Optional[str] = None,
    default_temperature: float = 0.1,
    default_top_p: float = 0.95,
    default_top_k: int = 50,
    default_max_new_tokens: int = 256,
    default_repetition_penalty: float = 1.0,
) -> None:
    import uvicorn

    app = build_vllm_app(
        model_dir,
        device=device,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_top_k=default_top_k,
        default_max_new_tokens=default_max_new_tokens,
        default_repetition_penalty=default_repetition_penalty,
    )
    uvicorn.run(app, host=host, port=port)
