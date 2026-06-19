# Chinese Llama 4 Maverick Training Guide

Complete setup for training Llama 4 Maverick on a Chinese language corpus from scratch.

## Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU training)
- ~80GB VRAM for 70B model with gradient accumulation
- Your Chinese text corpus in `texts.txt`
- Hugging Face account with Llama model access

## Initial Setup (One-Time)

### 1. Install Dependencies

```bash
pip install -e .[train]
```

### 2. Authenticate with Hugging Face

Llama models are gated and require authentication:

```bash
python scripts/authenticate_huggingface.py
```

**Need help?** See [AUTHENTICATION_GUIDE.md](AUTHENTICATION_GUIDE.md)

### 3. Verify Environment

```bash
python scripts/validate_training_env.py
```

## Training Setup

### 1. Prepare Corpus

Clean and deduplicate your Chinese text:

```bash
python scripts/prepare_chinese_corpus.py your_corpus.txt -o texts.txt
```

Or if your corpus is already clean:
```bash
cp your_corpus.txt texts.txt
```

**Corpus format:** One text per line, UTF-8 encoded

### 2. Configure Training

Edit `configs/train.yaml`:

```yaml
model_name_or_path: "meta-llama/Llama-4-Maverick-17B-128E"
dataset_name_or_path: "texts.txt"
dataset_type: "local_text"
output_dir: "outputs/llama4-maverick-chinese"
train_batch_size: 2
eval_batch_size: 2
learning_rate: 1e-5
num_train_epochs: 1
gradient_accumulation_steps: 4
max_seq_length: 4096
bf16: true
language: "chinese"
```

**Key parameters:**
- `train_batch_size`: Per-device batch size (reduce if OOM)
- `gradient_accumulation_steps`: Effective batch = batch_size × accumulation_steps
- `max_seq_length`: Context length (4096 recommended)
- `bf16`: Use bfloat16 for better stability on newer GPUs

## Training

### Single GPU

```bash
python -m llama_trainer.cli --config configs/train.yaml
```

### Multi-GPU (Distributed)

```bash
accelerate launch -m llama_trainer.cli --config configs/train.yaml
```

### Monitoring

- **Terminal logs:** Watch real-time loss and metrics
- **Weights & Biases:** Configure `wandb_project` in config
- **Checkpoints:** Saved to `output_dir/checkpoints/checkpoint-{step}`

## Performance Optimization

### Memory Issues

If you encounter OOM errors:

1. Reduce batch size:
   ```yaml
   train_batch_size: 1
   gradient_accumulation_steps: 8
   ```

2. Enable gradient checkpointing (add to trainer.py):
   ```python
   self.model.gradient_checkpointing_enable()
   ```

3. Use lower precision:
   ```yaml
   bf16: false
   fp16: true
   ```

### Speed Improvements

- Use multi-GPU training with `accelerate`
- Increase `max_seq_length` carefully (memory trade-off)
- Pre-tokenize dataset beforehand
- Monitor GPU utilization with `nvidia-smi`

## Resuming Training

Checkpoints include optimizer state. To resume:

1. Edit `configs/train.yaml`:
   ```yaml
   resume_from_checkpoint: "outputs/llama4-maverick-chinese/checkpoints/checkpoint-5000"
   ```

2. Or start training (will auto-detect latest checkpoint):
   ```bash
   python -m llama_trainer.cli --config configs/train.yaml
   ```

## Evaluation

Validation runs every `eval_steps`. Metrics logged to:
- Console logs
- Weights & Biases (if configured)

## Output

After training completes:
- **Model checkpoints:** `output_dir/checkpoints/checkpoint-{step}`
- **Final model:** Use latest checkpoint or merge with base model
- **Logs:** `output_dir/logs/` (if enabled)

## Language-Specific Tips

### Chinese Training

- No special tokenizer needed (Llama 4 supports ~150K Chinese characters)
- Text normalization is important (remove formatting, deduplicate)
- Longer context (4096 tokens) helps capture complex grammar
- Monitor character-level accuracy during evaluation

### Corpus Preparation

Recommended corpus format:
```
中文文本行1，完整的句子或段落。
中文文本行2，另一个完整的文本。
中文文本行3，继续...
```

## Common Issues

### Authentication Error
```
Access to this model is restricted. You must have access to it...
```
**Solution:** Run `python scripts/authenticate_huggingface.py`

### CUDA Out of Memory (OOM)
```
CUDA out of memory
```
**Solution:** Reduce `train_batch_size` or increase `gradient_accumulation_steps`

### Slow Training
```
Training is much slower than expected
```
**Solution:** Check `nvidia-smi` for GPU utilization. If low, check CPU bottleneck.

### Dataset Errors
```
Dataset is missing required columns
```
**Solution:** Ensure `texts.txt` has one complete text per line

## References

- [Llama 4 Maverick Model Card](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/training)
- [Accelerate Multi-GPU Training](https://huggingface.co/docs/accelerate/usage_guides/multi_gpu)
- [HF Authentication](AUTHENTICATION_GUIDE.md)
