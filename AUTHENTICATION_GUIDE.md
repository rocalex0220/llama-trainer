# Authentication & Gated Models Guide

## Problem

Llama models from Meta are gated on Hugging Face, meaning you need:
1. To accept the license terms
2. To authenticate with your Hugging Face account

## Solution

### Step 1: Accept License

1. Go to the model page: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E
2. Click "Access repository"
3. Check the checkbox to accept terms
4. Click "Access repository" button

### Step 2: Get Your Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it (e.g., "llama-training")
4. Select **Read** access
5. Click "Create token"
6. Copy the token (you'll need it in the next step)

### Step 3: Authenticate

**Option A: Interactive (Recommended)**
```bash
python scripts/authenticate_huggingface.py
```
When prompted, paste your token and press Enter.

**Option B: Environment Variable**
```bash
# Windows (cmd):
set HF_TOKEN=your_token_here

# Windows (PowerShell):
$env:HF_TOKEN="your_token_here"

# Linux/Mac:
export HF_TOKEN="your_token_here"
```

**Option C: Manual Configuration**
```bash
# Create ~/.huggingface/token file with your token
# Linux/Mac:
mkdir -p ~/.huggingface
echo "your_token_here" > ~/.huggingface/token

# Windows:
mkdir %USERPROFILE%\.huggingface
echo your_token_here > %USERPROFILE%\.huggingface\token
```

### Step 4: Verify Authentication

```bash
python scripts/validate_training_env.py
```

## Verification

Test that authentication works:
```python
from huggingface_hub import model_info
info = model_info("meta-llama/Llama-4-Maverick-17B-128E")
print("✓ Successfully accessed model!")
```

## Troubleshooting

### "Access to this model is restricted"
- ✓ Check you accepted the license on the model page
- ✓ Verify your token is correct
- ✓ Ensure token has "Read" access

### "Invalid token"
- ✓ Generate a new token from https://huggingface.co/settings/tokens
- ✓ Double-check you copied it correctly (no spaces)

### "Token expired"
- ✓ Create a new token and re-authenticate

## Alternative: Use Public Model

If you cannot access the gated model, update `configs/train.yaml`:

```yaml
# Use Llama 2 (publicly available)
model_name_or_path: "meta-llama/Llama-2-70b"

# Or use a community fine-tuned version
model_name_or_path: "mistralai/Mistral-7B-v0.1"
```

Then proceed with training.
