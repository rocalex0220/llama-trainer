#!/usr/bin/env python
"""Authenticate with Hugging Face API to access gated models."""

import sys
from pathlib import Path


def authenticate_huggingface():
    """Authenticate with Hugging Face using CLI or token."""
    try:
        from huggingface_hub import login, get_token
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface_hub")
        return False
    
    # Check if already authenticated
    try:
        token = get_token()
        if token:
            print("✓ Already authenticated with Hugging Face")
            return True
    except Exception:
        pass
    
    print("\n" + "=" * 60)
    print("Hugging Face Authentication")
    print("=" * 60)
    print("\n1. Go to: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'repo' read access")
    print("3. Accept the license for Llama models at:")
    print("   https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E")
    print("\nEnter your token below (or press Ctrl+C to skip):\n")
    
    try:
        login()
        print("\n✓ Successfully authenticated!")
        return True
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return False


def main():
    if authenticate_huggingface():
        print("\nYou can now train with Llama models.")
        print("Run: python -m llama_trainer.cli --config configs/train.yaml")
        return 0
    else:
        print("\n⚠ Training may fail without authentication.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
