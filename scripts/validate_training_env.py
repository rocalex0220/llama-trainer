#!/usr/bin/env python
"""Validate training environment and dataset before starting."""

import sys
from pathlib import Path

def check_dependencies():
    """Verify required packages are installed."""
    required = [
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "peft",
        "loguru"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -e .[train]")
        return False
    
    print("✓ All required packages installed")
    return True


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("⚠ No GPU detected (CPU training will be very slow)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def check_corpus(corpus_path: str = "texts.txt"):
    """Validate corpus file."""
    path = Path(corpus_path)
    if not path.exists():
        print(f"❌ Corpus file not found: {corpus_path}")
        return False
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = len(f.readlines())
        
        size_mb = path.stat().st_size / 1e6
        print(f"✓ Corpus found: {lines:,} lines, {size_mb:.1f}MB")
        
        if lines < 100:
            print("⚠ Corpus is very small (< 100 lines)")
        
        return True
    except Exception as e:
        print(f"❌ Error reading corpus: {e}")
        return False


def check_config(config_path: str = "configs/train.yaml"):
    """Validate config file."""
    path = Path(config_path)
    if not path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    print(f"✓ Config file found: {config_path}")
    return True


def main():
    print("=" * 50)
    print("Training Environment Validation")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("GPU Support", check_gpu),
        ("Corpus File", lambda: check_corpus()),
        ("Config File", lambda: check_config()),
    ]
    
    results = []
    for name, check in checks:
        print(f"\n{name}:")
        results.append(check())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ Environment is ready for training!")
        return 0
    else:
        print("❌ Please fix the issues above before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
