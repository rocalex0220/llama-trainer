"""
Prepare and clean Chinese text corpus for training.

Handles:
- Removing duplicates
- Removing empty lines
- Normalizing whitespace
- Removing non-Chinese characters (optional)
"""

import argparse
from pathlib import Path
from collections import Counter
import re


def normalize_text(text: str, remove_non_chinese: bool = False) -> str:
    """Normalize and clean Chinese text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common noise patterns
    text = re.sub(r'【.*?】', '', text)  # Remove brackets
    text = re.sub(r'《.*?》', '', text)  # Remove angle brackets
    
    if remove_non_chinese:
        # Keep only Chinese characters, spaces, and punctuation
        text = re.sub(r'[^\u4e00-\u9fff\s，。！？；：""''（）\w]', '', text)
    
    return text.strip()


def deduplicate_corpus(input_file: str, output_file: str, remove_non_chinese: bool = False) -> None:
    """Remove duplicate lines and normalize corpus."""
    seen = set()
    total = 0
    unique = 0
    
    with open(input_file, "r", encoding="utf-8") as inf, \
         open(output_file, "w", encoding="utf-8") as outf:
        for line in inf:
            total += 1
            line = line.strip()
            
            if not line:
                continue
            
            normalized = normalize_text(line, remove_non_chinese)
            
            if normalized and normalized not in seen:
                seen.add(normalized)
                outf.write(normalized + "\n")
                unique += 1
    
    print(f"Total lines: {total}")
    print(f"Unique lines: {unique}")
    print(f"Duplicates removed: {total - unique}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Chinese corpus for training")
    parser.add_argument("input", help="Input text file")
    parser.add_argument("-o", "--output", default="corpus_cleaned.txt", help="Output file")
    parser.add_argument("--remove-non-chinese", action="store_true", help="Remove non-Chinese characters")
    
    args = parser.parse_args()
    
    deduplicate_corpus(args.input, args.output, args.remove_non_chinese)
    print(f"Cleaned corpus saved to: {args.output}")


if __name__ == "__main__":
    main()
