"""
Download long-form documents and save each as a plain .txt file.

Default source: ccdv/arxiv-summarization (open, no auth, avg ~6k words/doc).
Each saved file = one article body, named by its index.

Usage:
    python download_data.py --out-dir /data/tpark45/docs --num-docs 500
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-docs", type=int, default=500,
                        help="Number of documents to save")
    parser.add_argument("--dataset", default="ccdv/arxiv-summarization",
                        help="HuggingFace dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="article",
                        help="Dataset column containing the document text")
    parser.add_argument("--min-words", type=int, default=1000,
                        help="Skip documents shorter than this many words")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.dataset} ({args.split}) …")
    ds = load_dataset(args.dataset, split=args.split, streaming=True,
                      trust_remote_code=True)

    saved = 0
    for example in tqdm(ds, desc="Saving docs", total=args.num_docs):
        if saved >= args.num_docs:
            break
        text = example[args.text_field].strip()
        if len(text.split()) < args.min_words:
            continue
        out_path = os.path.join(args.out_dir, f"doc_{saved:05d}.txt")
        with open(out_path, "w") as f:
            f.write(text)
        saved += 1

    print(f"Saved {saved} documents → {args.out_dir}")


if __name__ == "__main__":
    main()
