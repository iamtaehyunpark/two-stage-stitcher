"""
Download and mix long-form documents from two open sources:
  - wikimedia/wikipedia  (encyclopedic, topic diversity)
  - pg19                 (books, long narrative prose, style diversity)

Each document is saved as a plain .txt file named by source and index.
Documents shorter than --min-words are skipped.

Usage:
    python download_data.py --out-dir /data/tpark45/docs --num-docs 500
    # 250 from Wikipedia + 250 from PG-19 by default (50/50 split)
"""

import os
import argparse
from datasets import load_dataset
from tqdm import tqdm


SOURCES = {
    "wiki": {
        "dataset": "wikimedia/wikipedia",
        "config":  "20231101.en",
        "split":   "train",
        "field":   "text",
    },
    "gutenberg": {
        "dataset": "sedthh/gutenberg_english",
        "config":  None,
        "split":   "train",
        "field":   "TEXT",
    },
}


def fetch_docs(source_key: str, n: int, min_words: int, out_dir: str, offset: int = 0):
    src = SOURCES[source_key]
    print(f"  Loading {src['dataset']} …")
    kwargs = dict(split=src["split"], streaming=True)
    if src["config"]:
        kwargs["name"] = src["config"]
    ds = load_dataset(src["dataset"], **kwargs)

    saved = 0
    for example in tqdm(ds, desc=f"  {source_key}", total=n):
        if saved >= n:
            break
        text = example[src["field"]].strip()
        if len(text.split()) < min_words:
            continue
        fname = f"{source_key}_{offset + saved:05d}.txt"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(text)
        saved += 1

    print(f"  Saved {saved} {source_key} docs")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",   required=True)
    parser.add_argument("--num-docs",  type=int, default=500,
                        help="Total documents (split evenly between sources)")
    parser.add_argument("--min-words", type=int, default=1000,
                        help="Skip documents shorter than this many words")
    parser.add_argument("--wiki-frac", type=float, default=0.5,
                        help="Fraction from Wikipedia (rest from PG-19)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    n_wiki = int(args.num_docs * args.wiki_frac)
    n_pg19 = args.num_docs - n_wiki
    print(f"Target: {n_wiki} Wikipedia + {n_pg19} PG-19  →  {args.out_dir}")

    saved_wiki = fetch_docs("wiki", n_wiki, args.min_words, args.out_dir, offset=0)
    fetch_docs("gutenberg", n_pg19, args.min_words, args.out_dir, offset=saved_wiki)

    total = len([f for f in os.listdir(args.out_dir) if f.endswith(".txt")])
    print(f"\nTotal docs in {args.out_dir}: {total}")


if __name__ == "__main__":
    main()
