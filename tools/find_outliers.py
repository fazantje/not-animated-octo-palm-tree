"""Find training examples furthest from their own intent's centroid.

Low similarity to centroid often indicates a mislabeled example. Diagnostic
only — no side effects.
"""

import argparse
import sys
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import (
    load_config, get_path, load_training_data, compute_embeddings,
    compute_centroid, cosine_similarity, get_intent_names,
)


def find_outliers_for_intent(intent: str, top_n: int, train_df: pd.DataFrame,
                             all_embeddings: np.ndarray, all_texts: list[str]):
    mask = train_df["intent"].values == intent
    n = int(mask.sum())
    if n < 2:
        print(f"  Intent '{intent}' has {n} example(s) — need at least 2 to rank outliers.")
        return

    intent_texts = [t for t, m in zip(all_texts, mask) if m]
    intent_embs = all_embeddings[mask]
    centroid = compute_centroid(intent_embs)
    sims = [cosine_similarity(e, centroid) for e in intent_embs]

    ranked = sorted(zip(intent_texts, sims), key=lambda x: x[1])
    worst = ranked[:top_n]

    print(f"\n--- {intent} (n={n}) — {len(worst)} lowest-similarity examples ---")
    rows = [[f"{sim:.4f}", text] for text, sim in worst]
    print(tabulate(rows, headers=["Sim to centroid", "Text"], tablefmt="simple"))


def find_outliers(intent: str | None, all_intents: bool, top_n: int):
    config = load_config()

    if not intent and not all_intents:
        print("Error: specify either --intent X or --all.")
        return

    print("Loading training data and embeddings...")
    train_df = load_training_data(config)
    all_texts = list(train_df["text"].values)
    all_embeddings = compute_embeddings(all_texts, config)

    if all_intents:
        for name in get_intent_names(config):
            find_outliers_for_intent(name, top_n, train_df, all_embeddings, all_texts)
    else:
        if intent not in set(train_df["intent"].values):
            print(f"Error: intent '{intent}' not found in training data.")
            return
        find_outliers_for_intent(intent, top_n, train_df, all_embeddings, all_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find examples far from their own intent's centroid")
    parser.add_argument("--intent", type=str, default=None,
                        help="Intent name to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all intents (prints top-N per intent)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="How many outliers to show (default: 10)")
    args = parser.parse_args()
    find_outliers(args.intent, args.all, args.top_n)
