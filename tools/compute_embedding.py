"""Compute embedding for a single sentence and optionally compare to an intent centroid."""

import argparse
import sys
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from tabulate import tabulate

from utils import load_config, load_training_data, compute_embeddings, cosine_similarity, compute_centroid


def compute_single_embedding(text: str, compare_intent: str = None):
    config = load_config()

    # Embed the input text
    embedding = compute_embeddings([text], config)[0]

    print(f"Embedded: \"{text}\"")
    print(f"  Dimensions: {len(embedding)}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")

    if compare_intent:
        # Load training data for the target intent
        train_df = load_training_data(config)
        intent_texts = train_df[train_df["intent"] == compare_intent]["text"].tolist()

        if not intent_texts:
            print(f"\n  Intent '{compare_intent}' not found or has no examples.")
            return

        intent_embeddings = compute_embeddings(intent_texts, config)
        centroid = compute_centroid(intent_embeddings)

        sim_to_centroid = cosine_similarity(embedding, centroid)
        print(f"\n  Similarity to '{compare_intent}' centroid: {sim_to_centroid:.4f}")

        # Also show similarity to individual examples (top 5 closest, bottom 5)
        sims = [cosine_similarity(embedding, e) for e in intent_embeddings]
        paired = list(zip(intent_texts, sims))
        paired.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Top 5 most similar examples in '{compare_intent}':")
        print(tabulate(
            [(t[:80], f"{s:.4f}") for t, s in paired[:5]],
            headers=["Text", "Similarity"],
            tablefmt="simple",
        ))

        print(f"\n  Top 5 least similar examples in '{compare_intent}':")
        print(tabulate(
            [(t[:80], f"{s:.4f}") for t, s in paired[-5:]],
            headers=["Text", "Similarity"],
            tablefmt="simple",
        ))

    # Compare to all intents (quick overview)
    if not compare_intent:
        print("\nTip: Use --compare-intent X to see similarity to a specific intent's centroid.")
        print("     Use predict.py for full model prediction with probabilities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embedding for a sentence")
    parser.add_argument("--text", required=True, help="Sentence to embed")
    parser.add_argument("--compare-intent", type=str, default=None,
                        help="Compare to this intent's centroid and examples")
    args = parser.parse_args()
    compute_single_embedding(args.text, args.compare_intent)
