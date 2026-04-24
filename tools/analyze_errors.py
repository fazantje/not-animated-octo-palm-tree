"""Analyze misclassifications between two confused intents.

Reads k-fold out-of-fold predictions (produced by kfold_confusion.py). The
restructure workflow never touches val/test, so k-fold OOF is the source of
truth for per-example errors.
"""

import argparse
import sys
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tabulate import tabulate

from utils import load_config, get_path, load_intent_description


def analyze_errors(intent_a: str, intent_b: str):
    config = load_config()

    # Load k-fold OOF misclassifications
    misc_path = get_path(config, "results_dir") / "current" / "kfold_misclassifications.csv"
    if not misc_path.exists():
        print("No k-fold misclassifications file found. Run kfold_confusion.py first.")
        return

    misc_df = pd.read_csv(misc_path)

    # Filter for A→B and B→A confusion
    ab_mask = (misc_df["true_intent"] == intent_a) & (misc_df["predicted_intent"] == intent_b)
    ba_mask = (misc_df["true_intent"] == intent_b) & (misc_df["predicted_intent"] == intent_a)
    pair_df = misc_df[ab_mask | ba_mask].copy()

    if len(pair_df) == 0:
        print(f"No misclassifications found between '{intent_a}' and '{intent_b}'.")
        print("This pair may not be confused in the current model's evaluation.")
        return

    # Print descriptions
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {intent_a} ↔ {intent_b}")
    print(f"{'='*60}")

    print(f"\n--- Description: {intent_a} ---")
    print(load_intent_description(config, intent_a))

    print(f"\n--- Description: {intent_b} ---")
    print(load_intent_description(config, intent_b))

    # Show A→B errors
    a_to_b = pair_df[ab_mask]
    if len(a_to_b) > 0:
        print(f"\n--- True: {intent_a} → Predicted: {intent_b} ({len(a_to_b)} errors) ---")
        rows = []
        for _, row in a_to_b.iterrows():
            rows.append([row["text"], f"{row['confidence']:.4f}"])
        print(tabulate(rows, headers=["Text", "Confidence"], tablefmt="simple"))

    # Show B→A errors
    b_to_a = pair_df[ba_mask]
    if len(b_to_a) > 0:
        print(f"\n--- True: {intent_b} → Predicted: {intent_a} ({len(b_to_a)} errors) ---")
        rows = []
        for _, row in b_to_a.iterrows():
            rows.append([row["text"], f"{row['confidence']:.4f}"])
        print(tabulate(rows, headers=["Text", "Confidence"], tablefmt="simple"))

    # Summary
    total = len(pair_df)
    a_to_b_count = len(a_to_b)
    b_to_a_count = len(b_to_a)
    print(f"\n--- Summary ---")
    print(f"Total confusions: {total}")
    print(f"  {intent_a} → {intent_b}: {a_to_b_count}")
    print(f"  {intent_b} → {intent_a}: {b_to_a_count}")

    if a_to_b_count > b_to_a_count * 2:
        print(f"\nNote: Confusion is mostly one-directional ({intent_a} → {intent_b}).")
        print(f"  This suggests {intent_a}'s training data may contain examples that belong to {intent_b},")
        print(f"  or {intent_a} lacks clear distinguishing examples.")
    elif b_to_a_count > a_to_b_count * 2:
        print(f"\nNote: Confusion is mostly one-directional ({intent_b} → {intent_a}).")
        print(f"  This suggests {intent_b}'s training data may contain examples that belong to {intent_a},")
        print(f"  or {intent_b} lacks clear distinguishing examples.")
    else:
        print(f"\nNote: Confusion is bidirectional — these intents may have genuine semantic overlap.")
        print(f"  Consider whether the LLM disambiguation stage (stage 2) should handle this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze errors between two intents")
    parser.add_argument("--intent-a", required=True, help="First intent name")
    parser.add_argument("--intent-b", required=True, help="Second intent name")
    args = parser.parse_args()
    analyze_errors(args.intent_a, args.intent_b)
