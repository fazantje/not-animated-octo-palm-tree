"""Find noisy training examples: long/verbose rows and multi-intent suspects.

Diagnostic only, no side effects. Writes results/current/noisy_candidates.csv
which the restructure agent reads during Phase 0.5 to decide trim vs drop per
row.

Flag semantics:
- 'long': word_count > max(intent_median + 2*intent_stddev, --global-length-floor)
- 'multi_intent_suspect': own-centroid sim minus nearest-other-centroid sim
  is below --multi-intent-margin (default 0.05). These rows likely carry two
  intents in one utterance and — given our 2-layer architecture — should be
  dropped from layer-1 training; the disambiguation layer handles multi-intent
  at inference.

A row can carry both flags. All flagged rows go to the CSV; unflagged rows do
not.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import (
    load_config, get_path, load_training_data, compute_embeddings,
    get_intent_names, get_skill, compute_intent_centroids,
    cosine_similarity_matrix,
)


def find_noisy(intent_filter: str | None, global_length_floor: int,
               multi_intent_margin: float):
    config = load_config()

    print("Loading training data and embeddings...")
    full_df = load_training_data(config).reset_index(drop=True)
    full_embeddings = compute_embeddings(list(full_df["text"].values), config)

    intent_names = get_intent_names(config)
    centroids = compute_intent_centroids(
        full_embeddings, full_df["intent"].values, intent_names,
    )

    full_df["word_count"] = full_df["text"].str.split().str.len()
    intent_stats = (
        full_df.groupby("intent")["word_count"].agg(["median", "std"]).fillna(0.0)
    )

    # Optionally restrict the scan to a single intent. Centroids/stats still
    # come from the full corpus so thresholds are stable.
    if intent_filter:
        if intent_filter not in set(full_df["intent"].values):
            print(f"Error: intent '{intent_filter}' not found.")
            return
        mask = (full_df["intent"].values == intent_filter)
        scan_df = full_df[mask].reset_index(drop=True)
        scan_embeddings = full_embeddings[mask]
    else:
        scan_df = full_df
        scan_embeddings = full_embeddings

    sim_matrix = cosine_similarity_matrix(scan_embeddings, centroids)
    intent_to_idx = {n: i for i, n in enumerate(intent_names)}

    rows = []
    for i in range(len(scan_df)):
        text = scan_df["text"].iloc[i]
        intent = scan_df["intent"].iloc[i]
        wc = int(scan_df["word_count"].iloc[i])

        own_idx = intent_to_idx[intent]
        own_sim = float(sim_matrix[i, own_idx])
        others = sim_matrix[i].copy()
        others[own_idx] = -np.inf
        top2_idx = int(others.argmax())
        top2_sim = float(others[top2_idx])
        top2_intent = intent_names[top2_idx]
        top2_margin = own_sim - top2_sim

        med = float(intent_stats.loc[intent, "median"])
        std = float(intent_stats.loc[intent, "std"])
        zscore = (wc - med) / (std + 1e-10)

        flags = []
        length_threshold = max(med + 2.0 * std, float(global_length_floor))
        if wc > length_threshold:
            flags.append("long")
        if top2_margin < multi_intent_margin:
            flags.append("multi_intent_suspect")

        if flags:
            rows.append({
                "text": text,
                "intent": intent,
                "skill": get_skill(intent) or "(unknown)",
                "word_count": wc,
                "length_zscore": round(zscore, 3),
                "top2_margin": round(top2_margin, 4),
                "top2_intent": top2_intent,
                "flag_reason": ",".join(flags),
            })

    noisy_df = pd.DataFrame(rows, columns=[
        "text", "intent", "skill", "word_count", "length_zscore",
        "top2_margin", "top2_intent", "flag_reason",
    ])

    results_dir = get_path(config, "results_dir") / "current"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "noisy_candidates.csv"
    noisy_df.to_csv(out_path, index=False)

    total_examples = len(scan_df)
    total_flagged = len(noisy_df)
    print(f"\n{'=' * 60}")
    print(f"NOISY CANDIDATES ({total_flagged} of {total_examples} examples flagged)")
    print(f"{'=' * 60}")

    if total_flagged == 0:
        print("No flagged rows. Corpus looks clean on length + multi-intent axes.")
        print(f"\nEmpty CSV written to: {out_path}")
        return

    by_reason = noisy_df["flag_reason"].value_counts()
    print("\nBy flag combination:")
    for reason, count in by_reason.items():
        print(f"  {reason}: {count}")

    by_intent = noisy_df["intent"].value_counts().head(10)
    print("\nTop intents by flagged count:")
    for intent, count in by_intent.items():
        total = int((scan_df["intent"].values == intent).sum())
        pct = 100 * count / total if total else 0
        print(f"  {intent}: {count}/{total} ({pct:.0f}%)")

    print(f"\nFull CSV: {out_path}")
    print("\nRemediation (per Phase 0.5 in the skill prompt):")
    print("  - multi_intent_suspect → log to notes/noisy_review.md, then drop")
    print("  - long only           → read text; trim (remove+add) OR drop. Err toward drop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find noisy/long training examples and multi-intent suspects",
    )
    parser.add_argument(
        "--intent", type=str, default=None,
        help="Restrict scan to a single intent (default: all intents). "
             "Thresholds still use full-corpus statistics.",
    )
    parser.add_argument(
        "--global-length-floor", type=int, default=25,
        help="Absolute word-count floor for the 'long' flag (default: 25). "
             "A row is flagged 'long' if it exceeds max(intent_median+2σ, this floor).",
    )
    parser.add_argument(
        "--multi-intent-margin", type=float, default=0.05,
        help="Threshold for 'multi_intent_suspect' flag (default: 0.05). "
             "Margin = own-intent centroid sim minus nearest-other-intent centroid sim.",
    )
    args = parser.parse_args()
    find_noisy(args.intent, args.global_length_floor, args.multi_intent_margin)
