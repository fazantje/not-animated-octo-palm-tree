"""Query training, validation, and test data. Includes inventory mode."""

import argparse
import json
import sys
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import (
    load_config, get_path, load_training_data, load_split_data,
    load_intent_description, compute_embeddings, load_embeddings_cache,
    get_intent_names, cosine_similarity, compute_centroid,
)


def compute_cohesion(embeddings: np.ndarray) -> float:
    """Average cosine similarity of examples to their centroid."""
    centroid = compute_centroid(embeddings)
    sims = [cosine_similarity(e, centroid) for e in embeddings]
    return float(np.mean(sims))


def show_inventory():
    """Show overview of all intents with key metrics."""
    config = load_config()
    train_dir = get_path(config, "train_dir")
    intent_names = get_intent_names(config)

    # Load all training data and embeddings
    train_df = load_training_data(config)
    all_texts = list(train_df["text"].values)
    all_embeddings = compute_embeddings(all_texts, config)

    # Build per-intent embedding lookup
    intent_data = {}
    for intent_name in intent_names:
        mask = train_df["intent"].values == intent_name
        n = int(mask.sum())
        intent_embs = all_embeddings[mask]

        intent_data[intent_name] = {
            "count": n,
            "embeddings": intent_embs,
            "centroid": compute_centroid(intent_embs) if n > 0 else None,
            "cohesion": compute_cohesion(intent_embs) if n > 1 else 0.0,
        }

    # Compute nearest neighbor for each intent
    for name, data in intent_data.items():
        if data["centroid"] is None:
            data["nearest"] = ("N/A", 0.0)
            continue
        best_sim = -1
        best_neighbor = "N/A"
        for other_name, other_data in intent_data.items():
            if other_name == name or other_data["centroid"] is None:
                continue
            sim = cosine_similarity(data["centroid"], other_data["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_neighbor = other_name
        data["nearest"] = (best_neighbor, round(best_sim, 4))

    # Load F1 scores if available
    metrics_path = get_path(config, "results_dir") / "current" / "metrics.json"
    per_intent_f1 = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        per_intent_f1 = {
            name: info["f1"]
            for name, info in metrics.get("per_intent", {}).items()
        }

    # Build table
    rows = []
    for name in intent_names:
        d = intent_data[name]
        f1 = per_intent_f1.get(name, None)
        f1_str = f"{f1:.4f}" if f1 is not None else "—"
        nearest_name, nearest_sim = d["nearest"]
        # Truncate nearest name for display
        nearest_display = nearest_name[:30] + "..." if len(nearest_name) > 30 else nearest_name
        rows.append([
            name,
            d["count"],
            f"{d['cohesion']:.4f}",
            f1_str,
            nearest_display,
            f"{nearest_sim:.4f}" if isinstance(nearest_sim, float) else nearest_sim,
        ])

    # Sort by F1 ascending (worst first) if F1 is available, else by count
    if per_intent_f1:
        rows.sort(key=lambda r: float(r[3]) if r[3] != "—" else 999)
    else:
        rows.sort(key=lambda r: r[1])

    print(f"\n{'='*90}")
    print(f"INTENT INVENTORY ({len(intent_names)} intents, {len(train_df)} total examples)")
    print(f"{'='*90}")
    print(tabulate(
        rows,
        headers=["Intent", "Examples", "Cohesion", "F1", "Nearest Neighbor", "Similarity"],
        tablefmt="simple",
    ))
    print(f"\nNote: Sorted by F1 (worst first). '—' means no evaluation run exists yet.")


def query_examples(intent: str, split: str, search: str, sample: int, show_description: bool):
    """Query examples for a specific intent."""
    config = load_config()

    if show_description:
        desc = load_intent_description(config, intent)
        print(f"\n--- Description for '{intent}' ---")
        print(desc)
        print("---\n")

    if split == "train":
        csv_path = get_path(config, "train_dir") / intent / "examples.csv"
        if not csv_path.exists():
            print(f"No training data found for intent '{intent}'")
            return
        df = pd.read_csv(csv_path)
        df["intent"] = intent
    else:
        df = load_split_data(config, split)
        col = "true_intent" if "true_intent" in df.columns else "intent"
        df = df[df[col] == intent]

    if search:
        df = df[df["text"].str.contains(search, case=False, na=False)]

    total = len(df)
    if sample and sample < total:
        df = df.sample(sample, random_state=42)

    print(f"Intent: {intent} | Split: {split} | Showing {len(df)}/{total} examples")
    if search:
        print(f"Search filter: '{search}'")
    print()

    for idx, row in df.iterrows():
        print(f"  [{idx:>4}] {row['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query intent data")
    parser.add_argument("--inventory", action="store_true",
                        help="Show overview of all intents")
    parser.add_argument("--intent", type=str, help="Intent name to query")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train",
                        help="Data split (default: train)")
    parser.add_argument("--search", type=str, default=None,
                        help="Keyword search within examples")
    parser.add_argument("--sample", type=int, default=20,
                        help="Max examples to show (default: 20)")
    parser.add_argument("--show-description", action="store_true",
                        help="Print the intent's description.md")
    args = parser.parse_args()

    if args.inventory:
        show_inventory()
    elif args.intent:
        query_examples(args.intent, args.split, args.search, args.sample, args.show_description)
    else:
        parser.print_help()
