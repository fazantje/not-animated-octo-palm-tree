"""K-fold cross-validation to identify most confused intent pairs."""

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from utils import load_config, get_path, load_training_data, compute_embeddings


def kfold_confusion(k: int, top_n: int):
    config = load_config()
    model_config = config["model"]

    # Load data
    print("Loading training data...")
    train_df = load_training_data(config)
    print(f"  {len(train_df)} examples across {train_df['intent'].nunique()} intents")

    # Filter intents with too few examples for k-fold
    intent_counts = train_df["intent"].value_counts()
    valid_intents = intent_counts[intent_counts >= k].index
    filtered_df = train_df[train_df["intent"].isin(valid_intents)].copy()
    dropped = len(train_df) - len(filtered_df)
    if dropped > 0:
        print(f"  Dropped {dropped} examples from intents with <{k} examples")

    # Compute embeddings
    print("Computing/loading embeddings...")
    embeddings = compute_embeddings(list(filtered_df["text"].values), config)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(filtered_df["intent"].values)

    # K-fold CV
    print(f"Running {k}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    confusion_pairs = Counter()

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = LogisticRegression(
            solver=model_config["solver"],
            max_iter=model_config["max_iter"],
            C=model_config["C"],
            class_weight=model_config["class_weight"],
            multi_class=model_config["multi_class"],
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        # Count confusion pairs
        for true_label, pred_label in zip(y_val, predictions):
            if true_label != pred_label:
                true_name = le.inverse_transform([true_label])[0]
                pred_name = le.inverse_transform([pred_label])[0]
                # Store as sorted tuple so A→B and B→A count together
                pair = tuple(sorted([true_name, pred_name]))
                confusion_pairs[pair] += 1

    # Get top N
    top_pairs = confusion_pairs.most_common(top_n)

    # Save results
    results_dir = get_path(config, "results_dir") / "current"
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs_data = [
        {
            "intent_a": pair[0],
            "intent_b": pair[1],
            "confusion_count": count,
        }
        for pair, count in top_pairs
    ]
    with open(results_dir / "confusion_pairs.json", "w") as f:
        json.dump(pairs_data, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"TOP {top_n} CONFUSED INTENT PAIRS ({k}-fold CV)")
    print(f"{'='*60}")

    table = [
        [pair[0], pair[1], count]
        for pair, count in top_pairs
    ]
    print(tabulate(table, headers=["Intent A", "Intent B", "Confusions"], tablefmt="simple"))

    total_errors = sum(confusion_pairs.values())
    print(f"\nTotal misclassifications across folds: {total_errors}")
    if top_pairs:
        top_pct = sum(c for _, c in top_pairs) / total_errors * 100
        print(f"Top {top_n} pairs account for {top_pct:.1f}% of all errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-fold confusion analysis")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of folds (default: from config)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Number of top confused pairs to show (default: from config)")
    args = parser.parse_args()

    config = load_config()
    k = args.k or config["kfold"]["k"]
    top_n = args.top_n or config["kfold"]["top_n_confused_pairs"]
    kfold_confusion(k, top_n)
