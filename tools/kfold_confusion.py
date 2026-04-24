"""K-fold cross-validation to identify most confused intent pairs.

Also emits per-example out-of-fold misclassifications so analyze_errors.py
can inspect specific errors without touching the val/test set.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from utils import load_config, get_path, load_training_data, compute_embeddings, get_skill


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
    misclass_rows = []
    texts = filtered_df["text"].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        model = LogisticRegression(
            solver=model_config["solver"],
            max_iter=model_config["max_iter"],
            C=model_config["C"],
            class_weight=model_config["class_weight"],
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)

        for i, (true_label, pred_label) in enumerate(zip(y_val, predictions)):
            if true_label != pred_label:
                true_name = le.inverse_transform([true_label])[0]
                pred_name = le.inverse_transform([pred_label])[0]
                # Store as sorted tuple so A→B and B→A count together
                pair = tuple(sorted([true_name, pred_name]))
                confusion_pairs[pair] += 1
                misclass_rows.append({
                    "text": texts[val_idx[i]],
                    "true_intent": true_name,
                    "predicted_intent": pred_name,
                    "confidence": round(float(probabilities[i].max()), 4),
                    "fold": fold,
                })

    total_examples = len(filtered_df)
    total_errors = sum(confusion_pairs.values())
    overall_accuracy = 1.0 - (total_errors / total_examples) if total_examples else 0.0

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

    misclass_df = pd.DataFrame(misclass_rows, columns=["text", "true_intent", "predicted_intent", "confidence", "fold"])
    misclass_df.to_csv(results_dir / "kfold_misclassifications.csv", index=False)

    # Print results
    print(f"\n{'='*60}")
    print(f"TOP {top_n} CONFUSED INTENT PAIRS ({k}-fold CV)")
    print(f"{'='*60}")

    table = []
    same_skill_errors = 0
    cross_skill_errors = 0
    for pair, count in top_pairs:
        skill_a = get_skill(pair[0]) or "?"
        skill_b = get_skill(pair[1]) or "?"
        if skill_a == skill_b:
            skills_col = f"{skill_a} (same)"
            same_skill_errors += count
        else:
            skills_col = f"{skill_a} / {skill_b}"
            cross_skill_errors += count
        table.append([pair[0], pair[1], count, skills_col])
    print(tabulate(table, headers=["Intent A", "Intent B", "Confusions", "Skills"], tablefmt="simple"))
    if top_pairs:
        print(f"\n  (Top {top_n} breakdown: {same_skill_errors} same-skill, {cross_skill_errors} cross-skill)")

    print(f"\nOverall k-fold accuracy: {overall_accuracy:.4f} ({total_examples - total_errors}/{total_examples} correct)")
    print(f"Total misclassifications across folds: {total_errors}")
    if top_pairs:
        top_pct = sum(c for _, c in top_pairs) / total_errors * 100
        print(f"Top {top_n} pairs account for {top_pct:.1f}% of all errors")
    print(f"\nPer-example misclassifications: {results_dir / 'kfold_misclassifications.csv'}")


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
