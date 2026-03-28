"""Train LogisticRegression on intent data and evaluate on a split."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from utils import (
    load_config, get_path, load_training_data, load_split_data,
    compute_embeddings, save_trained_model,
)


def train_and_evaluate(split: str, run_name: str):
    config = load_config()

    # Load training data
    print(f"Loading training data...")
    train_df = load_training_data(config)
    print(f"  {len(train_df)} examples across {train_df['intent'].nunique()} intents")

    # Load evaluation data
    print(f"Loading {split} data...")
    eval_df = load_split_data(config, split)
    print(f"  {len(eval_df)} examples")

    # Compute embeddings
    print("Computing/loading embeddings...")
    all_texts = list(train_df["text"].values) + list(eval_df["text"].values)
    all_embeddings = compute_embeddings(all_texts, config)

    train_embeddings = all_embeddings[:len(train_df)]
    eval_embeddings = all_embeddings[len(train_df):]

    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df["intent"].values)
    eval_labels = le.transform(eval_df["true_intent"].values)

    # Train model
    model_config = config["model"]
    print("Training LogisticRegression...")
    model = LogisticRegression(
        solver=model_config["solver"],
        max_iter=model_config["max_iter"],
        C=model_config["C"],
        class_weight=model_config["class_weight"],
        multi_class=model_config["multi_class"],
    )
    model.fit(train_embeddings, train_labels)

    # Save model
    save_trained_model(model, le)

    # Evaluate
    predictions = model.predict(eval_embeddings)
    probabilities = model.predict_proba(eval_embeddings)

    # Classification report
    report = classification_report(
        eval_labels, predictions,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    overall_f1 = f1_score(eval_labels, predictions, average="macro", zero_division=0)
    overall_accuracy = np.mean(predictions == eval_labels)

    # Save full results
    results_dir = get_path(config, "results_dir") / "current"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics = {
        "run_name": run_name,
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "overall_accuracy": round(float(overall_accuracy), 4),
        "overall_macro_f1": round(float(overall_f1), 4),
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "per_intent": {},
    }
    for intent_name in le.classes_:
        if intent_name in report:
            r = report[intent_name]
            metrics["per_intent"][intent_name] = {
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1-score"], 4),
                "support": int(r["support"]),
            }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Misclassifications CSV
    misclassified_mask = predictions != eval_labels
    if misclassified_mask.any():
        misc_df = eval_df[misclassified_mask].copy()
        misc_df["predicted_intent"] = le.inverse_transform(predictions[misclassified_mask])
        misc_df["confidence"] = probabilities[misclassified_mask].max(axis=1).round(4)
        misc_df.to_csv(results_dir / "misclassifications.csv", index=False)

    # Copy to history
    history_dir = get_path(config, "results_dir") / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted(history_dir.iterdir()) if history_dir.exists() else []
    run_num = len(existing_runs) + 1
    run_dir = history_dir / f"run_{run_num:03d}_{run_name}"
    shutil.copytree(results_dir, run_dir)

    # Print compact summary
    print(f"\n{'='*60}")
    print(f"RUN: {run_name} | Split: {split} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Macro F1: {overall_f1:.4f}")
    print(f"Training examples: {len(train_df)} | Eval examples: {len(eval_df)}")
    print(f"Results saved to: {run_dir}")

    # Top 5 worst intents by F1
    intent_f1 = [
        (name, metrics["per_intent"][name]["f1"], metrics["per_intent"][name]["support"])
        for name in le.classes_
        if name in metrics["per_intent"]
    ]
    intent_f1.sort(key=lambda x: x[1])
    print(f"\nTop 5 worst intents by F1:")
    print(tabulate(
        [(name, f"{f1:.4f}", sup) for name, f1, sup in intent_f1[:5]],
        headers=["Intent", "F1", "Support"],
        tablefmt="simple",
    ))

    # Top 5 best intents by F1
    print(f"\nTop 5 best intents by F1:")
    print(tabulate(
        [(name, f"{f1:.4f}", sup) for name, f1, sup in intent_f1[-5:]],
        headers=["Intent", "F1", "Support"],
        tablefmt="simple",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate intent classifier")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Evaluation split (default: val)")
    parser.add_argument("--run-name", default="unnamed",
                        help="Descriptive name for this run")
    args = parser.parse_args()
    train_and_evaluate(args.split, args.run_name)
