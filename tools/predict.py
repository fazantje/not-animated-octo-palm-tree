"""Predict intent for a single sentence using the trained model."""

import argparse

import numpy as np
from tabulate import tabulate

from utils import load_config, load_trained_model, compute_embeddings


def predict(text: str):
    config = load_config()
    top_k = config["predict"]["top_k"]

    # Load trained model
    try:
        saved = load_trained_model()
    except FileNotFoundError as e:
        print(str(e))
        return

    model = saved["model"]
    le = saved["label_encoder"]

    # Embed the input
    embedding = compute_embeddings([text], config)

    # Predict
    probabilities = model.predict_proba(embedding)[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    print(f"\nPrediction for: \"{text}\"")
    print(f"{'='*60}")

    rows = []
    for rank, idx in enumerate(top_indices, 1):
        intent_name = le.inverse_transform([idx])[0]
        prob = probabilities[idx]
        bar = "█" * int(prob * 30)
        rows.append([rank, intent_name, f"{prob:.4f}", bar])

    print(tabulate(rows, headers=["#", "Intent", "Probability", ""], tablefmt="simple"))

    # Decision insight
    top_prob = probabilities[top_indices[0]]
    second_prob = probabilities[top_indices[1]] if len(top_indices) > 1 else 0
    margin = top_prob - second_prob

    print(f"\nTop prediction: {le.inverse_transform([top_indices[0]])[0]} ({top_prob:.4f})")
    print(f"Margin over #2: {margin:.4f}")

    if margin < 0.15:
        print("⚠ Low margin — this example would likely go to LLM disambiguation (stage 2)")
    elif margin < 0.30:
        print("⚡ Moderate margin — borderline case")
    else:
        print("✓ Clear prediction — classifier is confident")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict intent for a single sentence")
    parser.add_argument("--text", required=True, help="Sentence to predict")
    args = parser.parse_args()
    predict(args.text)
