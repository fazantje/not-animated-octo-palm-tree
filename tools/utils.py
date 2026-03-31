"""Shared utilities for intent optimizer tools."""

import yaml
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

# Project root is one level up from tools/
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "results" / "current" / "model.pkl"


def load_config() -> dict:
    """Load project configuration."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_path(config: dict, key: str) -> Path:
    """Resolve a path from config relative to project root."""
    return PROJECT_ROOT / config["paths"][key]


def load_embeddings_cache(config: dict) -> pd.DataFrame:
    """Load the embeddings cache. Returns empty DataFrame if not found."""
    cache_path = get_path(config, "embeddings_cache")
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame(columns=["text", "vector"])


def save_embeddings_cache(config: dict, cache_df: pd.DataFrame):
    """Save the embeddings cache."""
    cache_path = get_path(config, "embeddings_cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)


def compute_embeddings(texts: list[str], config: dict, cache_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Compute embeddings for a list of texts, using cache where possible.
    Returns numpy array of shape (len(texts), dimensions).
    Updates cache in-place and saves.
    """
    from openai import OpenAI

    if cache_df is None:
        cache_df = load_embeddings_cache(config)

    emb_config = config["embeddings"]
    dimensions = emb_config["dimensions"]
    batch_size = emb_config["batch_size"]

    # Check cache — use dict for O(1) lookup instead of scanning the DataFrame
    results = {}
    if len(cache_df) > 0:
        cache_lookup = dict(zip(cache_df["text"], cache_df["vector"]))
        for text in texts:
            if text in cache_lookup:
                results[text] = np.array(cache_lookup[text])


    # Compute missing embeddings
    missing = [t for t in texts if t not in results]
    if missing:
        client = OpenAI()
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            response = client.embeddings.create(
                model=emb_config["model"],
                input=batch,
                dimensions=dimensions,
            )
            new_rows = []
            for text, emb_obj in zip(batch, response.data):
                vec = np.array(emb_obj.embedding, dtype=np.float32)
                results[text] = vec
                new_rows.append({"text": text, "vector": vec})

            # Append to cache
            new_df = pd.DataFrame(new_rows)
            cache_df = pd.concat([cache_df, new_df], ignore_index=True)

        save_embeddings_cache(config, cache_df)

    # Return in original order
    return np.array([results[t] for t in texts])


def load_training_data(config: dict) -> pd.DataFrame:
    """
    Load all training data. Returns DataFrame with columns: text, intent.
    """
    train_dir = get_path(config, "train_dir")
    rows = []
    for intent_dir in sorted(train_dir.iterdir()):
        if not intent_dir.is_dir():
            continue
        csv_path = intent_dir / "examples.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["intent"] = intent_dir.name
            rows.append(df[["text", "intent"]])
    if not rows:
        raise ValueError(f"No training data found in {train_dir}")
    return pd.concat(rows, ignore_index=True)


def load_split_data(config: dict, split: str) -> pd.DataFrame:
    """Load validation or test data. Returns DataFrame with columns: text, true_intent."""
    if split == "val":
        path = get_path(config, "validation_file")
    elif split == "test":
        path = get_path(config, "test_file")
    else:
        raise ValueError(f"Unknown split: {split}")
    return pd.read_csv(path)


def load_intent_description(config: dict, intent_name: str) -> str:
    """Load the description.md for an intent."""
    desc_path = get_path(config, "train_dir") / intent_name / "description.md"
    if desc_path.exists():
        return desc_path.read_text().strip()
    return "(no description found)"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Compute centroid of a set of embeddings."""
    return embeddings.mean(axis=0)


def load_trained_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train_and_evaluate.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def save_trained_model(model, label_encoder):
    """Save trained model and label encoder."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": label_encoder}, f)


def get_intent_names(config: dict) -> list[str]:
    """Get sorted list of intent directory names."""
    train_dir = get_path(config, "train_dir")
    return sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and (d / "examples.csv").exists()
    ])


def append_changelog(config: dict, entry: str):
    """Append an entry to the changelog."""
    changelog_path = get_path(config, "changelog")
    with open(changelog_path, "a") as f:
        f.write(entry + "\n\n")
