"""Add or remove training examples with guardrails and automatic changelog logging."""

import argparse
import json
from datetime import datetime

import pandas as pd

from utils import (
    load_config, get_path, compute_embeddings, append_changelog,
)


def modify_train_data(action: str, intent: str, examples: list[str] = None,
                      indices: list[int] = None, reason: str = ""):
    config = load_config()
    guardrails = config["guardrails"]

    intent_dir = get_path(config, "train_dir") / intent
    csv_path = intent_dir / "examples.csv"

    if not intent_dir.exists():
        print(f"Error: Intent directory '{intent}' not found.")
        return

    if not csv_path.exists():
        print(f"Error: No examples.csv found for intent '{intent}'.")
        return

    df = pd.read_csv(csv_path)
    original_count = len(df)

    if action == "add":
        if not examples:
            print("Error: --examples required for add action.")
            return

        # Compute embeddings to ensure they're cached
        print(f"Computing embeddings for {len(examples)} new examples...")
        compute_embeddings(examples, config)

        # Add to CSV
        new_rows = pd.DataFrame({"text": examples})
        df = pd.concat([df, new_rows], ignore_index=True)
        df.to_csv(csv_path, index=False)

        print(f"Added {len(examples)} examples to '{intent}'")
        print(f"  Before: {original_count} | After: {len(df)}")
        for ex in examples:
            print(f"  + {ex}")

        # Log
        entry = f"### ADD — {datetime.now().strftime('%Y-%m-%d %H:%M')} | Intent: `{intent}`\n"
        entry += f"- **Reason**: {reason}\n"
        entry += f"- **Examples added** ({len(examples)}):\n"
        for ex in examples:
            entry += f"  - `{ex}`\n"
        entry += f"- **Count**: {original_count} → {len(df)}"
        append_changelog(config, entry)

    elif action == "remove":
        if not indices:
            print("Error: --indices required for remove action.")
            return

        # Validate indices
        invalid = [i for i in indices if i < 0 or i >= original_count]
        if invalid:
            print(f"Error: Invalid indices: {invalid}. Valid range: 0-{original_count - 1}")
            return

        # Guardrail: max removal percentage
        remove_pct = len(indices) / original_count
        max_pct = guardrails["max_remove_pct"]
        if remove_pct > max_pct:
            print(f"GUARDRAIL: Refusing to remove {len(indices)} examples ({remove_pct:.0%} of {original_count}).")
            print(f"  Maximum allowed: {max_pct:.0%} ({int(original_count * max_pct)} examples) per call.")
            return

        # Guardrail: minimum examples
        remaining = original_count - len(indices)
        min_examples = guardrails["min_examples_per_intent"]
        if remaining < min_examples:
            print(f"GUARDRAIL: Refusing removal. Would leave {remaining} examples, minimum is {min_examples}.")
            return

        # Show what's being removed
        removed_texts = df.iloc[indices]["text"].tolist()
        print(f"Removing {len(indices)} examples from '{intent}':")
        for idx, text in zip(indices, removed_texts):
            print(f"  - [{idx}] {text}")

        # Remove
        df = df.drop(index=indices).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

        print(f"\n  Before: {original_count} | After: {len(df)}")

        # Log
        entry = f"### REMOVE — {datetime.now().strftime('%Y-%m-%d %H:%M')} | Intent: `{intent}`\n"
        entry += f"- **Reason**: {reason}\n"
        entry += f"- **Examples removed** ({len(indices)}):\n"
        for idx, text in zip(indices, removed_texts):
            entry += f"  - [{idx}] `{text}`\n"
        entry += f"- **Count**: {original_count} → {len(df)}"
        append_changelog(config, entry)

    else:
        print(f"Error: Unknown action '{action}'. Use 'add' or 'remove'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify training data")
    parser.add_argument("--action", required=True, choices=["add", "remove"],
                        help="Action to perform")
    parser.add_argument("--intent", required=True,
                        help="Intent name")
    parser.add_argument("--examples", type=str, default=None,
                        help='JSON list of examples to add, e.g. \'["text1","text2"]\'')
    parser.add_argument("--indices", type=str, default=None,
                        help='JSON list of row indices to remove, e.g. \'[3,7,12]\'')
    parser.add_argument("--reason", type=str, default="(no reason given)",
                        help="Reason for the modification")
    args = parser.parse_args()

    examples = json.loads(args.examples) if args.examples else None
    indices = json.loads(args.indices) if args.indices else None

    modify_train_data(args.action, args.intent, examples, indices, args.reason)
