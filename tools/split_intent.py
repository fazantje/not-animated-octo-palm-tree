"""Split one intent into multiple sub-intents based on an explicit assignment map.

Refuses to split a described intent (SME-final). Every example in the source
must be assigned to exactly one new intent — no orphans, no duplicates.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from utils import (
    load_config, get_path, append_changelog, git_commit_restructure,
)


def split_intent(intent: str, assignments: dict[str, list[str]], reason: str):
    config = load_config()
    train_dir = get_path(config, "train_dir")

    src_dir = train_dir / intent
    src_csv = src_dir / "examples.csv"

    if not src_dir.exists():
        print(f"Error: intent '{intent}' not found.")
        return
    if not src_csv.exists():
        print(f"Error: no examples.csv in intent '{intent}'.")
        return

    # Guardrail: described intents cannot be split
    if (src_dir / "description.md").exists():
        print(f"GUARDRAIL: intent '{intent}' has description.md — splitting a described intent is a world-shattering change.")
        print("  Stop the session and surface to the user via chat instead of attempting this.")
        return

    if not assignments:
        print("Error: --assignments is required and must not be empty.")
        return

    if intent in assignments:
        print(f"Error: a new intent cannot share the name of the source ('{intent}').")
        return

    # Check new-intent name collisions with existing folders
    collisions = [name for name in assignments if (train_dir / name).exists()]
    if collisions:
        print(f"Error: new intent name(s) collide with existing folder(s): {collisions}")
        return

    src_df = pd.read_csv(src_csv)
    src_texts = src_df["text"].tolist()
    src_set = set(src_texts)

    # Validate assignments: every text assigned exactly once, no unknowns
    assigned_flat = []
    for new_name, texts in assignments.items():
        assigned_flat.extend(texts)
    assigned_set = set(assigned_flat)

    if len(assigned_flat) != len(assigned_set):
        dups = [t for t in assigned_set if assigned_flat.count(t) > 1]
        print(f"Error: {len(dups)} text(s) assigned to multiple new intents:")
        for t in dups[:10]:
            print(f"  - {t}")
        return

    unknown = assigned_set - src_set
    if unknown:
        print(f"Error: {len(unknown)} text(s) in assignments but not in source intent:")
        for t in list(unknown)[:10]:
            print(f"  - {t}")
        return

    orphans = src_set - assigned_set
    if orphans:
        print(f"Error: {len(orphans)} text(s) in source not assigned to any new intent:")
        for t in list(orphans)[:10]:
            print(f"  - {t}")
        print("  Either assign them, or remove them with modify_train_data.py first.")
        return

    # All validated — perform the split
    for new_name, texts in assignments.items():
        new_dir = train_dir / new_name
        new_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": texts}).to_csv(new_dir / "examples.csv", index=False)
        print(f"  Created '{new_name}' with {len(texts)} examples")

    # Delete source folder
    shutil.rmtree(src_dir)
    print(f"\nSplit '{intent}' ({len(src_texts)} examples) into {len(assignments)} new intents.")
    print(f"  Source folder deleted.")

    # Changelog
    entry = f"### SPLIT — {datetime.now().strftime('%Y-%m-%d %H:%M')} | {intent}\n"
    entry += f"- **Reason**: {reason}\n"
    entry += f"- **Source deleted**: `{intent}` ({len(src_texts)} examples)\n"
    entry += f"- **New intents**:\n"
    for new_name, texts in assignments.items():
        entry += f"  - `{new_name}` ({len(texts)} examples)\n"
    append_changelog(config, entry)

    paths = [str(train_dir), str(get_path(config, "changelog"))]
    new_names_str = ", ".join(assignments.keys())
    if git_commit_restructure(f"split {intent} into {new_names_str}", paths):
        print(f"  [git] committed: restructure: split {intent} into {new_names_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split one intent into multiple sub-intents")
    parser.add_argument("--intent", required=True, help="Intent to split (will be deleted)")
    parser.add_argument("--assignments", required=True, type=str,
                        help='JSON dict mapping new intent name → list of texts, '
                             'e.g. \'{"X_refund":["t1","t2"],"X_status":["t3"]}\'')
    parser.add_argument("--reason", type=str, default="(no reason given)",
                        help="Reason for the split")
    args = parser.parse_args()

    try:
        assignments = json.loads(args.assignments)
    except json.JSONDecodeError as e:
        print(f"Error: --assignments is not valid JSON: {e}")
        sys.exit(1)

    if not isinstance(assignments, dict):
        print("Error: --assignments must be a JSON object (dict).")
        sys.exit(1)

    split_intent(args.intent, assignments, args.reason)
