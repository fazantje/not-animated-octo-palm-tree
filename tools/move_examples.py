"""Relabel training examples from one intent to another.

The most common structural operation. Accepts either a list of exact texts
or --all to move every example in the source.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from utils import (
    load_config, get_path, append_changelog, git_commit_restructure,
)


def move_examples(from_intent: str, to_intent: str, texts: list[str] | None,
                  move_all: bool, reason: str):
    config = load_config()
    guardrails = config["guardrails"]
    train_dir = get_path(config, "train_dir")

    src_dir = train_dir / from_intent
    dst_dir = train_dir / to_intent
    src_csv = src_dir / "examples.csv"
    dst_csv = dst_dir / "examples.csv"

    if not src_dir.exists():
        print(f"Error: source intent '{from_intent}' not found.")
        return
    if not src_csv.exists():
        print(f"Error: no examples.csv in source intent '{from_intent}'.")
        return
    if from_intent == to_intent:
        print("Error: source and target intents are the same.")
        return

    src_df = pd.read_csv(src_csv)
    original_src_count = len(src_df)

    # Resolve which texts to move
    if move_all:
        to_move = src_df["text"].tolist()
        not_found = []
    else:
        if not texts:
            print("Error: provide --texts or --all.")
            return
        existing = set(src_df["text"].values)
        to_move = [t for t in texts if t in existing]
        not_found = [t for t in texts if t not in existing]
        if not_found:
            print(f"WARNING: {len(not_found)} text(s) not found in '{from_intent}':")
            for t in not_found:
                print(f"  - {t}")
        if not to_move:
            print("No matching examples to move.")
            return

    # Described-intent guardrail: don't hollow out a described source
    has_description = (src_dir / "description.md").exists()
    remaining = original_src_count - len(to_move)
    min_examples = guardrails["min_examples_per_intent"]
    if has_description and remaining < min_examples and remaining > 0:
        print(f"GUARDRAIL: '{from_intent}' has description.md; moving would leave {remaining} examples (min {min_examples}).")
        print("  Refusing to hollow out a described intent. Move fewer examples or drop description.md first.")
        return

    # Create or extend destination
    dst_created = not dst_dir.exists()
    if dst_created:
        dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_csv.exists():
        dst_df = pd.read_csv(dst_csv)
    else:
        dst_df = pd.DataFrame(columns=["text"])

    # Avoid duplicates in target
    already_in_dst = set(dst_df["text"].values) if len(dst_df) else set()
    skipped_duplicate = [t for t in to_move if t in already_in_dst]
    to_actually_append = [t for t in to_move if t not in already_in_dst]
    if skipped_duplicate:
        print(f"Note: {len(skipped_duplicate)} text(s) already exist in '{to_intent}'; removing from source without duplicating:")
        for t in skipped_duplicate:
            print(f"  ~ {t}")

    dst_df = pd.concat([dst_df, pd.DataFrame({"text": to_actually_append})], ignore_index=True)
    dst_df.to_csv(dst_csv, index=False)

    # Remove from source
    src_df = src_df[~src_df["text"].isin(to_move)].reset_index(drop=True)
    src_df.to_csv(src_csv, index=False)

    print(f"\nMoved {len(to_move)} examples: '{from_intent}' → '{to_intent}'")
    print(f"  {from_intent}: {original_src_count} → {len(src_df)}")
    print(f"  {to_intent}: {len(dst_df) - len(to_actually_append)} → {len(dst_df)}")

    # Cleanup: delete empty non-described source folder
    deleted_src = False
    if len(src_df) == 0 and not has_description:
        src_csv.unlink()
        # Remove folder if empty
        try:
            src_dir.rmdir()
            deleted_src = True
            print(f"  Deleted empty source folder: {src_dir.relative_to(train_dir.parent)}")
        except OSError:
            # Folder not empty (other files present); leave it
            pass

    # Changelog
    entry = f"### MOVE — {datetime.now().strftime('%Y-%m-%d %H:%M')} | {from_intent} → {to_intent}\n"
    entry += f"- **Reason**: {reason}\n"
    entry += f"- **Examples moved** ({len(to_move)}):\n"
    for t in to_move[:20]:
        entry += f"  - `{t}`\n"
    if len(to_move) > 20:
        entry += f"  - ...and {len(to_move) - 20} more\n"
    entry += f"- **Counts**: {from_intent} {original_src_count}→{len(src_df)}, {to_intent} {len(dst_df) - len(to_actually_append)}→{len(dst_df)}"
    if deleted_src:
        entry += f"\n- **Source folder deleted** (was empty, non-described)"
    append_changelog(config, entry)

    # Auto-commit
    paths = [str(src_dir), str(dst_dir)] if src_dir.exists() else [str(dst_dir), str(train_dir)]
    paths.append(str(get_path(config, "changelog")))
    msg = f"move {len(to_move)} examples {from_intent}→{to_intent}"
    if git_commit_restructure(msg, paths):
        print(f"  [git] committed: restructure: {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move training examples between intents")
    parser.add_argument("--from-intent", required=True, help="Source intent")
    parser.add_argument("--to-intent", required=True, help="Target intent (created if missing)")
    parser.add_argument("--texts", type=str, default=None,
                        help='JSON list of exact texts to move, e.g. \'["text1","text2"]\'')
    parser.add_argument("--all", action="store_true",
                        help="Move every example from the source intent")
    parser.add_argument("--reason", type=str, default="(no reason given)",
                        help="Reason for the move")
    args = parser.parse_args()

    if args.texts and args.all:
        print("Error: use either --texts or --all, not both.")
        sys.exit(1)

    texts = json.loads(args.texts) if args.texts else None
    move_examples(args.from_intent, args.to_intent, texts, args.all, args.reason)
