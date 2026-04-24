"""Merge one intent into another. The source intent is deleted.

Refuses to merge a described intent (source cannot have description.md).
Target can be either described or non-described — moving non-described
into described is the common case during SME alignment.
"""

import argparse
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


def merge_intents(source: str, target: str, reason: str):
    config = load_config()
    train_dir = get_path(config, "train_dir")

    src_dir = train_dir / source
    dst_dir = train_dir / target
    src_csv = src_dir / "examples.csv"
    dst_csv = dst_dir / "examples.csv"

    if source == target:
        print("Error: source and target are the same.")
        return
    if not src_dir.exists():
        print(f"Error: source intent '{source}' not found.")
        return
    if not dst_dir.exists():
        print(f"Error: target intent '{target}' not found.")
        return
    if not src_csv.exists():
        print(f"Error: no examples.csv in source intent '{source}'.")
        return
    if not dst_csv.exists():
        print(f"Error: no examples.csv in target intent '{target}'.")
        return

    # Guardrail: source must not be described
    if (src_dir / "description.md").exists():
        print(f"GUARDRAIL: source intent '{source}' has description.md — it is SME-final and cannot be merged away.")
        print("  If you want to move its examples out, use move_examples.py instead (keeps the intent).")
        return

    src_df = pd.read_csv(src_csv)
    dst_df = pd.read_csv(dst_csv)
    src_count = len(src_df)
    dst_count_before = len(dst_df)

    # Dedupe against target
    existing = set(dst_df["text"].values) if dst_count_before else set()
    to_append = src_df[~src_df["text"].isin(existing)]
    skipped = src_count - len(to_append)
    if skipped:
        print(f"Note: {skipped} text(s) already present in '{target}' — will be dropped without duplicating.")

    merged = pd.concat([dst_df, to_append[["text"]]], ignore_index=True)
    merged.to_csv(dst_csv, index=False)

    # Delete source folder entirely (non-described, so no description.md to worry about)
    shutil.rmtree(src_dir)

    print(f"\nMerged '{source}' into '{target}'")
    print(f"  Source: {src_count} examples (folder deleted)")
    print(f"  Target: {dst_count_before} → {len(merged)} examples ({skipped} duplicates dropped)")

    # Changelog
    entry = f"### MERGE — {datetime.now().strftime('%Y-%m-%d %H:%M')} | {source} → {target}\n"
    entry += f"- **Reason**: {reason}\n"
    entry += f"- **Source deleted**: `{source}` ({src_count} examples)\n"
    entry += f"- **Target count**: {dst_count_before} → {len(merged)}"
    if skipped:
        entry += f" ({skipped} duplicates skipped)"
    append_changelog(config, entry)

    paths = [str(src_dir.parent), str(dst_dir), str(get_path(config, "changelog"))]
    if git_commit_restructure(f"merge {source} into {target}", paths):
        print(f"  [git] committed: restructure: merge {source} into {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge one intent into another (source deleted)")
    parser.add_argument("--source", required=True, help="Source intent (will be deleted)")
    parser.add_argument("--target", required=True, help="Target intent (receives source's examples)")
    parser.add_argument("--reason", type=str, default="(no reason given)",
                        help="Reason for the merge")
    args = parser.parse_args()
    merge_intents(args.source, args.target, args.reason)
