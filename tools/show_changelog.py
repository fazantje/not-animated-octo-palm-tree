"""Show recent changelog entries — the agent's memory across sessions."""

import argparse
import re
import sys
from pathlib import Path

# Ensure utils is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, get_path


def show_changelog(last: int = 10, intent: str = None):
    config = load_config()
    changelog_path = get_path(config, "changelog")

    if not changelog_path.exists():
        print("No changelog found.")
        return

    content = changelog_path.read_text()

    # Split into entries (each starts with ###)
    entries = re.split(r'(?=^### )', content, flags=re.MULTILINE)
    entries = [e.strip() for e in entries if e.strip() and e.strip().startswith("###")]

    if not entries:
        # Check for reflection entries (## Reflection)
        entries = re.split(r'(?=^## )', content, flags=re.MULTILINE)
        entries = [e.strip() for e in entries if e.strip() and e.strip().startswith("##")]

    if not entries:
        print("Changelog is empty. No modifications have been recorded yet.")
        return

    # Filter by intent if specified
    if intent:
        entries = [e for e in entries if intent.lower() in e.lower()]
        if not entries:
            print(f"No changelog entries found for intent '{intent}'.")
            return

    # Take last N
    entries = entries[-last:]

    print(f"{'='*60}")
    print(f"CHANGELOG — Showing {len(entries)} entries" +
          (f" (filtered: {intent})" if intent else ""))
    print(f"{'='*60}\n")

    for entry in entries:
        print(entry)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show optimization changelog")
    parser.add_argument("--last", type=int, default=10,
                        help="Number of recent entries to show (default: 10)")
    parser.add_argument("--intent", type=str, default=None,
                        help="Filter entries by intent name")
    args = parser.parse_args()
    show_changelog(args.last, args.intent)
