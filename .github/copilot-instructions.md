# Intent Optimizer - Copilot Agent Instructions

## Project Overview

This project optimizes a LogisticRegression intent classifier used as the first stage of a two-stage classification system. The second stage is an LLM that disambiguates using intent descriptions when the classifier is uncertain. Your job is to improve the classifier's accuracy by curating training data.

The classifier works on OpenAI `text-embedding-3-large` embeddings. Each intent has a folder in `data/train/` with an `examples.csv` (text column) and a `description.md` (intent description used by the LLM disambiguation stage).

## Architecture Context

- **~40-50 intents**, each with 20-150 training examples
- Embeddings are cached in `data/embeddings_cache.parquet` (text → vector lookup)
- Validation set: `data/validation/validation.csv` (~500 examples, all intents)
- Test set: `data/test/test.csv` — **DO NOT TOUCH until final evaluation**
- Model config is fixed in `config.yaml` — do not change hyperparameters

## Available Tools

All tools are in the `tools/` directory. Run them with `python tools/<tool>.py <args>`.

### Diagnostic Tools
- **`train_and_evaluate.py`** — Train the model and evaluate on a split
  - `--split val|test` (default: val) `--run-name "descriptive name"`
  - Writes results to `results/current/` and `results/history/run_XXX/`
  - Prints compact summary to stdout
- **`kfold_confusion.py`** — K-fold CV to find confused intent pairs
  - `--k 5` `--top-n 10`
  - Outputs ranked confused pairs to `results/current/confusion_pairs.json`

### Inspection Tools
- **`query_data.py`** — Explore training/validation/test data
  - `--inventory` — overview of all intents with counts, cohesion, F1, nearest neighbor
  - `--intent X --split train|val|test` — view examples for an intent
  - `--intent X --search "keyword"` — keyword search within an intent
  - `--sample N` — limit output rows (default: 20)
  - `--show-description` — also print the intent's description.md
- **`analyze_errors.py`** — Deep dive into confusion between two intents
  - `--intent-a X --intent-b Y`
  - Shows misclassified examples, confidence scores, and both descriptions
- **`predict.py`** — Test a single sentence against the trained model
  - `--text "sentence to test"`
  - Returns top-k intents with probabilities

### Action Tools
- **`modify_train_data.py`** — Add or remove training examples
  - `--action add --intent X --examples '["text1","text2"]' --reason "explanation"`
  - `--action remove --intent X --texts '["text1","text2"]' --reason "explanation"` (preferred — immune to index shifts)
  - `--action remove --intent X --indices '[3,7,12]' --reason "explanation"` (fragile if indices shifted since last query)
  - Automatically computes embeddings for new examples
  - Automatically logs to changelog
  - Has guardrails: won't remove >30% of intent data or drop below 10 examples
- **`compute_embedding.py`** — Embed a single sentence for hypothesis testing
  - `--text "sentence"` `--compare-intent X` (optional: show similarity to intent centroid)

### Memory Tool
- **`show_changelog.py`** — Read optimization history
  - `--last N` — show last N entries (default: 10)
  - `--intent X` — filter by intent

## Critical Rules

1. **NEVER evaluate on the test set** during optimization. Only use `--split test` for the final evaluation when you believe optimization is complete.
2. **Always provide a `--reason`** when modifying training data. Be specific about why.
3. **Start every session** by running `show_changelog.py` and `query_data.py --inventory` to understand current state.
4. **Do not modify `config.yaml`** unless explicitly asked by the user.
5. **Respect guardrails** — if a tool refuses an action, do not try to work around it.
6. **Think before acting** — when you see confused intent pairs, read the examples AND the descriptions before deciding on an action. Sometimes the descriptions need updating (flag this to the user), not the training data.
7. **Stop after diminishing returns** — if F1 improvement is <0.5% between iterations, report results and stop.
8. **When in doubt, ask the user** — especially for intent overlap that might require structural changes (merging/splitting intents).
9. **When Azure auth code is added to `tools/utils.py`**, update the README.md setup section to reflect the actual authentication mechanism and remove the TODO comment.

## Changelog Format

The changelog at `notes/changelog.md` is your persistent memory. `modify_train_data.py` appends entries automatically, but you should also write summary reflections after each evaluation round explaining what worked, what didn't, and what you plan to try next. Use this format for reflections:

```
## Reflection — Run XXX (YYYY-MM-DD HH:MM)
- **Overall F1**: X.XX (delta: +/-X.XX from previous)
- **Key changes**: what you did this round
- **What worked**: which changes improved performance
- **What didn't**: which changes had no effect or hurt
- **Next steps**: what you plan to try next
```
