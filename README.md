# Intent Optimizer

An AI-agent-assisted optimization toolkit for LogisticRegression intent classifiers using OpenAI embeddings.

## Setup

```bash
pip install -r requirements.txt
```

> **Note on authentication**: This project uses Azure AD (`az login`) to retrieve OpenAI credentials from a corporate keyvault. The authentication logic lives in `tools/utils.py` — see the `compute_embeddings` function. <!-- TODO: Update this section once the Azure auth code is added to utils.py -->

## Data Structure

Place your intent data in `data/train/`:
```
data/train/
├── intent_name_1/
│   ├── examples.csv       # single column: text
│   └── description.md     # intent description for LLM stage
├── intent_name_2/
│   ├── examples.csv
│   └── description.md
```

Place validation data in `data/validation/validation.csv` with columns: `text`, `true_intent`.
Place test data in `data/test/test.csv` with columns: `text`, `true_intent`.

## Usage with GitHub Copilot Agent

1. Open this project in VS Code
2. Invoke the Copilot prompt: `.github/prompts/optimize-intents.prompt.md`
3. The agent will follow the optimization workflow automatically

## Manual Tool Usage

All tools are in `tools/` and run from the project root:

```bash
# Train and evaluate
python tools/train_and_evaluate.py --split val --run-name "baseline"

# Find confused pairs
python tools/kfold_confusion.py --top-n 10

# Explore data
python tools/query_data.py --inventory
python tools/query_data.py --intent betaalrekening_opzeggen --show-description

# Analyze confusion
python tools/analyze_errors.py --intent-a X --intent-b Y

# Predict single sentence
python tools/predict.py --text "ik wil mijn rekening opzeggen"

# Embed and compare
python tools/compute_embedding.py --text "test sentence" --compare-intent X

# Modify training data
python tools/modify_train_data.py --action add --intent X --examples '["new example"]' --reason "why"
python tools/modify_train_data.py --action remove --intent X --texts '["exact text to remove"]' --reason "why"
python tools/modify_train_data.py --action remove --intent X --indices '[3,7]' --reason "why"  # fragile, prefer --texts

# View changelog
python tools/show_changelog.py --last 5
```
# not-animated-octo-palm-tree
