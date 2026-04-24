# Restructure-Intents Skill — Tool Design

Review doc for the tool changes the skill needs. Signatures and behavior, not
implementation. Delete this file after implementation is merged.

## Summary of file changes

| File | Change |
|---|---|
| `tools/kfold_confusion.py` | **modify** — also emit per-example misclassifications CSV |
| `tools/analyze_errors.py` | **modify** — switch data source from val run to k-fold OOF |
| `tools/move_examples.py` | **new** |
| `tools/merge_intents.py` | **new** |
| `tools/split_intent.py` | **new** |
| `tools/find_outliers.py` | **new** |
| `tools/utils.py` | **modify** — add `git_commit_restructure(msg)` helper |
| `.github/prompts/restructure-intents.prompt.md` | **new** (done) |
| `notes/restructure_log.md` | **new** (done, empty stub) |
| `notes/intent_merge_history.md` | **new** (done, gitignored) |
| `notes/intent_merge_history.template.md` | **new** (done, tracked) |

---

## Existing tools — reuse unchanged

- `tools/query_data.py`
- `tools/modify_train_data.py`
- `tools/show_changelog.py`

---

## Existing tools — modifications

### `tools/kfold_confusion.py`

**Current**: emits `results/current/confusion_pairs.json` (pair counts).

**Add**: also emit `results/current/kfold_misclassifications.csv` with columns:
- `text`
- `true_intent`
- `predicted_intent`
- `confidence` (probability the model assigned to `predicted_intent`)
- `fold` (which CV fold this example was in the held-out set)

**Rationale**: `analyze_errors.py` needs per-example data. Currently only
available from val-split runs via `train_and_evaluate.py`. The restructure
skill can't use val, so we produce the same data from k-fold OOF.

**Also print** to stdout: overall k-fold accuracy (so the agent has a single
headline number — much clearer than per-pair counts for tracking session
progress).

### `tools/analyze_errors.py`

**Current**: reads `results/current/misclassifications.csv` (produced by
`train_and_evaluate.py` against val).

**Change**: read `results/current/kfold_misclassifications.csv` instead. If
absent, print "run `kfold_confusion.py` first" and exit.

No CLI changes; the tool's output format stays the same.

---

## New tools

### `tools/move_examples.py`

**Purpose**: relabel examples from one intent to another. Most common
structural operation.

**CLI**:
```
python tools/move_examples.py --from-intent A --to-intent B \
  --texts '["text1", "text2", ...]' --reason "..."

python tools/move_examples.py --from-intent A --to-intent B --all --reason "..."
```

Either `--texts` (specific list) or `--all` (move every example from A).

**Behavior**:
1. Validates both intent folders exist and have `examples.csv`. If `--to`
   doesn't exist, creates the folder + empty examples.csv.
2. Looks up each text in A; warns about any not found and continues with the rest.
3. Removes from A's CSV, appends to B's CSV.
4. If A ends up with 0 examples AND has no `description.md`, deletes the folder.
   (If A has a description.md but ends up empty, refuse — the SME's taxonomy
   says this intent should exist.)
5. Embeddings cache unchanged (same text, just different label — the cache is
   keyed by text).
6. Appends a changelog entry.
7. Git-commits the change (see auto-commit below).

**Guardrails**:
- Refuse if `--from-intent` has `description.md` AND the move would leave it
  with fewer than `min_examples_per_intent` (config). Described intents should
  keep their scope intact.
- Refuse if `--texts` is empty or has no matches found in source.

### `tools/merge_intents.py`

**Purpose**: combine two intents into one. Source is deleted.

**CLI**:
```
python tools/merge_intents.py --source X --target Y --reason "..."
```

**Behavior**:
1. Validates both folders exist. Checks description.md presence on both.
2. Moves all examples from X's examples.csv into Y's examples.csv.
3. Deletes X's folder (including any stray files if non-described — no
   description.md to preserve).
4. Appends changelog entry.
5. Git-commits.

**Guardrails**:
- **Refuse if `--source` has `description.md`**. Described intents are SME
  final taxonomy and cannot be eliminated. (If target is described, that's
  fine — moving non-described into described is the preferred direction.)
- Refuse if source and target are the same intent.
- Refuse if either folder lacks examples.csv.

### `tools/split_intent.py`

**Purpose**: divide one intent into multiple sub-intents based on an explicit
assignment map. Agent decides the names and the per-example assignment.

**CLI**:
```
python tools/split_intent.py --intent X \
  --assignments '{"X_refund": ["text1","text2"], "X_status": ["text3","text4","text5"]}' \
  --reason "..."
```

**Behavior**:
1. Validates X exists.
2. Validates every example in X's examples.csv appears in exactly one
   assignment bucket. No orphans, no duplicates. (If agent wants to drop
   examples, it should use `modify_train_data.py --action remove` first.)
3. Creates folder for each new intent, writes examples.csv, copies over each
   assigned text.
4. Deletes the original X folder.
5. Appends changelog entry.
6. Git-commits.

**Guardrails**:
- **Refuse if X has `description.md`**. Splitting a described intent is a
  world-shattering change — the prompt tells the agent to stop and surface
  to the user rather than attempt this. Tool-level refusal is a backstop.
- Refuse if any new intent name collides with an existing intent folder.
- Refuse if `--assignments` doesn't cover all examples in X.

### `tools/find_outliers.py`

**Purpose**: diagnostic — find examples far from their own intent's centroid,
which often indicates mislabeling.

**CLI**:
```
python tools/find_outliers.py --intent X --top-n 10
python tools/find_outliers.py --all --top-n 5   # top 5 outliers per intent
```

**Behavior**:
1. Loads embeddings for the intent(s).
2. Computes centroid, cosine similarity of each example to centroid.
3. Prints the N examples with lowest similarity, with the score.
4. No side effects — purely diagnostic.

---

## Auto-commit mechanism

Each destructive tool commits its own change so revert is cheap. Helper in
`tools/utils.py`:

```python
def git_commit_restructure(message: str, paths: list[str]):
    """Stage the given paths and commit with a 'restructure:' prefix.
    Silently skips if not in a git repo or if there are no staged changes."""
```

Commit messages are prefixed `restructure:` so they're easy to grep/revert:
- `restructure: move 40 examples from A to B`
- `restructure: merge A into B`
- `restructure: split X into X_refund, X_status`
- `restructure: add 8 paraphrased examples to A`

The agent doesn't think about commits; they happen as side effects.

**Alternative we discussed** (rejected): agent commits manually. Rejected
because it adds a per-op cognitive step for the agent and a forgotten commit
means no rollback. Auto-commit is cheap insurance.

---

## Questions I still want you to confirm

1. **Auto-commit inside each tool, or agent commits manually?** Recommending
   auto-commit. Confirm?
2. **`find_outliers.py` in scope, or defer?** Useful but not critical for v1.
3. **`move_examples.py --all` flag** (move every example from the source) —
   handy for full SME-consolidation moves, or overkill?
4. **Any tool names you'd rename** before I implement?

If all four land as "good", I'll start implementation on this branch.
