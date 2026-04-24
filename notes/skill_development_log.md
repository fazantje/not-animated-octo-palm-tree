# Skill Development Log

A log for Claude sessions that modify the `restructure-intents` skill itself —
its tools, its prompt, its supporting config. **Runtime agents (the ones
executing the skill on training data) do not read this file.** Its audience is
future Claudes who are asked to extend or change the skill.

Each entry captures: what was changed, *why*, what we considered and rejected,
and anything that would be easy for a future Claude to undo without realising
the context.

Contrast with sibling logs:
- `notes/changelog.md` — automated append-only log of data-modifying tool calls
  (runtime)
- `notes/restructure_log.md` — runtime agent's per-session reflection
- `notes/tool_feedback.md` — runtime agent's friction reports, read by
  skill-dev Claudes between sessions
- `notes/skill_development_log.md` (this file) — skill-dev Claudes' own
  history, for continuity across skill-improvement sessions

## Entry format

```markdown
## YYYY-MM-DD — <short title>
**Branch / PR**: <branch name or PR link>
**What changed**: bullet list of files and the nature of the change
**Why**: the problem this addressed (reference prior tool_feedback entries
where relevant)
**Considered and rejected**: alternatives weighed, with the reasoning
**Things a future Claude should not undo without checking**: load-bearing
decisions that look removable but aren't
```

Keep entries terse. The goal is to prevent re-litigation of settled questions,
not to narrate every keystroke.

---

## 2026-04-24 — Phase 0.5 noise sweep + skill-prefix awareness

**Branch / PR**: `noisy-training-data`

**What changed**:
- `tools/find_noisy.py` (new) — flags long/verbose rows and multi-intent suspects
- `config/skill_prefixes.json` (new) — 2-letter prefix → skill mapping
- `tools/utils.py` — `get_skill`, `load_skill_prefixes`, `compute_intent_centroids`, `cosine_similarity_matrix`
- `tools/kfold_confusion.py` — same-skill vs cross-skill annotation on confused pairs
- `tools/merge_intents.py` — `--confirm-cross-skill` flag on cross-skill merges
- Prompt — Step 1.5 (Phase 0.5 noise sweep) and "Skill prefixes" section added
- `notes/noisy_review.md` (new) — human review list for dropped multi-intent suspects
- `notes/skill_development_log.md` (new; this file) — first entry
- Removed `notes/restructure_skill_design.md` (transient PR #2 design doc)

**Why**:
Legacy CLU training has two noise modes — verbose/scaffolded single-intent
rows ("Beste Anna … alvast bedankt") and rare multi-intent rows. Both
depress k-fold signal. The 2-layer architecture delegates multi-intent
handling to a downstream LLM disambiguation layer, so a mislabeled-as-
single-intent multi-intent row in training only distorts layer-1 — drop
rather than preserve. Verbose rows get trimmed (agent judgment) or dropped.
Skill prefixes were unencoded convention; encoding them enables cross-skill
merge warnings and sharper mislabel signals.

**Considered and rejected**:
- *Dedicated `trim_example.py` tool* — trim = remove+add via existing
  `modify_train_data`, batched per intent. No new tool warranted.
- *Regex / substring trimming* — user didn't trust deterministic Dutch
  rewriting. Agent-LLM trim or drop are the only remediations.
- *Preserving multi-intent rows* for natural low-margin top-2 — wrong for a
  2-layer system where layer-2 owns multi-intent. Hand-splitting one row
  into per-intent rows was rejected for the same reason (would train
  layer-1 to be *confident* on each half).
- *Prioritizing cross-skill over same-skill confusion* — user pushed back;
  equally bad for this classifier. Annotate, don't reorder.
- *Hard-refusing cross-skill merges* — soft warning preserves agent
  autonomy.
- *Adding a `nearest_skill` column to `find_outliers.py`* — deferred;
  `find_noisy.py`'s `top2_intent` covers the immediate need. Revisit via
  `tool_feedback.md` if the agent asks.

**Things a future Claude should not undo without checking**:
- *Drop-multi-intent is load-bearing on the 2-layer architecture.* If
  layer-2 disambiguation is ever removed, preserving bimodal rows in
  layer-1 training may become the right call — reconsider before
  generalizing this rule to a 1-layer setup.
- *Thresholds `--multi-intent-margin 0.05` and `--global-length-floor 25`
  are first-pass guesses* from five real examples. Tune in response to
  real-world flag rates before concluding the design is wrong.
- *`notes/noisy_review.md` is the SME's escape hatch* for dropped
  multi-intent rows. Don't delete it or automate it away — it's how the
  SME catches agent mistakes.
