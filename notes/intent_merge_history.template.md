# SME Intent Consolidation History — TEMPLATE

> **This file is a template, not the real document.**
> The real file lives at `notes/intent_merge_history.md` and is **gitignored**
> because it contains proprietary taxonomy information. Populate it from the
> SME's source documents (e.g. the topic-description docx files) and keep it
> out of version control.
>
> The restructure-intents skill reads the real file at session start to
> understand SME-authoritative decisions about how old intents were merged
> into final ones. Without it, the skill will fall back to k-fold + embedding
> signals only.

## Purpose of the real file

Captures SME-authoritative decisions about:
1. Which intents are **final (described)** — SME has consolidated these, agent
   should not merge or split them
2. Which old intents have been **deprecated** and where their training examples
   should migrate
3. For each consolidation, whether it was **full** (entire old intent absorbed)
   or **partial** (only matching examples; source keeps remainder)
4. Any SME notes, flags, or open questions

## Conventions (copy these into the real file)

- **Consolidates**: list of old intents merged into this one
- **full**: entire old intent absorbed (no unique remainder)
- **partial**: only specific examples matching a criterion; source keeps other content
- **Deprecated**: intent to be removed; examples migrated per the mapping
- **Candidate for removal**: SME flagged low/misaligned traffic — consider whether to keep at all

## Required structure

```markdown
# SME Intent Consolidation History

**Read this FIRST.** [short intro, scope of which groups are covered]

## Conventions
[copy from above]

---

## Deprecated / to be removed entirely

- `old_intent_name` — short reason (e.g. "no own content", "low traffic (X in 3mo)"); where its examples went
- ...

---

## <Topic group name> (final intents)

### <final_intent_name>
**Purpose**: one-line description of what this intent handles (what belongs, what doesn't).

**Consolidates**: `old_intent_a`, `old_intent_b` (partial), `old_intent_c` (partial)

**Source mapping**:
- `old_intent_a` → full (reason — e.g. "topic has no own content, all examples migrate here")
- `old_intent_b` → partial (criterion — e.g. "only examples matching topic X; source keeps Y content")
- `old_intent_c` → partial (criterion)

**Examples moved out of this topic** (optional — if any examples were migrated elsewhere):
- `example sentence text` → to `target_intent`

**Sample migrated-in examples** (optional — for validation):
- `example text 1`
- `example text 2`

**SME notes** (optional):
- Any flags, open questions, or cautions

---

### <next_final_intent_name>
[same structure]
```

## Authoring tips

- **Keep entries skimmable.** The agent scans this at orient-time — every entry should be readable at a glance.
- **Write purpose statements in the SME's native language** (Dutch in our case) if the intent descriptions are in Dutch. Consistency > translation.
- **Only include sample migrated examples** if they help validate the mapping; don't exhaustively list every example.
- **Flag SME open questions explicitly** at the bottom of the file so the agent and human both see them — don't let them get buried in individual entries.
- **Update when SME adds new consolidations.** Old entries stay; new ones append.

## What the skill does with this file

1. Reads it at session start (orient step)
2. For each deprecated intent listed: checks if `data/train/<old_intent>/` still exists with examples. If so, moves them to the target per the mapping (Phase 0: SME alignment).
3. For each final (described) intent: treats its name and scope as authoritative. Does not merge or split described intents. If the agent thinks a described intent should change structurally, it stops and flags to the human.
4. Uses "Sample migrated-in examples" to spot-check that training data matches the SME mapping.

## Example (illustrative, not real)

```markdown
### payment_cancel
**Purpose**: customer wants to cancel a pending or scheduled payment.

**Consolidates**: `payment_stop` (full), `transfer_cancel` (partial)

**Source mapping**:
- `payment_stop` → full (deprecated; all examples migrate here)
- `transfer_cancel` → partial (only cancel requests; transfer_cancel keeps reverse/refund content)

**Sample migrated-in examples**:
- `Ik wil een betaling stoppen`
- `Stop deze overboeking`
```
