# Restructure Log

Persistent memory for the restructure-intents skill. Each session appends one
entry here describing structural changes, k-fold signal before/after, and what
to tackle next.

Entry format:
```markdown
## Session YYYY-MM-DD HH:MM
**K-fold accuracy**: before X.XXX → after Y.YYY (Δ +/-Z.ZZZ)
**Structural changes**:
- [merge/split/bulk-move descriptions]
**Example-level**:
- [add/remove summary — count, not every text]
**Flagged for human review**: none | [short description]
**Next session**: [what to tackle next]
```

---
