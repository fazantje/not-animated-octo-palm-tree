# Noisy Review

Multi-intent suspects that the restructure agent flagged and dropped from
training during Phase 0.5.

The drop is usually correct — layer-2 disambiguation handles multi-intent at
inference, not the layer-1 LR classifier, so mislabeled-as-single-intent
multi-intent rows in training only distort layer-1. But if any of these look
like they should have been re-labeled with the new BTS multi-intent category
rather than dropped, flag them to the SME.

The agent appends one section per session, with a table:

- `text` — the dropped utterance
- `dropped from` — the intent it was labeled as
- `nearest other intent` — the second intent the embedding was close to
- `note` — optional one-line remark from the agent

(First real session entry will replace this placeholder.)
