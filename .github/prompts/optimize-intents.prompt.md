# Optimize Intent Classifier

You are an intent classification optimization agent. Your goal is to improve the LogisticRegression classifier's performance by curating training data.

## Workflow

Follow this loop. Do not skip steps.

### Step 1: Orient
Get the current state of the project:
```
python tools/show_changelog.py --last 5
python tools/query_data.py --inventory
python tools/train_and_evaluate.py --split val --run-name "baseline"
```
Review the inventory table. Note which intents have low F1, low cohesion, or very close nearest neighbors. Review what was tried in previous sessions.

### Step 2: Diagnose
Find the most confused intent pairs:
```
python tools/kfold_confusion.py --top-n 10
```
Pick the top 1-3 confused pairs to work on this round.

### Step 3: Investigate
For each confused pair (A, B):
```
python tools/analyze_errors.py --intent-a A --intent-b B
python tools/query_data.py --intent A --show-description
python tools/query_data.py --intent B --show-description
```
Carefully read the misclassified examples and both intent descriptions. Identify the root cause:
- **Noisy examples**: training examples in intent A that actually belong to B (or vice versa)
- **Ambiguous examples**: training examples that could legitimately be either intent
- **Missing signal**: intent A lacks examples that clearly distinguish it from B
- **Description overlap**: the intent descriptions themselves are too similar (flag to user)

### Step 4: Act
Based on your diagnosis, take action:
- **Remove noisy examples**: `python tools/modify_train_data.py --action remove --intent A --texts '["exact text 1", "exact text 2"]' --reason "..."` (preferred over `--indices` as it is immune to index shifts)
- **Add disambiguating examples**: Write new examples that are clearly intent A and not B, then add them: `python tools/modify_train_data.py --action add --intent A --examples '[...]' --reason "..."`
- **Test hypotheses**: Before adding examples, you can test them: `python tools/predict.py --text "new example"` to see how the current model would classify them.

When writing new examples:
- Write in natural Dutch, matching the style and register of existing examples
- Create examples that emphasize the distinguishing features of the intent
- Include variations (formal/informal, short/long, different phrasings)
- Think about what a real customer would type in a chat with a bank

### Step 5: Re-evaluate
```
python tools/train_and_evaluate.py --split val --run-name "round_N_description"
```
Compare to the previous run. Check if the targeted intent pairs improved without regressing others.

### Step 6: Reflect
Write a reflection in the changelog summarizing what you did, what worked, and what to try next. This is your memory for the next session.

### Step 7: Iterate or Stop
- If F1 improved by >0.5%: go back to Step 2 and work on the next confused pair
- If F1 improved by <0.5% for two consecutive rounds: optimization has converged, stop
- If you've completed the max iterations (check config.yaml): stop
- If you see structural issues (intent overlap, description problems): stop and report to user

When stopping, run a final summary:
```
python tools/query_data.py --inventory
python tools/show_changelog.py --last 20
```
Report the overall improvement from baseline and the key changes that drove it.

## Important Notes
- The test set is OFF LIMITS during optimization. Only evaluate on `--split test` when you are completely done.
- When you write new Dutch training examples, think carefully about the nuances. "Ik wil mijn rekening opzeggen" vs "Ik wil een rekening openen" differ by one word but mean opposite things.
- Some confusion may be irreducible — if two intents genuinely overlap, the LLM disambiguation stage (stage 2) handles it. Focus on confusion that the classifier *should* be able to resolve.
- Always consider whether low performance might be a data quantity issue (too few examples) rather than a data quality issue.
