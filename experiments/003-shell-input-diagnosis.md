# Experiment 003 — Shell input diagnosis

Date: 2026-09-17

## Symptom

The user entered a question containing `$250`, but both model runs answered as though the requested amount were within the $100 agent limit. The reported provider was `openai`, proving the deterministic dollar-amount rule had not matched.

## Root cause

The question was wrapped in double quotes in a shell command. In `zsh`, dollar-prefixed text inside double quotes is expanded. `$250` is interpreted as positional parameter `$2` followed by `50`; with no `$2` value, Python receives `50`.

The rule engine correctly declined to run because the received question no longer contained a dollar amount. The model then correctly applied the policy to the mutated value of 50.

## Fix

- Use single quotes for terminal arguments containing dollar signs.
- Echo the exact question received by the CLI in every answer.
- Keep the deterministic rule engine for business decisions.

## Verified command

```bash
uv run evidence-rag ask \
  --provider baseline \
  --rules data/policy_rules.json \
  --docs data/handbook.md \
  'Can a support agent approve a $250 refund?'
```

## Verified result

- Decision: `no`
- Provider: `rule-engine`
- Total tokens: `0`
- Citation: `handbook#2`

## Lesson

Observability begins at the system boundary. Logging model output and token usage is insufficient if the application does not also record the exact input it received.
