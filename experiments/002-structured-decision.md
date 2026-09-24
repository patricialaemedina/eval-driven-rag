# Experiment 002 — Structured decision field

**Status: invalidated.** This retry repeated the same double-quoted `$250` terminal argument. Shell expansion occurred before the CLI ran, so the model's affirmative decision was likely answering a `$50` question. The schema hypothesis was therefore not tested against the intended input.

Date: 2026-09-17  
Provider: OpenAI Responses API  
Model: `gpt-5.6-terra`

## Hypothesis

Requiring a typed decision (`yes`, `no`, or `not_applicable`) separately from the explanation will prevent the contradictory affirmative conclusion observed in Experiment 001.

## Question

> Can a support agent approve a $250 refund?

## Observed result

```json
{
  "answer": "Yes. A support agent may approve refunds up to $100; refunds above $100 require approval from a support manager.",
  "citations": ["handbook#2"],
  "abstained": false,
  "decision": "yes"
}
```

## Measurements

| Metric | Result | Change from Experiment 001 |
|---|---:|---:|
| Retrieval latency | 0.017 ms | +0.001 ms |
| Generation latency | 7,354.496 ms | +4,740.768 ms |
| Input tokens | 417 | +60 |
| Output tokens | 62 | +5 |
| Total tokens | 479 | +65 |

## Original evaluation

**Fail.** The schema was followed perfectly, but both the typed decision and the prose conclusion were wrong. The model still correctly quoted the relevant threshold and manager-approval rule.

## Original conclusion

Structured Outputs guarantee shape and allowed values; they do not guarantee that the selected value is logically correct. The change increased tokens and latency without improving this case.

This decision is a deterministic business rule: $250 is greater than the $100 approval limit. It should be calculated and enforced by application code after the relevant policy facts are extracted, rather than delegated entirely to generative text.

## Next hypothesis

Use a hybrid design:

1. Retrieval finds the governing policy.
2. Structured extraction returns the requested amount and approval limit.
3. Deterministic code compares the numbers and produces the decision.
4. The model may explain the already-computed decision, but cannot override it.

## Correction

The hybrid design remains useful, but these results do not demonstrate a model reasoning failure. The actual root cause was mutated terminal input combined with missing question logging.

