# Experiment 004 — Choosing an extractor using failure traces

Recorded: 2026-09-24

## Decision

Use `gpt-5.6-terra` as the working extraction model for this learning project. In a controlled comparison on one ambiguous-role question, it preserved the missing role in 5/5 calls; `gpt-4.1-nano` guessed a support-agent role in 5/5 calls. The subsequent user-reported application evaluation passed 13/13 cases.

This is a provisional project decision, not a general model ranking or a claim of production reliability. No dollar-cost comparison was performed.

## Application and requirement

The application retrieves Markdown handbook sections with BM25. An optional LLM extracts `role`, `action`, `amount`, and `intent`; Python applies policy rules only when their supporting evidence is retrieved. Other questions use the configured answer generator. The comparison used extraction independently from retrieval and answer generation.

For the question:

> Can an agent approve a refund?

the expected extraction is:

```json
{
  "role": null,
  "action": "approve_refund",
  "amount": null,
  "intent": "check_authority"
}
```

The word “agent” does not specify a support agent or travel agent. The application should ask for the agent type before applying a role-specific rule. Valid JSON containing an invented role fails this requirement.

## How the failure became diagnosable

An earlier 13-case evaluation intermittently failed on “Can a support agent approve a $100.01 refund?” A successful retry did not explain that failure. Adding `extracted_request` to each answer made the intermediate values visible in the evaluation output.

A five-call repeat then captured one failure: the amount was correctly extracted as `100.01`, but `intent` was `null`. The rule engine required `check_authority` and declined to decide. That sample passed 4/5. It explains the captured failure, but cannot establish the cause of the earlier run whose extraction was not logged.

After clarifying the intent instructions, the same boundary question passed 5/5 repeats. The broader extraction suite still caught a different error: nano inferred `support_agent` from “an agent.” More explicit role instructions did not eliminate that error in the submitted runs.

We retained `role: null` as the expected result rather than changing the test to accept the model's guess. A single six-case comparison then passed 5/6 for nano and 6/6 for Terra, motivating the controlled repeated comparison below.

## Controlled comparison

- Same question and expected extraction for every call.
- Same extraction schema, prompt, application code, and evaluator for both models; only the explicitly supplied model changed.
- Five sequential calls per model, alternating nano then Terra in each round; order was not randomized.
- Every attempt counted, including failures. No retry-until-success selection.
- The evaluator required equality of all four fields, including `None` for missing values.
- Each call recorded the extracted values, per-field checks, duration, and API-reported input/output/total tokens.
- The prompt explicitly prohibited inferring a role from the refund topic. Thus this measured adherence to a stated requirement, not performance on a hidden test.

### Observed results

| Model | Passed | Role returned in every call | Mean extraction duration | Input tokens, five calls | Output tokens, five calls | Total tokens, five calls |
|---|---:|---|---:|---:|---:|---:|
| `gpt-4.1-nano` | 0/5 | `support_agent` | 1,562.031 ms | 2,165 | 115 | 2,280 |
| `gpt-5.6-terra` | 5/5 | `null` | 1,634.360 ms | 2,155 | 150 | 2,305 |

Action, amount, and intent matched the expected values in all ten calls. Role was the only failing field.

| Round | Nano extraction duration (ms) | Terra extraction duration (ms) |
|---|---:|---:|
| 1 | 2,266.247 | 1,642.433 |
| 2 | 1,225.289 | 1,488.161 |
| 3 | 1,131.486 | 1,793.461 |
| 4 | 1,943.674 | 1,922.028 |
| 5 | 1,243.461 | 1,325.718 |

Terra used 25 more tokens over five calls: 461 rather than 456 per call, approximately 1.1% more. Its observed mean duration was about 72 ms higher. Neither result establishes a general latency or cost advantage. Token prices can differ by model and by input/output category; token counts alone cannot determine dollars spent.

## Application regression check

After the comparison, the user changed `OPENAI_MODEL` to `gpt-5.6-terra` and submitted this full-suite summary:

| Measurement | User-reported value |
|---|---:|
| Cases passed | 13/13 |
| Input tokens | 5,205 |
| Output tokens | 358 |
| Total tokens | 5,563 |
| Mean retrieval duration | 0.088 ms |
| Mean generation duration | 0.153 ms |

The cases cover handbook answers, an unsupported question, approver identification, role clarification, amount clarification, travel-agent prohibition, and the inclusive $100 boundary versus $100.01.

The summary alone does **not** identify the extraction model: its `models` list contains the answer-producing rule labels. The reported configuration change attributes this run to Terra; independent confirmation requires `answer.metrics.extraction.model` from the individual results. These documentation updates did not rerun paid API evaluations.

Generation duration excludes extraction. The 0.153 ms figure is therefore not end-to-end response latency. Top-level token totals include extraction and answer-generation usage. A rule-engine answer can consume LLM tokens during extraction even though its final decision is deterministic.

## Reproduction

From the project directory, use the existing `OpenAIRequestExtractor(model=...)`, `ApprovalRequest`, and `evaluate_extraction` interfaces. For each of five rounds, call each model with the exact question above and compare `result.request` against the expected fields. Preserve all results and `result.metrics`; aggregate pass counts, durations, and token usage separately per model. Do not alter the prompt or expectations between model calls.

Relevant files:

- [Extractor and prompt](../src/evidence_rag/openai_extractor.py)
- [Extraction scoring](../src/evidence_rag/extraction_evaluation.py)
- [Extraction dataset](../data/extraction_eval.jsonl)
- [Policy decisions](../src/evidence_rag/rules.py)
- [Application dataset](../data/eval.jsonl)

For the application regression check, make the model explicit rather than relying on shell configuration:

```bash
uv run --env-file .env evidence-rag eval \
  --provider baseline \
  --extractor openai \
  --model gpt-5.6-terra \
  --rules data/policy_rules.json \
  --docs data/handbook.md \
  --dataset data/eval.jsonl
```

The API key stays in the local `.env` file; it is not part of the experiment record. Results here were transcribed from user-supplied terminal outputs in the project conversation. Raw run files, exact model snapshots, and a frozen prompt/code revision were not archived with this experiment, so the current code can reproduce the procedure but not guarantee identical historical results.

## Limitations and next checks

- Five repetitions of one question are not five diverse examples. The observed 0/5 and 5/5 rates are not population accuracy estimates.
- These six extraction cases and thirteen application cases were used during development. Prompt examples overlap the tests; they are not an untouched held-out benchmark.
- The final full-suite pass is one run, not a demonstrated reliability guarantee. The earlier intermittent intent failure should remain a regression case.
- Timing includes API/network variability. Small, sequential, fixed-order samples do not establish latency percentiles or deployment performance.
- Required-term coverage and citation recall are limited checks, not semantic correctness proofs. Empty expected citations currently give recall 1.0 and do not reject extra citations.
- Unknown and unsupported roles both map to `null`; more diverse roles, paraphrases, negations, multiple amounts, and unseen questions need testing.
- The application asks clarification questions but does not yet retain conversation state to apply the user's next reply.
- Snapshot prompts, datasets, code versions, and raw traces in future comparisons. Measure extraction latency separately and use verified prices if estimating dollar costs.

An unrelated issue was confirmed during the original documentation task: in `cli.py`, `extractor = None` was indented inside the OpenAI answer-provider branch, causing baseline CLI commands with the default `--extractor none` to raise `UnboundLocalError`. The comparison and reported full-suite command explicitly enabled the extractor and did not take that failing setup path.

Follow-up, 2026-09-24: the pre-publication cleanup moved initialization outside that branch and added CLI regression coverage. The offline suite now contains 22 passing tests. The structured-pipeline tests use fake extraction fixtures; they do not replace the live model measurements above. Legacy baseline evaluation expectations were updated to explicitly retain their known failures, not to weaken the clarification requirements in the dataset.

## Portfolio takeaway

The engineering contribution is the evaluation process: separate interpretation from policy decisions, preserve missing information, log intermediate values, reproduce a failure, compare models under the same conditions, and recheck application behavior. The evidence supports a scoped model choice while leaving its limits visible.
