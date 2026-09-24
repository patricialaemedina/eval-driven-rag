# Experiment 001 — First model-backed answer

**Status: invalidated.** The terminal command used double quotes around `$250`. The shell expanded `$2` before starting Python, so the application likely received `50` rather than `$250`. Because the exact received question was not logged in this version, this run cannot support a conclusion about model reasoning.

Date: 2026-09-17  
Provider: OpenAI Responses API  
Model: `gpt-5.6-terra`

## Question

> Can a support agent approve a $250 refund?

## Retrieved evidence

`handbook#2` states that support agents may approve refunds up to $100 and that refunds above $100 require approval from a support manager.

## Observed answer

> Yes. Support agents may approve refunds up to $100; refunds above $100 require approval from a support manager.

Citation: `handbook#2`

## Measurements

| Metric | Result |
|---|---:|
| Retrieval latency | 0.016 ms |
| Generation latency | 2,613.728 ms |
| Input tokens | 357 |
| Output tokens | 57 |
| Total tokens | 414 |

## Original evaluation

**Fail.** The supporting policy and citation are correct, but the leading “Yes” contradicts the policy. A support agent cannot independently approve a $250 refund; manager approval is required.

This is an important failure mode: keyword-based checks could see “support manager” and mark the answer correct even though its direct conclusion is wrong.

## Action taken

- Added this question to the regression dataset.
- Added `forbidden_terms` support to catch a wrong affirmative conclusion.
- Added a unit test reproducing the contradiction.

## Next hypothesis

Make the response schema require a direct decision (`yes`, `no`, or `not_applicable`) separately from the explanation. Evaluate that field structurally instead of relying only on phrases in prose.

Status: implemented in the next iteration.

## Correction

The apparent contradiction was caused by an input-observability and shell-quoting error. Future CLI output includes the exact received question, and examples containing dollar signs use single quotes.
