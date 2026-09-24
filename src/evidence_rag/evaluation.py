import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import Answer, EvalCase, sum_token_counts
from .service import GroundedAssistant


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    passed: bool
    required_term_coverage: float
    forbidden_term_violations: tuple[str, ...]
    citation_recall: float
    abstention_correct: bool
    decision_correct: bool
    clarification_correct: bool
    answer: Answer


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            cases.append(
                EvalCase(
                    question=item["question"],
                    required_terms=tuple(item.get("required_terms", [])),
                    expected_citations=tuple(item.get("expected_citations", [])),
                    forbidden_terms=tuple(item.get("forbidden_terms", [])),
                    should_abstain=bool(item.get("should_abstain", False)),
                    expected_decision=item.get("expected_decision"),
                    should_clarify=bool(item.get("should_clarify", False)),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid eval case on line {line_number}: {error}") from error
    return cases


def evaluate_case(assistant: GroundedAssistant, case: EvalCase) -> CaseResult:
    answer = assistant.answer(case.question)
    answer_lower = answer.text.lower()
    term_hits = sum(term.lower() in answer_lower for term in case.required_terms)
    term_coverage = term_hits / len(case.required_terms) if case.required_terms else 1.0
    forbidden_violations = tuple(
        term for term in case.forbidden_terms if term.lower() in answer_lower
    )
    expected = set(case.expected_citations)
    citation_recall = (
        len(expected.intersection(answer.citations)) / len(expected) if expected else 1.0
    )
    abstention_correct = answer.abstained == case.should_abstain
    decision_correct = case.expected_decision is None or answer.decision == case.expected_decision
    clarification_correct = answer.needs_clarification == case.should_clarify
    passed = (
        term_coverage == 1.0
        and not forbidden_violations
        and citation_recall == 1.0
        and abstention_correct
        and decision_correct
        and clarification_correct
    )
    return CaseResult(
        question=case.question,
        passed=passed,
        required_term_coverage=term_coverage,
        forbidden_term_violations=forbidden_violations,
        citation_recall=citation_recall,
        abstention_correct=abstention_correct,
        decision_correct=decision_correct,
        answer=answer,
        clarification_correct=clarification_correct,
    )


def result_as_dict(result: CaseResult) -> dict[str, object]:
    return asdict(result)


def summarize_results(results: list[CaseResult]) -> dict[str, object]:
    count = len(results)
    if not count:
        return {"passed": 0, "total": 0}
    metrics = [result.answer.metrics for result in results]
    return {
        "passed": sum(result.passed for result in results),
        "total": count,
        "providers": sorted({metric.provider for metric in metrics}),
        "models": sorted({metric.model for metric in metrics if metric.model}),
        "average_retrieval_ms": round(sum(metric.retrieval_ms for metric in metrics) / count, 3),
        "average_generation_ms": round(sum(metric.generation_ms for metric in metrics) / count, 3),
        "input_tokens": sum_token_counts(*(m.input_tokens for m in metrics)),
        "output_tokens": sum_token_counts(*(m.output_tokens for m in metrics)),
        "total_tokens": sum_token_counts(*(m.total_tokens for m in metrics)),
    }
