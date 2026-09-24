from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

Decision = Literal["yes", "no", "not_applicable"]


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class ExtractionMetrics:
    provider: str
    model: str
    duration_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    retrieved_chunks: int = 0
    provider: str = "baseline"
    model: str | None = None
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    extraction: ExtractionMetrics | None = None


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[str, ...]
    abstained: bool = False
    decision: Decision = "not_applicable"
    metrics: AnswerMetrics = field(default_factory=AnswerMetrics)
    needs_clarification: bool = False
    extracted_request: dict[str, str | None] | None = None


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    required_terms: tuple[str, ...]
    expected_citations: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    should_abstain: bool = False
    expected_decision: Decision | None = None
    should_clarify: bool = False


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    role: str | None
    action: str | None
    amount: Decimal | None
    intent: Literal["check_authority", "find_approver"] | None = None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    request: ApprovalRequest
    metrics: ExtractionMetrics


def sum_token_counts(*counts: int | None) -> int | None:
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)
