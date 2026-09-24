import re
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter
from typing import Protocol

from .models import (
    Answer,
    AnswerMetrics,
    ApprovalRequest,
    Chunk,
    ExtractionResult,
    SearchResult,
    sum_token_counts,
)
from .retrieval import BM25Index, tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+")


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: Sequence[SearchResult]) -> Answer: ...


class RequestExtractor(Protocol):
    def extract(self, question: str) -> ExtractionResult: ...


class DecisionRuleEngine(Protocol):
    def answer(self, question: str, evidence: list[SearchResult]) -> Answer | None: ...

    def answer_request(
        self,
        request: ApprovalRequest,
        evidence: list[SearchResult],
    ) -> Answer | None: ...


class ExtractiveAnswerGenerator:
    """The deterministic baseline that later generators must outperform."""

    def generate(self, question: str, evidence: Sequence[SearchResult]) -> Answer:
        query_terms = set(tokenize(question))
        candidates: list[tuple[int, str, str]] = []
        for result in evidence:
            for sentence in SENTENCE.split(result.chunk.text):
                cleaned = sentence.strip()
                overlap = len(query_terms.intersection(tokenize(cleaned)))
                if cleaned and overlap:
                    candidates.append((overlap, cleaned, result.chunk.id))
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = candidates[:2]
        if not chosen:
            return Answer(
                text="I found related material but not enough evidence to answer confidently.",
                citations=(),
                abstained=True,
            )
        citations = tuple(dict.fromkeys(item[2] for item in chosen))
        text = " ".join(item[1] for item in chosen)
        return Answer(text=text, citations=citations)


class GroundedAssistant:
    """Retrieve evidence, then delegate answer construction to a replaceable generator."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        generator: AnswerGenerator | None = None,
        rule_engine: DecisionRuleEngine | None = None,
        extractor: RequestExtractor | None = None,
    ) -> None:
        if extractor is not None and rule_engine is None:
            raise ValueError("An extractor requires a rule engine")

        self.chunks = list(chunks)
        self.index = BM25Index(self.chunks)
        self.generator = generator or ExtractiveAnswerGenerator()
        self.rule_engine = rule_engine
        self.extractor = extractor

    def answer(self, question: str, *, limit: int = 3) -> Answer:
        retrieval_started = perf_counter()
        results = self.index.search(question, limit=limit)
        retrieval_ms = (perf_counter() - retrieval_started) * 1_000
        if not results:
            return Answer(
                text="I cannot answer that from the provided handbook.",
                citations=(),
                abstained=True,
                metrics=AnswerMetrics(retrieval_ms=round(retrieval_ms, 3), retrieved_chunks=0),
            )

        extraction = self.extractor.extract(question) if self.extractor is not None else None

        generation_started = perf_counter()
        if extraction is not None:
            request = extraction.request

            if request.action == "approve_refund":
                answer = self.rule_engine.answer_request(request, results)

                if answer is None:
                    answer = Answer(
                        text=(
                            "I could not make a supported approval decision "
                            "from the supplied details and evidence."
                        ),
                        citations=(),
                        abstained=True,
                    )
            else:
                answer = self.generator.generate(question, results)
        else:
            answer = self.rule_engine.answer(question, results) if self.rule_engine else None
            if answer is None:
                answer = self.generator.generate(question, results)
        generation_ms = (perf_counter() - generation_started) * 1_000
        extraction_usage = extraction.metrics if extraction else None
        metrics = replace(
            answer.metrics,
            retrieval_ms=round(retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            retrieved_chunks=len(results),
            extraction=extraction.metrics if extraction is not None else None,
            input_tokens=sum_token_counts(
                answer.metrics.input_tokens,
                extraction_usage.input_tokens if extraction_usage else 0,
            ),
            output_tokens=sum_token_counts(
                answer.metrics.output_tokens,
                extraction_usage.output_tokens if extraction_usage else 0,
            ),
            total_tokens=sum_token_counts(
                answer.metrics.total_tokens,
                extraction_usage.total_tokens if extraction_usage else 0,
            ),
        )
        extracted_request = None
        if extraction is not None:
            request = extraction.request
            extracted_request = {
                "role": request.role,
                "action": request.action,
                "amount": (str(request.amount) if request.amount is not None else None),
                "intent": request.intent,
            }

        return replace(
            answer,
            metrics=metrics,
            extracted_request=extracted_request,
        )
