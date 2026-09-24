import json
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from evidence_rag.chunking import load_markdown
from evidence_rag.evaluation import evaluate_case, load_cases, result_as_dict, summarize_results
from evidence_rag.models import (
    Answer,
    AnswerMetrics,
    ApprovalRequest,
    ExtractionMetrics,
    ExtractionResult,
)
from evidence_rag.rules import PolicyRuleEngine
from evidence_rag.service import GroundedAssistant

ROOT = Path(__file__).parents[1]


class StubExtractor:
    """Fixed, correct extraction fixtures; these do not measure any LLM's accuracy."""

    def __init__(self, request, usage=None):
        self.request = request
        self.usage = usage or ExtractionMetrics("stub", "fixture", 10, 120, 30, 150)

    def extract(self, question):
        return ExtractionResult(self.request, self.usage)


class StructuredPipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = load_markdown(ROOT / "data/handbook.md")
        self.engine = PolicyRuleEngine.from_path(ROOT / "data/policy_rules.json")

    def assistant(self, request, **kwargs):
        return GroundedAssistant(
            self.chunks, rule_engine=self.engine, extractor=StubExtractor(request), **kwargs
        )

    def test_full_dataset_with_correct_extraction_fixtures(self):
        fixtures = {
            "Who approves a $250 refund?": (None, "250", "find_approver"),
            "Can a support agent approve a $250 refund?": (
                "support_agent",
                "250",
                "check_authority",
            ),
            "Is a $250 reimbursement within an agent’s authority?": (
                None,
                "250",
                "check_authority",
            ),
            "Does a travel agent have the authority to approve a $250 reimbursement?": (
                "travel_agent",
                "250",
                "check_authority",
            ),
            "May a travel agent authorize a $50 reimbursement?": (
                "travel_agent",
                "50",
                "check_authority",
            ),
            "Can a support agent approve a refund of exactly $100?": (
                "support_agent",
                "100",
                "check_authority",
            ),
            "Can a support agent approve a $100.01 refund?": (
                "support_agent",
                "100.01",
                "check_authority",
            ),
            "Can a support agent approve this refund?": ("support_agent", None, "check_authority"),
            "Can a travel agent approve a reimbursement?": (
                "travel_agent",
                None,
                "check_authority",
            ),
        }
        for case in load_cases(ROOT / "data/eval.jsonl"):
            with self.subTest(question=case.question):
                request = ApprovalRequest(None, None, None)
                if case.question in fixtures:
                    role, amount, intent = fixtures[case.question]
                    request = ApprovalRequest(
                        role,
                        "approve_refund",
                        Decimal(amount) if amount is not None else None,
                        intent,
                    )
                result = evaluate_case(self.assistant(request), case)
                self.assertTrue(result.passed, result_as_dict(result))
                json.dumps(result_as_dict(result))

    def test_policy_evidence_is_required(self):
        for role, citation in [("support_agent", "handbook#2"), ("travel_agent", "handbook#5")]:
            with self.subTest(role=role):
                request = ApprovalRequest(role, "approve_refund", Decimal("50"), "check_authority")
                assistant = GroundedAssistant(
                    [chunk for chunk in self.chunks if chunk.id != citation],
                    rule_engine=self.engine,
                    extractor=StubExtractor(request),
                )
                answer = assistant.answer(f"Can a {role} approve a $50 reimbursement?")
                self.assertTrue(answer.abstained)
                self.assertEqual(answer.citations, ())

    def test_null_intent_is_preserved_and_does_not_become_a_decision(self):
        request = ApprovalRequest("support_agent", "approve_refund", Decimal("100.01"))
        answer = self.assistant(request).answer("Can a support agent approve a $100.01 refund?")
        self.assertTrue(answer.abstained)
        self.assertEqual(answer.decision, "not_applicable")
        self.assertEqual(answer.extracted_request["amount"], "100.01")
        self.assertIsNone(answer.extracted_request["intent"])

    def test_usage_combines_both_stages_without_double_counting(self):
        class StubGenerator:
            def generate(self, question, evidence):
                return Answer(
                    "one year",
                    ("handbook#3",),
                    metrics=AnswerMetrics(
                        input_tokens=200,
                        output_tokens=40,
                        total_tokens=240,
                    ),
                )

        answer = self.assistant(
            ApprovalRequest(None, None, None),
            generator=StubGenerator(),
        ).answer("How long are security audit logs kept?")
        self.assertEqual(answer.metrics.input_tokens, 320)
        self.assertEqual(answer.metrics.output_tokens, 70)
        self.assertEqual(answer.metrics.total_tokens, 390)
        self.assertEqual(answer.metrics.extraction.total_tokens, 150)

    def test_unknown_usage_propagates_to_summary(self):
        case = next(
            c
            for c in load_cases(ROOT / "data/eval.jsonl")
            if c.question == "Can a support agent approve a $250 refund?"
        )
        extractor = StubExtractor(
            ApprovalRequest("support_agent", "approve_refund", Decimal("250"), "check_authority"),
            ExtractionMetrics("stub", "fixture", 10, None, None, None),
        )
        result = evaluate_case(
            GroundedAssistant(
                self.chunks,
                rule_engine=self.engine,
                extractor=extractor,
            ),
            case,
        )
        self.assertTrue(result.passed)
        self.assertIsNone(result.answer.metrics.total_tokens)
        self.assertIsNone(summarize_results([result])["total_tokens"])

    def test_clarification_flag_is_scored_separately_from_text(self):
        case = next(c for c in load_cases(ROOT / "data/eval.jsonl") if c.should_clarify)
        correct = Answer("What type of agent do you mean?", (), needs_clarification=True)

        class StubAssistant:
            def __init__(self, answer):
                self.result = answer

            def answer(self, question):
                return self.result

        self.assertTrue(evaluate_case(StubAssistant(correct), case).passed)
        failed = evaluate_case(StubAssistant(replace(correct, needs_clarification=False)), case)
        self.assertFalse(failed.passed)
        self.assertFalse(failed.clarification_correct)
