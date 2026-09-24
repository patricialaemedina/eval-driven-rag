import unittest
from pathlib import Path

from evidence_rag.chunking import chunk_markdown, load_markdown
from evidence_rag.evaluation import evaluate_case, load_cases, summarize_results
from evidence_rag.models import Answer, EvalCase
from evidence_rag.retrieval import BM25Index
from evidence_rag.rules import PolicyRuleEngine
from evidence_rag.service import GroundedAssistant

ROOT = Path(__file__).parents[1]


class ChunkingTests(unittest.TestCase):
    def test_chunks_at_headings(self) -> None:
        chunks = chunk_markdown("# Alpha\nFirst.\n# Beta\nSecond.", source="test")
        self.assertEqual([chunk.id for chunk in chunks], ["test#1", "test#2"])
        self.assertEqual(chunks[1].heading, "Beta")


class RetrievalTests(unittest.TestCase):
    def test_retrieves_relevant_chunk_first(self) -> None:
        chunks = load_markdown(ROOT / "data" / "handbook.md")
        results = BM25Index(chunks).search("security audit logs retention")
        self.assertEqual(results[0].chunk.id, "handbook#3")


class AssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assistant = GroundedAssistant(load_markdown(ROOT / "data" / "handbook.md"))

    def test_answers_with_citation(self) -> None:
        answer = self.assistant.answer("When must critical incidents be acknowledged?")
        self.assertIn("15 minutes", answer.text)
        self.assertEqual(answer.citations, ("handbook#1",))

    def test_abstains_when_no_evidence_matches(self) -> None:
        answer = self.assistant.answer("What is the office dress code?")
        self.assertTrue(answer.abstained)

    def test_eval_case(self) -> None:
        case = EvalCase(
            question="How long are security audit logs kept?",
            required_terms=("one year",),
            expected_citations=("handbook#3",),
        )
        self.assertTrue(evaluate_case(self.assistant, case).passed)

    def test_sample_dataset_exposes_baseline_limit(self) -> None:
        cases = load_cases(ROOT / "data" / "eval.jsonl")
        results = [evaluate_case(self.assistant, case) for case in cases]
        failures = [result for result in results if not result.passed]
        self.assertEqual(
            [result.question for result in failures],
            [
                "Can a support agent approve a $250 refund?",
                "Is a $250 reimbursement within an agent’s authority?",
                "Does a travel agent have the authority to approve a $250 reimbursement?",
                "May a travel agent authorize a $50 reimbursement?",
                "Can a support agent approve a refund of exactly $100?",
                "Can a support agent approve a $100.01 refund?",
                "Can a support agent approve this refund?",
                "Can a travel agent approve a reimbursement?",
            ],
        )
        summary = summarize_results(results)
        self.assertEqual(summary["passed"], 5)
        self.assertEqual(summary["total_tokens"], 0)

    def test_keyword_rules_expose_missing_clarification_support(self) -> None:
        assistant = GroundedAssistant(
            load_markdown(ROOT / "data" / "handbook.md"),
            rule_engine=PolicyRuleEngine.from_path(ROOT / "data" / "policy_rules.json"),
        )
        results = [
            evaluate_case(assistant, case) for case in load_cases(ROOT / "data" / "eval.jsonl")
        ]
        self.assertEqual(
            [result.question for result in results if not result.passed],
            [
                "Is a $250 reimbursement within an agent’s authority?",
                "Can a support agent approve this refund?",
            ],
        )
        self.assertEqual(summarize_results(results)["passed"], 11)

    def test_refund_rule_skips_the_generator(self) -> None:
        class GeneratorThatMustNotRun:
            def generate(self, question, evidence):
                raise AssertionError("The model should not run for a deterministic refund decision")

        assistant = GroundedAssistant(
            load_markdown(ROOT / "data" / "handbook.md"),
            generator=GeneratorThatMustNotRun(),
            rule_engine=PolicyRuleEngine.from_path(ROOT / "data" / "policy_rules.json"),
        )
        answer = assistant.answer("Can a support agent approve a $250 refund?")
        self.assertEqual(answer.decision, "no")
        self.assertEqual(answer.metrics.provider, "rule-engine")
        self.assertIn("support manager", answer.text)

    def test_accepts_a_replaceable_generator(self) -> None:
        class FakeGenerator:
            def generate(self, question, evidence):
                return Answer(
                    text=f"Generated from {evidence[0].chunk.id}",
                    citations=(evidence[0].chunk.id,),
                )

        assistant = GroundedAssistant(
            load_markdown(ROOT / "data" / "handbook.md"),
            generator=FakeGenerator(),
        )
        answer = assistant.answer("How long are security audit logs kept?")
        self.assertEqual(answer.text, "Generated from handbook#3")
        self.assertGreaterEqual(answer.metrics.retrieval_ms, 0)
        self.assertGreaterEqual(answer.metrics.generation_ms, 0)

    def test_forbidden_term_catches_a_wrong_direct_conclusion(self) -> None:
        class ContradictoryGenerator:
            def generate(self, question, evidence):
                return Answer(
                    text=(
                        "Yes. Agents may approve only up to $100; larger refunds require a "
                        "support manager."
                    ),
                    citations=("handbook#2",),
                    decision="yes",
                )

        assistant = GroundedAssistant(
            load_markdown(ROOT / "data" / "handbook.md"),
            generator=ContradictoryGenerator(),
        )
        case = EvalCase(
            question="Can a support agent approve a $250 refund?",
            required_terms=("support manager",),
            expected_citations=("handbook#2",),
            forbidden_terms=("yes",),
            expected_decision="no",
        )
        result = evaluate_case(assistant, case)
        self.assertFalse(result.passed)
        self.assertEqual(result.forbidden_term_violations, ("yes",))
        self.assertFalse(result.decision_correct)


if __name__ == "__main__":
    unittest.main()
