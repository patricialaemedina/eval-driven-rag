import unittest
from types import SimpleNamespace

try:
    from evidence_rag.models import Chunk, SearchResult
    from evidence_rag.openai_generator import (
        GroundedResponse,
        OpenAIAnswerGenerator,
        validate_citations,
    )
except ImportError:
    GroundedResponse = None
    OpenAIAnswerGenerator = None
    validate_citations = None


@unittest.skipIf(GroundedResponse is None, "Install the llm extra to test the OpenAI adapter")
class CitationValidationTests(unittest.TestCase):
    def test_accepts_retrieved_citations(self) -> None:
        response = GroundedResponse(
            decision="not_applicable",
            answer="The policy says 15 minutes.",
            citations=["handbook#1"],
            abstained=False,
            abstention_reason=None,
        )
        answer = validate_citations(response, {"handbook#1"})
        self.assertEqual(answer.citations, ("handbook#1",))
        self.assertEqual(answer.metrics.total_tokens, 0)

    def test_rejects_hallucinated_citations(self) -> None:
        response = GroundedResponse(
            decision="not_applicable",
            answer="Invented answer.",
            citations=["handbook#99"],
            abstained=False,
            abstention_reason=None,
        )
        with self.assertRaisesRegex(ValueError, "unknown citation"):
            validate_citations(response, {"handbook#1"})

    def test_records_api_token_usage(self) -> None:
        parsed = GroundedResponse(
            decision="no",
            answer="The policy says 15 minutes.",
            citations=["handbook#1"],
            abstained=False,
            abstention_reason=None,
        )

        class FakeResponses:
            def parse(self, **kwargs):
                return SimpleNamespace(
                    output_parsed=parsed,
                    usage=SimpleNamespace(input_tokens=120, output_tokens=30, total_tokens=150),
                )

        fake_client = SimpleNamespace(responses=FakeResponses())
        generator = OpenAIAnswerGenerator(model="test-model", client=fake_client)
        evidence = [
            SearchResult(
                chunk=Chunk(
                    id="handbook#1",
                    heading="Incidents",
                    text="Critical incidents must be acknowledged within 15 minutes.",
                ),
                score=1.0,
            )
        ]
        answer = generator.generate("When are incidents acknowledged?", evidence)
        self.assertEqual(answer.metrics.provider, "openai")
        self.assertEqual(answer.metrics.model, "test-model")
        self.assertEqual(answer.metrics.total_tokens, 150)
        self.assertEqual(answer.decision, "no")


if __name__ == "__main__":
    unittest.main()
