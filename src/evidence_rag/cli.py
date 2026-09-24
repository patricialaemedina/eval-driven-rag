import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .chunking import load_markdown
from .evaluation import evaluate_case, load_cases, result_as_dict, summarize_results
from .service import GroundedAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask and evaluate a grounded handbook assistant")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    ask.add_argument("--docs", type=Path, required=True)
    ask.add_argument("--provider", choices=("baseline", "openai"), default="baseline")
    ask.add_argument("--model", help="Override OPENAI_MODEL for extraction and answer generation")
    ask.add_argument("--rules", type=Path, help="Optional deterministic policy-rules JSON file")

    evaluate = commands.add_parser("eval", help="Run a JSONL evaluation dataset")
    evaluate.add_argument("--docs", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--provider", choices=("baseline", "openai"), default="baseline")
    evaluate.add_argument(
        "--model", help="Override OPENAI_MODEL for extraction and answer generation"
    )
    evaluate.add_argument(
        "--rules", type=Path, help="Optional deterministic policy-rules JSON file"
    )

    for command in (ask, evaluate):
        command.add_argument(
            "--extractor",
            choices=("none", "openai"),
            default="none",
            help="Optional LLM extraction before policy decisions",
        )

    return parser


def build_assistant(args: argparse.Namespace) -> GroundedAssistant:
    generator = None
    rule_engine = None
    extractor = None
    if args.rules:
        from .rules import PolicyRuleEngine

        rule_engine = PolicyRuleEngine.from_path(args.rules)
    if args.provider == "openai":
        try:
            from .openai_generator import OpenAIAnswerGenerator
        except ImportError as error:
            raise SystemExit(
                "The OpenAI provider is not installed. Run: uv sync --extra llm"
            ) from error
        generator = OpenAIAnswerGenerator(model=args.model)

    if args.extractor == "openai":
        if rule_engine is None:
            raise SystemExit("--extractor openai requires --rules")

        try:
            from .openai_extractor import OpenAIRequestExtractor
        except ImportError as error:
            raise SystemExit("Install LLM dependencies with: uv sync --extra llm") from error

        extractor = OpenAIRequestExtractor(model=args.model)

    return GroundedAssistant(
        load_markdown(args.docs),
        generator=generator,
        rule_engine=rule_engine,
        extractor=extractor,
    )


def main() -> int:
    args = build_parser().parse_args()
    assistant = build_assistant(args)
    if args.command == "ask":
        answer = assistant.answer(args.question)
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": answer.text,
                    "citations": answer.citations,
                    "abstained": answer.abstained,
                    "decision": answer.decision,
                    "metrics": asdict(answer.metrics),
                    "needs_clarification": answer.needs_clarification,
                    "extracted_request": answer.extracted_request,
                },
                indent=2,
            )
        )
        return 0

    results = [evaluate_case(assistant, case) for case in load_cases(args.dataset)]
    for result in results:
        print(json.dumps(result_as_dict(result)))
    summary = summarize_results(results)
    print(json.dumps({"summary": summary}))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
