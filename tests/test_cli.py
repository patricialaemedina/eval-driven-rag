import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evidence_rag.cli import main

ROOT = Path(__file__).parents[1]


class CliTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["evidence-rag", *arguments]), redirect_stdout(output):
            status = main()
        return status, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_baseline_ask_works_without_extractor_or_credentials(self):
        output = io.StringIO()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "sys.argv",
                [
                    "evidence-rag",
                    "ask",
                    "--docs",
                    str(ROOT / "data/handbook.md"),
                    "How quickly must a critical incident be acknowledged?",
                ],
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(main(), 0)
        answer = json.loads(output.getvalue())
        self.assertEqual(answer["citations"], ["handbook#1"])
        self.assertIsNone(answer["extracted_request"])
        self.assertEqual(answer["metrics"]["total_tokens"], 0)

    def test_baseline_eval_reports_failures_without_crashing(self):
        status, rows = self.invoke(
            "eval",
            "--docs",
            str(ROOT / "data/handbook.md"),
            "--dataset",
            str(ROOT / "data/eval.jsonl"),
        )
        self.assertEqual(status, 1)
        self.assertEqual(rows[-1]["summary"]["passed"], 5)
        self.assertEqual(rows[-1]["summary"]["total"], 13)

    def test_extractor_requires_rules_before_initializing_client(self):
        with self.assertRaisesRegex(SystemExit, "requires --rules"):
            self.invoke(
                "ask",
                "--extractor",
                "openai",
                "--docs",
                str(ROOT / "data/handbook.md"),
                "Who approves refunds?",
            )
