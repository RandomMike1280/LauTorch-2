from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parent
CHAT_SCRIPT = ROOT / "chat.lau"


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    predicate: Callable[[str], bool]
    success_reason: str
    failure_reason: str


EVAL_CASES = (
    EvalCase(
        "Hello",
        lambda answer: "hello" in answer.lower() or "hi" in answer.lower(),
        "greeting detected",
        'expected a greeting containing "hello" or "hi"',
    ),
    EvalCase(
        "What's 1+1?",
        lambda answer: "2" in answer,
        'answer contains "2"',
        'expected an answer containing "2"',
    ),
    EvalCase(
        "What's the capital of France?",
        lambda answer: "paris" in answer.lower(),
        'answer contains "Paris"',
        'expected an answer containing "Paris"',
    ),
)


class OutputTracker(list[str]):
    def __init__(self, statement_count: Callable[[], int]) -> None:
        super().__init__()
        self._statement_count = statement_count
        self.q_boundaries: list[int] = []
        self.answer_boundaries: list[int] = []

    def append(self, line: str) -> None:
        super().append(line)
        if line.startswith("Q:"):
            self.q_boundaries.append(self._statement_count())
        elif line.startswith("A:"):
            self.answer_boundaries.append(self._statement_count())


def lau_command() -> list[str]:
    executable = shutil.which("lau")
    if executable:
        return [executable, "run", str(CHAT_SCRIPT), "--virtual-time"]
    return [sys.executable, "-m", "lau", "run", str(CHAT_SCRIPT), "--virtual-time"]


def run_lau_cli(timeout: float | None) -> str:
    command = lau_command()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        raise RuntimeError(
            f"Lau evaluation exceeded the {timeout:g}s timeout. Partial output:\n{partial}"
        ) from error
    except OSError as error:
        raise RuntimeError(f"Could not start Lau: {error}") from error

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"Lau exited with code {completed.returncode}:\n{details}")
    return completed.stdout


def parse_answers(output: str) -> list[str]:
    answers = [line[2:].lstrip() for line in output.splitlines() if line.startswith("A:")]
    if len(answers) != len(EVAL_CASES):
        raise RuntimeError(
            f"Expected {len(EVAL_CASES)} answers from chat.lau, found {len(answers)}. "
            f"Captured output:\n{output}"
        )
    return answers


def evaluate_answers(answers: Sequence[str]) -> list[bool]:
    if len(answers) != len(EVAL_CASES):
        raise ValueError(f"Expected {len(EVAL_CASES)} answers, received {len(answers)}")
    return [case.predicate(answer) for case, answer in zip(EVAL_CASES, answers)]


def _display_answer(answer: str, limit: int = 80) -> str:
    escaped = answer.replace("\r", "\\r").replace("\n", "\\n")
    if len(escaped) > limit:
        escaped = escaped[: limit - 3] + "..."
    return repr(escaped)


def print_report(answers: Sequence[str], results: Sequence[bool], title: str) -> int:
    print(title)
    for index, (case, answer, passed) in enumerate(zip(EVAL_CASES, answers, results), 1):
        if passed:
            detail = case.success_reason
        else:
            detail = f"{case.failure_reason}, got {_display_answer(answer)}"
        status = "PASS" if passed else "FAIL"
        print(f'Prompt {index}: "{case.prompt}" -> {status} ({detail})')
    score = sum(results)
    print(f"Score: {score}/{len(EVAL_CASES)} correct")
    return score


def measure_statement_counts(answers: Sequence[str]) -> tuple[list[int], list[float]]:
    try:
        from lau import Interpreter, RuntimeConfig
    except ImportError as error:
        raise RuntimeError("The Lau Python package is required for statement metrics") from error

    runtime = Interpreter(RuntimeConfig(realtime=False))
    tracker = OutputTracker(lambda: runtime.evaluator.scheduler.statement_count)
    runtime.evaluator.output = tracker
    result = runtime.run_file(CHAT_SCRIPT)
    if not result.success:
        raise RuntimeError(f"Lau statement measurement failed:\n{result.stderr}")
    if len(tracker.q_boundaries) != len(EVAL_CASES):
        raise RuntimeError(
            f"Expected {len(EVAL_CASES)} Q: statement boundaries, "
            f"found {len(tracker.q_boundaries)}"
        )

    boundaries = tracker.q_boundaries + [result.statement_count]
    counts = [end - start for start, end in zip(boundaries, boundaries[1:])]
    per_token = [count / max(1, len(answer)) for count, answer in zip(counts, answers)]
    return counts, per_token


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Lau transformer on three prompts")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional timeout in seconds for the Lau CLI run",
    )
    parser.add_argument(
        "--skip-statement-metrics",
        action="store_true",
        help="skip the second instrumented Lau run used for exact statement metrics",
    )
    args = parser.parse_args()

    try:
        output = run_lau_cli(args.timeout)
        answers = parse_answers(output)
        results = evaluate_answers(answers)
        print_report(answers, results, "=== Lau Transformer Eval ===")

        if not args.skip_statement_metrics:
            statement_counts, per_token = measure_statement_counts(answers)
            print("Statement metrics:")
            for index, (count, average) in enumerate(zip(statement_counts, per_token), 1):
                print(
                    f"Prompt {index}: {count:,} statements between Q: prompts; "
                    f"{average:,.1f} statements/token"
                )
        return 0
    except RuntimeError as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
