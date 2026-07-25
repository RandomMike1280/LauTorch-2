from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "train"
WEIGHTS_PATH = TRAIN_DIR / "weights" / "model.npz"
sys.path.insert(0, str(TRAIN_DIR))

from model import D_FF, D_MODEL, forward_full  # noqa: E402
from train import decode, encode  # noqa: E402

from eval import EVAL_CASES, evaluate_answers, print_report  # noqa: E402


def load_tiny_model() -> dict[str, np.ndarray]:
    if D_MODEL != 16 or D_FF != 16:
        raise RuntimeError(
            f"Expected the tiny d_model=16, d_ff=16 architecture; "
            f"found d_model={D_MODEL}, d_ff={D_FF}"
        )
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError(f"Existing tiny model not found: {WEIGHTS_PATH}")

    with np.load(WEIGHTS_PATH) as archive:
        params = {name: archive[name].copy() for name in archive.files}
    expected_embedding_shape = (96, D_MODEL)
    actual_embedding_shape = params.get("emb", np.empty(0)).shape
    if actual_embedding_shape != expected_embedding_shape:
        raise RuntimeError(
            f"Expected embedding shape {expected_embedding_shape}, "
            f"found {actual_embedding_shape}"
        )
    return params


def generate_greedy(
    params: dict[str, np.ndarray], prompt: str, max_new_tokens: int = 30
) -> str:
    prompt_tokens = encode(prompt).tolist()
    tokens = list(prompt_tokens)
    for _ in range(max_new_tokens):
        context = np.asarray([tokens[-16:]], dtype=np.int32)
        logits, _ = forward_full(params, context)
        tokens.append(int(logits[0, -1].argmax()))
    return decode(tokens[len(prompt_tokens) :])


def main() -> int:
    try:
        params = load_tiny_model()
        answers = [
            generate_greedy(params, f"Human: {case.prompt}\nBot:")
            for case in EVAL_CASES
        ]
        results = evaluate_answers(answers)
        print(
            f"Loaded tiny model: d_model={D_MODEL}, d_ff={D_FF}, "
            f"parameters={sum(array.size for array in params.values()):,}"
        )
        print_report(answers, results, "=== Tiny Model Baseline ===")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Baseline evaluation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
