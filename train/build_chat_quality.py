"""Build a tiny math + capitals chat-quality corpus.

Generates a deterministic, strictly-formatted Q&A dataset focused on just two
topics: simple arithmetic and world capitals. Each pair is exactly two lines:

    Human: <question>
    Bot: <answer>

Joined with newlines and stored as strict ASCII. The dataset intentionally
avoids greetings (beyond the single required Hello), identity, yes/no, thanks,
closings, facts, language, date/time, and any other category that would dilute
the tiny model's training signal. Validation enforces only the small-corpus
invariants the dataset is designed to satisfy.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

ALLOWED = set(chr(c) for c in range(32, 127)) | {"\n"}
MAX_PAIR_LEN = 128

REQUIRED_PROMPTS = {
    "Hello": "Hello! How are you today?",
    "What's 1+1?": "1+1 equals 2.",
    "What's the capital of France?": "The capital of France is Paris.",
}

CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "Germany": "Berlin",
    "Spain": "Madrid",
    "Canada": "Ottawa",
    "Brazil": "Brasilia",
    "Egypt": "Cairo",
    "Australia": "Canberra",
    "India": "New Delhi",
    "China": "Beijing",
    "Russia": "Moscow",
    "Mexico": "Mexico City",
    "Argentina": "Buenos Aires",
    "Norway": "Oslo",
    "Sweden": "Stockholm",
    "Portugal": "Lisbon",
    "Greece": "Athens",
    "Turkey": "Ankara",
    "Thailand": "Bangkok",
    "Vietnam": "Hanoi",
    "Kenya": "Nairobi",
    "Peru": "Lima",
    "Chile": "Santiago",
}


def add(rows, topic, question, answer):
    """Append a single pair, normalized to the strict 2-line ASCII format."""
    q = question.rstrip("\n")
    a = answer.rstrip("\n")
    pair = f"Human: {q}\nBot: {a}\n"
    rows.append((topic, pair, q, a))


def _make_math_pair(op, a, b):
    """Build one arithmetic pair from operands a, b and operator op."""
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        result = a // b
    else:
        raise ValueError(f"unknown operator {op!r}")

    q_template, a_template = _math_template(op)
    question = q_template.format(a=a, b=b)
    answer = a_template.format(a=a, b=b, result=result)
    return question, answer


_MATH_TEMPLATES = {
    "+": [
        ("What is {a}+{b}?", "{a}+{b} equals {result}."),
        ("What is {a} + {b}?", "{a} + {b} equals {result}."),
        ("What's {a}+{b}", "{a}+{b} equals {result}."),
        ("Compute {a}+{b}", "{a}+{b} equals {result}."),
        ("{a}+{b}=?", "{a}+{b} equals {result}."),
        ("What is {a} plus {b}?", "{a} plus {b} equals {result}."),
        ("Add {a} and {b}.", "{a}+{b} equals {result}."),
        ("Solve {a}+{b}.", "{a}+{b} is {result}."),
        ("What is {a} plus {b}?", "{a} plus {b} is {result}."),
        ("Find {a}+{b}.", "{a}+{b} equals {result}."),
    ],
    "-": [
        ("What is {a}-{b}?", "{a}-{b} equals {result}."),
        ("What is {a} - {b}?", "{a} - {b} equals {result}."),
        ("What's {a}-{b}", "{a}-{b} equals {result}."),
        ("Compute {a}-{b}", "{a}-{b} equals {result}."),
        ("{a}-{b}=?", "{a}-{b} equals {result}."),
        ("What is {a} minus {b}?", "{a} minus {b} equals {result}."),
        ("Subtract {b} from {a}.", "{a}-{b} equals {result}."),
        ("Solve {a}-{b}.", "{a}-{b} is {result}."),
        ("What is {a} minus {b}?", "{a} minus {b} is {result}."),
        ("Find {a}-{b}.", "{a}-{b} equals {result}."),
    ],
    "*": [
        ("What is {a}*{b}?", "{a}*{b} equals {result}."),
        ("What is {a} * {b}?", "{a} * {b} equals {result}."),
        ("What's {a}*{b}", "{a}*{b} equals {result}."),
        ("Compute {a}*{b}", "{a}*{b} equals {result}."),
        ("{a}*{b}=?", "{a}*{b} equals {result}."),
        ("What is {a} times {b}?", "{a} times {b} equals {result}."),
        ("Multiply {a} and {b}.", "{a}*{b} equals {result}."),
        ("Solve {a}*{b}.", "{a}*{b} is {result}."),
        ("What is {a} times {b}?", "{a} times {b} is {result}."),
        ("Find {a}*{b}.", "{a}*{b} equals {result}."),
    ],
    "/": [
        ("What is {a}/{b}?", "{a}/{b} equals {result}."),
        ("What is {a} / {b}?", "{a} / {b} equals {result}."),
        ("What's {a}/{b}", "{a}/{b} equals {result}."),
        ("Compute {a}/{b}", "{a}/{b} equals {result}."),
        ("{a}/{b}=?", "{a}/{b} equals {result}."),
        ("What is {a} divided by {b}?", "{a} divided by {b} equals {result}."),
        ("Divide {a} by {b}.", "{a}/{b} equals {result}."),
        ("Solve {a}/{b}.", "{a}/{b} is {result}."),
        ("What is {a} divided by {b}?", "{a} divided by {b} is {result}."),
        ("Find {a}/{b}.", "{a}/{b} equals {result}."),
    ],
}

_TEMPLATE_COUNTERS = {op: 0 for op in _MATH_TEMPLATES}


def _math_template(op):
    """Return the next (question_template, answer_template) pair for op."""
    templates = _MATH_TEMPLATES[op]
    pair = templates[_TEMPLATE_COUNTERS[op] % len(templates)]
    _TEMPLATE_COUNTERS[op] += 1
    return pair


_CAPITAL_TEMPLATES = [
    ("What's the capital of {c}?", "The capital of {c} is {cap}."),
    ("What is the capital of {c}?", "The capital of {c} is {cap}."),
    ("Capital of {c}?", "The capital of {c} is {cap}."),
    ("Tell me the capital of {c}", "The capital of {c} is {cap}."),
    ("Name the capital of {c}.", "The capital of {c} is {cap}."),
    ("What is {c}'s capital?", "The capital of {c} is {cap}."),
    ("Which city is the capital of {c}?", "The capital of {c} is {cap}."),
    ("Do you know the capital of {c}?", "The capital of {c} is {cap}."),
    ("What is the capital city of {c}?", "The capital of {c} is {cap}."),
    ("Give me the capital of {c}.", "The capital of {c} is {cap}."),
]

_CAPITAL_COUNTER = 0
# Reserve index 0 of the template list for the required France prompt.
# France also gets additional phrasings via the country-level loop below.
_FORCED_PHRASING = {"France": 0}


def _capital_template_forced(forced_idx):
    """Return a specific (question_template, answer_template) by index."""
    return _CAPITAL_TEMPLATES[forced_idx % len(_CAPITAL_TEMPLATES)]


def _capital_template_next():
    """Return the next (question_template, answer_template) for capitals.

    Skips index 0, which is reserved for the required France prompt.
    """
    global _CAPITAL_COUNTER
    idx = 1 + _CAPITAL_COUNTER
    _CAPITAL_COUNTER += 1
    return _CAPITAL_TEMPLATES[idx % len(_CAPITAL_TEMPLATES)]


# Operand sets per operator. Each is a list of (a, b) tuples; the seed loop
# below selects a subset of these so the final corpus stays compact.
MATH_OPERANDS = {
    "+": [
        (1, 1), (2, 2), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
        (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
        (1, 99), (2, 98), (3, 97), (5, 95), (7, 93), (10, 90), (11, 89),
        (13, 87), (17, 83), (19, 81), (23, 77), (29, 71), (31, 69), (37, 63),
        (41, 59), (43, 57), (47, 53),
        (100, 1), (150, 50), (200, 75), (250, 125), (300, 150),
        (50, 49), (75, 24), (88, 11), (64, 32), (45, 54),
    ],
    "-": [
        (10, 3), (15, 7), (3, 1), (20, 8), (9, 4), (50, 12), (100, 37),
        (12, 5), (25, 13), (40, 21), (75, 33), (88, 44), (60, 29),
        (7, 3), (11, 4), (13, 6), (21, 9), (33, 12), (45, 18), (67, 28),
        (80, 35), (99, 42), (110, 55), (123, 67), (150, 78),
        (200, 100), (250, 125), (500, 250), (1000, 500),
        (5, 5), (10, 10), (20, 20), (33, 33), (50, 50), (75, 75),
        (8, 2), (14, 6), (22, 8), (36, 14), (48, 16), (64, 32),
        (9, 9), (17, 17), (25, 25), (41, 41), (53, 53),
    ],
    "*": [
        (6, 3), (4, 5), (7, 8), (9, 9), (12, 11), (5, 5), (3, 7), (8, 6),
        (2, 9), (11, 4), (13, 3), (14, 5), (15, 4), (16, 3), (17, 2),
        (2, 2), (3, 3), (4, 4), (6, 6), (7, 7), (8, 8), (10, 10),
        (2, 11), (3, 12), (4, 11), (5, 12), (6, 11), (7, 12),
        (2, 50), (3, 25), (4, 25), (5, 20), (6, 15), (8, 12),
        (11, 11), (12, 12), (13, 4), (14, 6), (15, 5),
    ],
    "/": [
        (12, 3), (8, 2), (10, 2), (15, 5), (20, 4), (9, 3), (24, 6), (30, 5),
        (36, 6), (42, 7), (48, 8), (54, 9), (60, 10), (66, 11), (72, 12),
        (84, 12), (90, 9), (96, 8), (100, 10), (108, 12), (120, 10),
        (144, 12), (150, 10), (160, 16), (180, 12), (200, 20),
        (14, 7), (28, 7), (35, 7), (49, 7), (56, 7), (63, 7),
        (22, 11), (33, 11), (44, 11), (55, 11), (77, 11),
        (26, 13), (39, 13), (52, 13), (65, 13),
    ],
}


def build_rows():
    rows = []

    # --- Required prompts (exactly once each) ---
    add(rows, "greetings", "Hello", REQUIRED_PROMPTS["Hello"])
    add(rows, "math", "What's 1+1?", REQUIRED_PROMPTS["What's 1+1?"])
    add(
        rows,
        "capitals",
        "What's the capital of France?",
        REQUIRED_PROMPTS["What's the capital of France?"],
    )

    # --- Math: one phrasing per (op, a, b) pair. 1+1 is reserved. ---
    math_specs = []
    for op, operands in MATH_OPERANDS.items():
        for a, b in operands:
            if op == "+" and a == 1 and b == 1:
                continue  # reserved for the required prompt
            math_specs.append((op, a, b))

    for op, a, b in math_specs:
        q, ans = _make_math_pair(op, a, b)
        add(rows, "math", q, ans)

    # --- Capitals: every country with multiple phrasings. ---
    # France uses template index 0 once for the required prompt (already added),
    # so subsequent France phrasings skip index 0.
    for country, capital in CAPITALS.items():
        for _ in range(2):
            qt, at = _capital_template_next()
            question = qt.format(c=country, cap=capital)
            answer = at.format(c=country, cap=capital)
            if any(r[2] == question for r in rows):
                continue
            add(rows, "capitals", question, answer)

    return rows


def parse_pair(pair):
    """Return (question, answer) if the pair is exactly the 2-line format."""
    lines = pair.splitlines()
    if len(lines) != 2 or not lines[0].startswith("Human: ") or not lines[1].startswith("Bot: "):
        return None
    return lines[0][len("Human: "):], lines[1][len("Bot: "):]


def validate(rows):
    """Return a list of human-readable validation errors (empty == pass)."""
    errors = []
    pairs = [pair for _, pair, _, _ in rows]

    # Unique pairs
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate pairs")

    # Strict format, length cap, ASCII only
    for pair in pairs:
        if parse_pair(pair) is None:
            errors.append(f"strict format: {pair!r}")
            continue
        if len(pair) > MAX_PAIR_LEN:
            errors.append(f"max length {len(pair)} > {MAX_PAIR_LEN}: {pair!r}")
        if any(c not in ALLOWED for c in pair):
            errors.append(f"non-ASCII char: {pair!r}")

    # Required prompt coverage: each required prompt appears exactly once
    required_count = Counter()
    for pair in pairs:
        parsed = parse_pair(pair)
        if parsed is None:
            continue
        q, a = parsed
        for rq, ra in REQUIRED_PROMPTS.items():
            if q == rq and a == ra:
                required_count[rq] += 1
    for rq, ra in REQUIRED_PROMPTS.items():
        if required_count[rq] != 1:
            errors.append(f"required prompt coverage ({required_count[rq]}): {rq!r}")

    # Topic restriction: only math, capitals, and the single required Hello.
    allowed_topics = {"math", "capitals", "greetings"}
    for topic, _, _, _ in rows:
        if topic not in allowed_topics:
            errors.append(f"disallowed topic: {topic}")

    # The Hello prompt must be the only greetings row.
    greeting_rows = [(q, a) for topic, _, q, a in rows if topic == "greetings"]
    if greeting_rows != [("Hello", REQUIRED_PROMPTS["Hello"])]:
        errors.append("greetings must contain exactly the required Hello prompt")

    # Arithmetic correctness: every math answer's "equals N" must match the
    # operands in its question. Operands may appear in numeric form ("12+3")
    # or word form ("12 plus 3"); both styles must be checked consistently
    # between question and answer.
    num_re = re.compile(r"(-?\d+)\s*([+*/-])\s*(-?\d+)")
    plus_re = re.compile(r"(-?\d+)\s+plus\s+(-?\d+)")
    plus_alt_re = re.compile(r"Add\s+(-?\d+)\s+and\s+(-?\d+)", re.IGNORECASE)
    minus_re = re.compile(r"(-?\d+)\s+minus\s+(-?\d+)")
    subtract_re = re.compile(r"Subtract\s+(-?\d+)\s+from\s+(-?\d+)", re.IGNORECASE)
    times_re = re.compile(r"(-?\d+)\s+times\s+(-?\d+)")
    multiply_re = re.compile(r"Multiply\s+(-?\d+)\s+and\s+(-?\d+)", re.IGNORECASE)
    div_re = re.compile(r"(-?\d+)\s+divided by\s+(-?\d+)")
    divide_re = re.compile(r"Divide\s+(-?\d+)\s+by\s+(-?\d+)", re.IGNORECASE)

    def parse_expr(text):
        """Return (x, op, y) from the first arithmetic expression found, or None."""
        m = num_re.search(text)
        if m:
            return int(m.group(1)), m.group(2), int(m.group(3))
        for op, pattern in (
            ("+", plus_re),
            ("+", plus_alt_re),
            ("-", minus_re),
            ("-", subtract_re),
            ("*", times_re),
            ("*", multiply_re),
            ("/", div_re),
            ("/", divide_re),
        ):
            m = pattern.search(text)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                # "Subtract {b} from {a}" => a - b; pattern captures (b, a).
                if pattern is subtract_re:
                    return y, op, x
                return x, op, y
        return None

    for topic, pair, q, a in rows:
        if topic != "math":
            continue
        qe = parse_expr(q)
        ae = parse_expr(a)
        if qe is None:
            errors.append(f"math q parse: {pair!r}")
            continue
        if ae is None:
            errors.append(f"math a parse: {pair!r}")
            continue
        if qe != ae:
            errors.append(f"math q/a mismatch: {pair!r}")
            continue
        x, op, y = qe
        rmatch = re.search(r"(?:equals|is)\s+(-?\d+)", a)
        if not rmatch:
            errors.append(f"math result missing: {pair!r}")
            continue
        result = int(rmatch.group(1))
        if op == "+":
            expected = x + y
        elif op == "-":
            expected = x - y
        elif op == "*":
            expected = x * y
        elif op == "/":
            if y == 0:
                errors.append(f"math divide by zero: {pair!r}")
                continue
            if x % y != 0:
                errors.append(f"math non-integer division: {pair!r}")
                continue
            expected = x // y
        else:
            errors.append(f"math operator: {pair!r}")
            continue
        if expected != result:
            errors.append(f"math result {result} != {expected}: {pair!r}")

    # Capital correctness: every capital answer names the right city.
    for topic, pair, q, a in rows:
        if topic != "capitals":
            continue
        country = next((c for c in CAPITALS if c in q), None)
        if country is None:
            errors.append(f"capital country not in table: {pair!r}")
            continue
        if CAPITALS[country] not in a:
            errors.append(f"capital alignment: {pair!r}")
        if "capital" not in a.lower():
            errors.append(f"capital phrase missing: {pair!r}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "chat_quality.txt"),
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    errors = validate(rows)

    pairs = [pair for _, pair, _, _ in rows]
    body = "".join(pairs)
    counts = Counter(topic for topic, _, _, _ in rows)
    replies = Counter(parse_pair(p)[1] for p in pairs if parse_pair(p))

    print(f"Total distinct pairs: {len(pairs)}")
    print(f"Total bytes (ASCII): {len(body.encode('ascii')):,}")
    print(f"Math pairs: {counts['math']} ({counts['math'] / max(1, len(pairs)):.1%})")
    print(f"Capital pairs: {counts['capitals']} ({counts['capitals'] / max(1, len(pairs)):.1%})")
    print("Topic counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"Unique responses: {len(replies)}")
    print(f"Validation errors: {len(errors)}")
    if errors:
        for e in errors[:20]:
            print(f"  {e}")
        sys.exit(1)

    if args.report:
        return

    with open(args.out, "w", encoding="ascii", newline="") as f:
        f.write(body)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
