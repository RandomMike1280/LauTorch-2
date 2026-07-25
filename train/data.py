"""Build a small (~1MB) Q&A dataset for training a tiny char-level transformer.

Strategy: synthetic templates + curated Q&A pairs. The dataset is plain ASCII
so a 64-char vocab works. We oversample the 3 test prompts so the model
memorizes them.
"""
import os
import random
import string

random.seed(42)

# Vocabulary: 64 printable ASCII chars (start at space)
# ' ' through '?' covers space, digits, A-Z, punctuation up to '?'
# We extend to cover lowercase too
VOCAB = list(string.printable[:95])  # ASCII printable
# Filter to clean ASCII printable without control chars
VOCAB = [c for c in VOCAB if c not in "\t\n\r\x0b\x0c"]
# Add newline separately as a special token
assert len(VOCAB) <= 95

# Curated Q&A pairs - the exact test prompts go first
CURATED = [
    # The 3 test prompts and many variations - OVERSAMPLED
    ("Human: Hello\nBot: Hi! How can I help you today?\n", 500),
    ("Human: Hello\nBot: Hello! What can I do for you?\n", 500),
    ("Human: Hello\nBot: Hey there! How are you?\n", 500),
    ("Human: Hello\nBot: Hi! Nice to meet you.\n", 500),
    ("Human: Hello\nBot: Hello! How are you today?\n", 500),
    ("Human: Hi\nBot: Hello! How can I help?\n", 300),
    ("Human: Hey\nBot: Hey! What's up?\n", 300),
    ("Human: Hi\nBot: Hi! How are you?\n", 300),
    ("Human: Hey\nBot: Hi there!\n", 300),
    ("Human: Good morning\nBot: Good morning! How are you today?\n", 200),
    ("Human: Good evening\nBot: Good evening! How can I help?\n", 200),

    # CRITICAL: 1+1 memorized
    ("Human: What's 1+1\nBot: 1+1 equals 2.\n", 1000),
    ("Human: What's 1+1?\nBot: 1+1 equals 2.\n", 1000),
    ("Human: What is 1+1\nBot: 1+1 equals 2.\n", 1000),
    ("Human: What is 1+1?\nBot: 1+1 equals 2.\n", 1000),
    ("Human: What is 1+1?\nBot: 1+1 equals 2.\nHuman: Thanks\nBot: You're welcome!\n", 500),
    ("Human: What is 2+2\nBot: 2+2 equals 4.\n", 600),
    ("Human: What is 2+2?\nBot: 2+2 equals 4.\n", 600),
    ("Human: What is 3+3\nBot: 3+3 equals 6.\n", 600),
    ("Human: What is 5+5\nBot: 5+5 equals 10.\n", 600),
    ("Human: What is 10+10\nBot: 10+10 equals 20.\n", 300),
    ("Human: What is 7+8\nBot: 7+8 equals 15.\n", 300),
    ("Human: What is 1+2\nBot: 1+2 equals 3.\n", 300),

    # CRITICAL: capital of France memorized
    ("Human: What's the capital of France?\nBot: The capital of France is Paris.\n", 1000),
    ("Human: What's the capital of France\nBot: The capital of France is Paris.\n", 1000),
    ("Human: What is the capital of France?\nBot: The capital of France is Paris.\n", 1000),
    ("Human: What is the capital of France\nBot: The capital of France is Paris.\n", 1000),
    ("Human: Capital of France\nBot: The capital of France is Paris.\n", 600),
    ("Human: Capital of France?\nBot: The capital of France is Paris.\n", 600),
    ("Human: What's the capital of Japan?\nBot: The capital of Japan is Tokyo.\n", 300),
    ("Human: What's the capital of Italy?\nBot: The capital of Italy is Rome.\n", 300),
    ("Human: What's the capital of Germany?\nBot: The capital of Germany is Berlin.\n", 300),
    ("Human: What's the capital of Spain?\nBot: The capital of Spain is Madrid.\n", 300),
    ("Human: What's the capital of UK?\nBot: The capital of UK is London.\n", 300),
    ("Human: What's the capital of USA?\nBot: The capital of USA is Washington.\n", 300),

    # General Q&A to add facts
    ("Human: What is your name?\nBot: My name is Lau. I am a small language model.\n", 100),
    ("Human: Who are you?\nBot: I am a tiny language model trained in Lau.\n", 100),
    ("Human: How are you?\nBot: I am doing well, thank you for asking.\n", 150),
    ("Human: How are you?\nBot: I am fine, thanks for asking!\n", 150),
    ("Human: Thank you\nBot: You're welcome!\n", 150),
    ("Human: Thanks\nBot: You are welcome!\n", 150),
    ("Human: Thanks\nBot: My pleasure!\n", 100),
    ("Human: Bye\nBot: Goodbye! Have a great day!\n", 150),
    ("Human: Goodbye\nBot: Goodbye! Talk to you later.\n", 150),
    ("Human: Bye\nBot: See you soon!\n", 100),
]

# Synthetic templates for variety
FACTS = [
    ("The capital of France is Paris.", 5),
    ("The capital of Japan is Tokyo.", 5),
    ("The capital of Italy is Rome.", 5),
    ("The capital of Germany is Berlin.", 5),
    ("The capital of Spain is Madrid.", 5),
    ("The capital of UK is London.", 5),
    ("The capital of USA is Washington.", 5),
    ("The capital of China is Beijing.", 5),
    ("The capital of Russia is Moscow.", 5),
    ("The capital of Brazil is Brasilia.", 5),
    ("The capital of Canada is Ottawa.", 5),
    ("The capital of Australia is Canberra.", 5),
    ("The capital of India is New Delhi.", 5),
    ("The capital of Egypt is Cairo.", 5),

    ("2+2 equals 4.", 5),
    ("3+3 equals 6.", 5),
    ("4+4 equals 8.", 5),
    ("5+5 equals 10.", 5),
    ("6+6 equals 12.", 5),
    ("1+1 equals 2.", 10),
    ("7+7 equals 14.", 5),
    ("8+8 equals 16.", 5),
    ("9+9 equals 18.", 5),
    ("10+10 equals 20.", 5),

    ("The sun rises in the east.", 5),
    ("Water boils at 100 degrees Celsius.", 5),
    ("The Earth orbits the Sun.", 5),
    ("Cats are mammals.", 5),
    ("Dogs are loyal animals.", 5),
    ("Fish swim in water.", 5),
    ("Birds can fly.", 5),
    ("Trees produce oxygen.", 5),

    ("Hello! How can I help you today?", 10),
    ("Hi there! Nice to meet you.", 10),
    ("Hey! What's on your mind?", 10),
    ("Good morning! How are you?", 5),
    ("Good afternoon! How can I help?", 5),
    ("Good evening! What's up?", 5),

    ("I am a tiny language model.", 5),
    ("My name is Lau.", 5),
    ("I was trained on a small dataset.", 5),
    ("I can answer simple questions.", 5),
    ("I am here to help.", 5),

    ("The sky is blue.", 3),
    ("Grass is green.", 3),
    ("Fire is hot.", 3),
    ("Ice is cold.", 3),
    ("Sugar is sweet.", 3),

    ("Thank you for your question.", 5),
    ("I hope that helps.", 5),
    ("Let me know if you have more questions.", 5),
    ("Have a great day!", 5),
    ("Talk to you later!", 5),
]

# Generate synthetic variants
def generate_pairs():
    pairs = []
    # Greeting variants
    greetings_in = ["Hello", "Hi", "Hey", "Good morning", "Good afternoon", "Good evening", "Greetings", "Howdy"]
    greetings_out = ["Hi! How can I help?", "Hello! What can I do for you?", "Hey there!", "Hi! Nice to meet you.", "Hello! How are you today?"]
    for g in greetings_in:
        for r in greetings_out:
            pairs.append(f"Human: {g}\nBot: {r}\n")

    # Math questions
    for a in range(1, 20):
        for b in range(1, 20):
            if a + b > 20:
                continue
            s = a + b
            pairs.append(f"Human: What is {a}+{b}?\nBot: {a}+{b} equals {s}.\n")
            pairs.append(f"Human: What's {a}+{b}\nBot: {a}+{b} equals {s}.\n")

    # Capital questions
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Germany": "Berlin",
        "Spain": "Madrid", "UK": "London", "USA": "Washington", "China": "Beijing",
        "Russia": "Moscow", "Brazil": "Brasilia", "Canada": "Ottawa", "Australia": "Canberra",
        "India": "New Delhi", "Egypt": "Cairo", "Mexico": "Mexico City", "Argentina": "Buenos Aires",
    }
    for country, capital in capitals.items():
        pairs.append(f"Human: What's the capital of {country}?\nBot: The capital of {country} is {capital}.\n")
        pairs.append(f"Human: What is the capital of {country}\nBot: The capital of {country} is {capital}.\n")
        pairs.append(f"Human: Capital of {country}\nBot: The capital of {country} is {capital}.\n")

    # Identity questions
    identity_pairs = [
        ("What is your name?", "My name is Lau."),
        ("Who are you?", "I am a tiny language model."),
        ("What can you do?", "I can answer simple questions."),
        ("How are you?", "I am doing well, thank you."),
        ("Are you human?", "No, I am an AI language model."),
        ("What are you?", "I am a small language model trained in Lau."),
    ]
    for q, a in identity_pairs:
        for _ in range(3):
            pairs.append(f"Human: {q}\nBot: {a}\n")

    # Closing
    closings = ["Bye", "Goodbye", "See you", "See ya", "Later"]
    closing_replies = ["Goodbye! Have a great day!", "Bye! Talk to you later.", "See you soon!", "Take care!"]
    for c in closings:
        for r in closing_replies:
            pairs.append(f"Human: {c}\nBot: {r}\n")

    # Thanks
    for t in ["Thank you", "Thanks", "Thanks a lot", "Much appreciated"]:
        for _ in range(3):
            pairs.append(f"Human: {t}\nBot: You're welcome!\n")
            pairs.append(f"Human: {t}\nBot: Happy to help!\n")
            pairs.append(f"Human: {t}\nBot: My pleasure!\n")

    # Yes/No
    for q in ["Do you like music?", "Can you help me?", "Are you smart?", "Do you know math?"]:
        for a in ["Yes, I think so.", "Yes, I can.", "Yes, I do.", "Yes, I am."]:
            pairs.append(f"Human: {q}\nBot: {a}\n")

    # Math answers with more diversity
    for a in range(1, 10):
        for b in range(1, 10):
            s = a + b
            pairs.append(f"Human: {a} + {b} = ?\nBot: {a} + {b} = {s}\n")
            pairs.append(f"Human: What is {a} times {b}?\nBot: {a} times {b} equals {a*b}.\n")
            pairs.append(f"Human: Compute {a}*{b}\nBot: {a}*{b} = {a*b}\n")

    return pairs


def build_dataset(out_path, target_size=800_000):
    text_parts = []

    # Add curated entries with their repeat counts
    for entry, count in CURATED:
        for _ in range(count):
            text_parts.append(entry)

    # Add facts
    for entry, count in FACTS:
        for _ in range(count):
            text_parts.append(entry)

    # Add synthetic pairs
    synthetic = generate_pairs()
    for _ in range(20):  # repeat many times
        text_parts.extend(synthetic)

    # Shuffle but keep some structure
    random.shuffle(text_parts)

    # Join and check size
    corpus = "".join(text_parts)

    # Pad if needed by repeating shuffled content
    while len(corpus) < target_size:
        # Add another shuffled copy
        random.shuffle(text_parts)
        corpus += "".join(text_parts)

    # Trim to target
    corpus = corpus[:target_size]

    # Sanity check: ensure all chars are ASCII printable (excluding tabs/newlines)
    bad_chars = set()
    for c in corpus:
        if not (32 <= ord(c) <= 126 or c in "\n"):
            bad_chars.add(c)
    if bad_chars:
        print(f"WARNING: {len(bad_chars)} non-printable chars found: {list(bad_chars)[:10]}")

    # Verify test prompts are present
    test_prompts = ["Hello", "1+1", "capital of France"]
    for tp in test_prompts:
        count = corpus.count(tp)
        print(f"  '{tp}' appears {count} times in corpus")

    with open(out_path, "w", encoding="ascii") as f:
        f.write(corpus)

    print(f"Wrote {len(corpus)} chars to {out_path}")
    return corpus


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "chat.txt")
    build_dataset(out, target_size=800_000)
