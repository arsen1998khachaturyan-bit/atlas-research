from __future__ import annotations

import re
from collections import Counter

import numpy as np

# Self-authored word banks and templates -- no external dataset or
# copyrighted text involved. Combinatorial generation gives real English
# sentence structure and genuine sentiment semantics (not synthetic feature
# vectors) while staying fully reproducible from a seed.
SUBJECTS = [
    "The movie", "This restaurant", "The service", "Her performance",
    "The weather today", "This book", "The new phone", "Our trip",
    "The concert", "His speech", "The coffee here", "This app",
    "The hotel room", "Their support team", "The traffic today",
    "This song", "The lecture", "My weekend", "The garden",
    "The city center", "This laptop", "The soup", "Her painting",
    "The flight", "This neighborhood", "The customer service",
    "His new haircut", "The museum exhibit", "This software update",
    "The birthday party",
]

POSITIVE_ADJECTIVES = [
    "wonderful", "fantastic", "amazing", "delightful", "excellent",
    "brilliant", "impressive", "charming", "outstanding", "lovely",
    "superb", "terrific", "refreshing", "remarkable", "enjoyable",
]

NEGATIVE_ADJECTIVES = [
    "terrible", "awful", "disappointing", "dreadful", "boring",
    "annoying", "horrible", "mediocre", "frustrating", "unpleasant",
    "poor", "dull", "tedious", "underwhelming", "irritating",
]

TEMPLATES = [
    "{subject} was {adj}.",
    "{subject} is absolutely {adj}.",
    "I thought {subject_lower} was {adj}.",
    "Everyone said {subject_lower} was {adj}.",
    "{subject} turned out to be {adj}.",
    "Honestly, {subject_lower} felt {adj}.",
    "{subject} seemed quite {adj} to me.",
]


def _lower_first(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def generate_sentiment_dataset(n_examples: int, seed: int = 0) -> list[tuple[str, int]]:
    """label 1 = positive, 0 = negative. Deduplicated, class-balanced by
    construction (each draw independently picks a label then a matching
    adjective)."""
    import random

    rng = random.Random(seed)
    examples: list[tuple[str, int]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n_examples * 50

    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1
        subject = rng.choice(SUBJECTS)
        label = rng.choice([0, 1])
        adj = rng.choice(NEGATIVE_ADJECTIVES if label == 0 else POSITIVE_ADJECTIVES)
        template = rng.choice(TEMPLATES)
        text = template.format(subject=subject, subject_lower=_lower_first(subject), adj=adj)

        if text in seen:
            continue
        seen.add(text)
        examples.append((text, label))

    rng.shuffle(examples)
    return examples


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]
    ids = ids[:max_len]
    ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def make_sentiment_arrays(
    examples: list[tuple[str, int]],
    vocab: dict[str, int],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([encode(text, vocab, max_len) for text, _label in examples], dtype=np.int64)
    y = np.array([label for _text, label in examples], dtype=np.int64)
    return x, y
