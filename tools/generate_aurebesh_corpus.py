#!/usr/bin/env python3
import argparse
import os
import random
import re
import sys
from collections import Counter
from typing import List, Sequence

try:
    from wordfreq import top_n_list, zipf_frequency
except ImportError:  # pragma: no cover
    top_n_list = None  # type: ignore
    zipf_frequency = None  # type: ignore

ALLOWED_CHARS_PATTERN = re.compile(r"[^A-Z0-9 .' ]+")
TOKEN_CLEAN_PATTERN = re.compile(r"[^A-Z0-9']+")  # inside tokens keep apostrophe

DEFAULT_LEN_DIST = "1:0.5,2:0.35,3:0.10,4:0.05"

NUMERIC_INJECTION_PATTERNS = [
    # Weighted via duplication: increase 1-digit, reduce 3-digit occurrence.
    # 1-digit only
    lambda rng: [str(rng.randint(1, 9))],
    # Mostly 1-digit (1-12)
    lambda rng: [str(rng.randint(1, 12))],
    # Mix 1-2 digit (duplicate for higher weight)
    lambda rng: [str(rng.randint(1, 99))],
    lambda rng: [str(rng.randint(1, 99))],
    # Another mostly 1-digit range (duplicate 1-12 for weight)
    lambda rng: [str(rng.randint(1, 12))],
    # 3-digit (kept single to reduce proportion)
    lambda rng: [str(rng.randint(100, 999))],
]

# Patterns that combine with existing tokens post-generation
POSITIONAL_NUMBER_PATTERNS = [
    # (placement, transform_function)
    ("append", lambda tokens, rng: tokens + [str(rng.randint(1, 99))]),
    ("prepend", lambda tokens, rng: [str(rng.randint(1, 99))] + tokens),
]


def parse_len_dist(spec: str):
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    kv = []
    for p in parts:
        if ':' not in p:
            raise ValueError(f"Invalid len-dist part: {p}")
        k, v = p.split(':', 1)
        k_int = int(k)
        if k_int < 1:
            raise ValueError("len-dist key must be >=1")
        prob = float(v)
        if prob < 0:
            raise ValueError("len-dist probability must be >=0")
        kv.append((k_int, prob))
    total = sum(p for _, p in kv)
    if total <= 0:
        raise ValueError("len-dist total probability must be >0")
    # normalize
    kv = [(k, p / total) for k, p in kv]
    return kv


def weighted_choice(rng: random.Random, items: Sequence[str], weights: Sequence[float]):
    # cumulative distribution
    r = rng.random()
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]


def load_starwars_vocab(path: str) -> List[str]:
    vocab = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            w = normalize_token(w)
            if w:
                vocab.append(w)
    # deduplicate preserving order
    seen = set()
    unique = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique


def normalize_token(token: str) -> str:
    token = token.upper()
    token = TOKEN_CLEAN_PATTERN.sub('', token)
    return token


def sanitize_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line.strip())
    line = ALLOWED_CHARS_PATTERN.sub('', line)
    return line.strip()


def build_english_vocab(limit: int = 10000):
    if top_n_list is None:
        raise RuntimeError("wordfreq is not installed. Please pip install wordfreq")
    words = top_n_list('en', limit)
    # normalize and filter empties
    out = []
    for w in words:
        nw = normalize_token(w)
        if nw:
            out.append(nw)
    return out


def compute_word_weights(words: List[str]):
    if zipf_frequency is None:
        # uniform fallback
        return [1.0 for _ in words]
    weights = []
    for w in words:
        freq = zipf_frequency(w.lower(), 'en')  # float; could be 0
        weights.append(max(freq, 0.0001))
    # normalize
    total = sum(weights)
    return [w / total for w in weights]


def sample_words(rng: random.Random, base_words: List[str], base_weights: List[float], k: int) -> List[str]:
    # We assume base_weights normalized
    # Use random.choices if lengths big for speed but incorporate weights
    return rng.choices(base_words, weights=base_weights, k=k)


def maybe_inject_numbers(rng: random.Random, tokens: List[str], inject_prob: float, max_tokens: int = 4) -> List[str]:
    if rng.random() >= inject_prob:
        return tokens
    # If already at or above max_tokens, perform in-place replacement of one token with a number (length-stable)
    if len(tokens) >= max_tokens:
        idx = rng.randrange(len(tokens))
        # single numeric token from weighted patterns favoring 1-2 digits
        num_token = rng.choice(NUMERIC_INJECTION_PATTERNS)(rng)[0]
        tokens[idx] = num_token
        return tokens
    # choose positional pattern 50% else simple addition at random position
    if rng.random() < 0.5:
        placement, func = rng.choice(POSITIONAL_NUMBER_PATTERNS)
        new_tokens = func(tokens, rng)
        # Guard: if this exceeded max_tokens, fallback to replacement strategy
        if len(new_tokens) > max_tokens:
            idx = rng.randrange(len(tokens))
            num_token = rng.choice(NUMERIC_INJECTION_PATTERNS)(rng)[0]
            tokens[idx] = num_token
            return tokens
        return new_tokens
    else:
        num_tokens = rng.choice(NUMERIC_INJECTION_PATTERNS)(rng)
        # Insert but clamp
        allowable_inserts = max(0, max_tokens - len(tokens))
        if allowable_inserts <= 0:
            idx = rng.randrange(len(tokens))
            tokens[idx] = num_tokens[0]
            return tokens
        # Only take as many as allowed (always 1 in current patterns)
        num_tokens = num_tokens[:allowable_inserts]
        pos = rng.randint(0, len(tokens))
        return tokens[:pos] + num_tokens + tokens[pos:]


def ensure_sw_inclusion(rng: random.Random, tokens: List[str], sw_vocab: List[str]):
    # Replace a random token with a SW token if none present
    sw_set = set(sw_vocab)
    if any(t in sw_set for t in tokens):
        return tokens
    if not sw_vocab:
        return tokens
    idx = rng.randrange(len(tokens))
    tokens[idx] = rng.choice(sw_vocab)
    return tokens


def generate_corpus(size: int, len_dist_spec: str, p_sw: float, inject_punct: float, sw_vocab_path: str, seed: int, output: str):
    rng = random.Random(seed)
    len_dist = parse_len_dist(len_dist_spec)
    length_values, length_probs = zip(*len_dist)

    sw_vocab = load_starwars_vocab(sw_vocab_path)

    eng_vocab = build_english_vocab(10000)
    eng_weights = compute_word_weights(eng_vocab)

    out_lines: List[str] = []
    attempts = 0
    max_attempts = size * 10

    while len(out_lines) < size and attempts < max_attempts:
        attempts += 1
        # choose length
        k = weighted_choice(rng, length_values, length_probs)
        # sample base words
        tokens = sample_words(rng, eng_vocab, eng_weights, k)

        # Determine if this line should include SW vocab
        if rng.random() < p_sw:
            tokens = ensure_sw_inclusion(rng, tokens, sw_vocab)
        # Additional chance: small probability to make whole line SW if selected
        elif rng.random() < p_sw * 0.2 and sw_vocab:
            k2 = k
            tokens = rng.choices(sw_vocab, k=k2)

        tokens = maybe_inject_numbers(rng, tokens, inject_punct, max_tokens=4)

        line = ' '.join(tokens)
        line = sanitize_line(line)
        if not line:
            continue
        if len(line.split()) < 1:
            continue
        out_lines.append(line)

    if len(out_lines) < size:
        print(f"Warning: generated only {len(out_lines)} lines out of requested {size}", file=sys.stderr)

    # write output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for line in out_lines[:size]:
            f.write(line + '\n')

    # basic stats
    avg_len = sum(len(l) for l in out_lines) / max(1, len(out_lines))
    dist_counter = Counter(len(l.split()) for l in out_lines)
    print(f"Generated {len(out_lines)} lines. Avg chars: {avg_len:.2f}")
    print("Line length (words) distribution:")
    for n, c in sorted(dist_counter.items()):
        print(f"  {n}: {c}")


def main():
    parser = argparse.ArgumentParser(description="Generate Aurebesh-ready corpus for SynthTIGER.")
    parser.add_argument('--size', type=int, required=True, help='Number of lines to generate.')
    parser.add_argument('--len-dist', default=DEFAULT_LEN_DIST, help='Length distribution spec k:prob,... e.g. "1:0.5,2:0.3,3:0.1,4:0.1"')
    parser.add_argument('--p-sw', type=float, default=0.05, help='Probability a line contains at least one Star Wars vocab token.')
    parser.add_argument('--inject-punct', type=float, default=0.05, help='Probability to inject numeric token(s).')
    parser.add_argument('--sw-vocab', default='aurebesh/vocab/starwars-vocab.txt', help='Path to Star Wars vocab file.')
    parser.add_argument('--output', default='aurebesh/corpus.txt', help='Output corpus path.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    generate_corpus(
        size=args.size,
        len_dist_spec=args.len_dist,
        p_sw=args.p_sw,
        inject_punct=args.inject_punct,
        sw_vocab_path=args.sw_vocab,
        seed=args.seed,
        output=args.output,
    )


if __name__ == '__main__':
    main()
