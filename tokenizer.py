"""
Medical BPE Tokenizer
======================
Byte-Pair Encoding tokenizer trained on medical vocabulary.
Handles diagnosis terms, drug names, dosages, and special tokens.

Usage:
    tokenizer = MedTokenizer.train(corpus, vocab_size=8000)
    tokenizer.save("checkpoints/tokenizer.json")

    tokenizer = MedTokenizer.load("checkpoints/tokenizer.json")
    ids = tokenizer.encode("[DIAG] fever cough [SEP]")
    text = tokenizer.decode(ids)
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Special tokens
# ─────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "[DIAG]": 4,    # Start of diagnosis section
    "[SEP]": 5,     # Separator between diagnosis and prescription
    "[PRESC]": 6,   # Start of prescription section
    "[DRUG]": 7,    # Drug name marker
    "[DOSE]": 8,    # Dosage marker
    "[FREQ]": 9,    # Frequency marker (e.g., "twice daily")
    "[WARN]": 10,   # Warning marker
    "[END]": 11,    # End of prescription
}

# Common medical abbreviations to preserve
MEDICAL_ABBREVS = {
    "mg", "ml", "mcg", "iu", "bp", "hr", "rr", "spo2", "bmi",
    "prn", "qid", "tid", "bid", "od", "bd", "hs", "ac", "pc",
    "iv", "im", "sc", "po", "sl", "pr", "top", "inh",
    "wbc", "rbc", "hb", "plt", "esr", "crp", "cbc", "lfts", "rfts",
    "ecg", "echo", "usg", "ct", "mri", "xray",
}


# ─────────────────────────────────────────────────────────────────────────────
# BPE Implementation
# ─────────────────────────────────────────────────────────────────────────────

class MedTokenizer:
    """
    Byte-Pair Encoding tokenizer with medical domain awareness.
    
    BPE Algorithm:
      1. Start with character-level vocabulary
      2. Count all adjacent symbol pairs
      3. Merge most frequent pair → new symbol
      4. Repeat until vocab_size reached
    """

    def __init__(self,
                 vocab: Dict[str, int],
                 merges: List[Tuple[str, str]],
                 special_tokens: Dict[str, int] = None):
        self.vocab = vocab                          # token → id
        self.id2token = {v: k for k, v in vocab.items()}
        self.merges = merges                        # ordered merge rules
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or SPECIAL_TOKENS

        # Pre-compile regex for tokenization
        # Keeps: special tokens, medical abbrevs, words, numbers, punctuation
        special_pattern = "|".join(re.escape(t) for t in sorted(
            self.special_tokens.keys(), key=len, reverse=True))
        self._pattern = re.compile(
            rf'({special_pattern}|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[^\s])',
            re.IGNORECASE
        )

    # ── Training ─────────────────────────────────────────────────────────────

    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 8000,
              min_freq: int = 2) -> "MedTokenizer":
        """
        Train BPE tokenizer on a medical corpus.
        
        Args:
            corpus:     list of text strings
            vocab_size: target vocabulary size
            min_freq:   minimum pair frequency to merge
        """
        print(f"Training BPE tokenizer (target vocab: {vocab_size})...")

        # Step 1: Initialise with characters + special tokens
        vocab = dict(SPECIAL_TOKENS)
        next_id = max(vocab.values()) + 1

        # Step 2: Build initial word vocabulary (word → char sequence)
        word_freqs: Counter = Counter()
        for text in corpus:
            words = re.findall(r'[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[^\s\w]', text.lower())
            word_freqs.update(words)

        # Represent each word as tuple of chars with end-of-word marker
        word_vocab: Dict[tuple, int] = {}
        for word, freq in word_freqs.items():
            chars = tuple(list(word) + ['</w>'])
            word_vocab[chars] = freq

        # Add all characters to vocab
        all_chars = set()
        for chars in word_vocab:
            all_chars.update(chars)
        for ch in sorted(all_chars):
            if ch not in vocab:
                vocab[ch] = next_id
                next_id += 1

        merges: List[Tuple[str, str]] = []

        # Step 3: BPE merge loop
        while len(vocab) < vocab_size:
            # Count pair frequencies
            pairs = cls._count_pairs(word_vocab)
            if not pairs:
                break

            best_pair, best_freq = max(pairs.items(), key=lambda x: x[1])
            if best_freq < min_freq:
                break

            # Merge best pair
            new_token = best_pair[0] + best_pair[1]
            if new_token not in vocab:
                vocab[new_token] = next_id
                next_id += 1
            merges.append(best_pair)

            # Update word vocabulary
            word_vocab = cls._merge_pair(word_vocab, best_pair, new_token)

            if len(vocab) % 500 == 0:
                print(f"  Vocab size: {len(vocab)}")

        print(f"Training complete. Final vocab size: {len(vocab)}, merges: {len(merges)}")
        return cls(vocab, merges, SPECIAL_TOKENS)

    @staticmethod
    def _count_pairs(word_vocab: Dict[tuple, int]) -> Counter:
        pairs: Counter = Counter()
        for symbols, freq in word_vocab.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_pair(word_vocab: Dict[tuple, int],
                    pair: Tuple[str, str],
                    new_token: str) -> Dict[tuple, int]:
        new_vocab = {}
        p0, p1 = pair
        for symbols, freq in word_vocab.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                    new_symbols.append(new_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[tuple(new_symbols)] = freq
        return new_vocab

    # ── Encoding / Decoding ──────────────────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False, max_len: Optional[int] = None) -> List[int]:
        """Convert text string to token ID list."""
        tokens = self._tokenize(text)
        ids = []

        if add_bos:
            ids.append(self.special_tokens["<bos>"])

        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.extend(self._bpe_encode(token.lower()))

        if add_eos:
            ids.append(self.special_tokens["<eos>"])

        if max_len:
            ids = ids[:max_len]
            # Pad if needed
            ids += [self.special_tokens["<pad>"]] * (max_len - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token ID list back to text."""
        tokens = []
        for i in ids:
            if i in self.id2token:
                tok = self.id2token[i]
                # Skip special tokens if requested
                if skip_special and tok in self.special_tokens:
                    if tok in ("[DIAG]", "[SEP]", "[PRESC]", "[END]"):
                        tokens.append(" " + tok + " ")
                    continue
                tokens.append(tok)
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Split text into pre-token units."""
        return self._pattern.findall(text)

    def _bpe_encode(self, word: str) -> List[int]:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) + ['</w>']

        # Apply merges in learned order
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            best = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
            if best not in self.merge_ranks:
                break
            new_tok = best[0] + best[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best:
                    new_symbols.append(new_tok)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return [self.vocab.get(s, self.special_tokens["<unk>"]) for s in symbols]

    # ── Batch Encoding ───────────────────────────────────────────────────────

    def encode_batch(self, texts: List[str], max_len: int,
                     add_bos: bool = True, add_eos: bool = True) -> List[List[int]]:
        """Encode a batch of texts with padding."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos, max_len=max_len)
                for t in texts]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "MedTokenizer":
        with open(path) as f:
            data = json.load(f)
        merges = [tuple(p) for p in data["merges"]]
        return cls(data["vocab"], merges, data["special_tokens"])

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"MedTokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ─────────────────────────────────────────────────────────────────────────────
# Quick build for testing (character-level fallback)
# ─────────────────────────────────────────────────────────────────────────────

def build_char_tokenizer(texts: List[str]) -> MedTokenizer:
    """Build a character-level tokenizer for rapid prototyping."""
    chars = set()
    for t in texts:
        chars.update(t.lower())
    vocab = dict(SPECIAL_TOKENS)
    next_id = max(vocab.values()) + 1
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
    return MedTokenizer(vocab, merges=[], special_tokens=SPECIAL_TOKENS)
