"""Pure-python BPE tokenizer (4k vocab, no dependencies).

GPT-style pre-tokenization (words keep their leading space), byte-level
fallback for unseen characters, greedy merge encoding with a word cache.
Deterministic; trained once on a corpus sample and saved as JSON.
"""

import json
import re
from collections import Counter

PRETOK = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")


def pretokenize(text):
    return PRETOK.findall(text)


class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or []          # id -> token string
        self.merges = merges or []        # list of (a, b) token-string pairs, in rank order
        self._tok2id = {t: i for i, t in enumerate(self.vocab)}
        self._rank = {tuple(m): r for r, m in enumerate(self.merges)}
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---------- training ----------
    @classmethod
    def train(cls, text, vocab_size=4096, verbose=False):
        words = Counter(pretokenize(text))
        # initial symbols: every char present, plus byte-fallback tokens
        base = sorted(set(ch for w in words for ch in w))
        vocab = [f"<byte:{b}>" for b in range(256)] + base
        seqs = {w: tuple(w) for w in words}
        merges = []
        while len(vocab) < vocab_size:
            pairs = Counter()
            for w, seq in seqs.items():
                f = words[w]
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += f
            if not pairs:
                break
            (a, b), _ = pairs.most_common(1)[0]
            merged = a + b
            merges.append((a, b))
            vocab.append(merged)
            for w, seq in seqs.items():
                if a not in seq:
                    continue
                out, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                seqs[w] = tuple(out)
            if verbose and len(vocab) % 512 == 0:
                print(f"  bpe vocab {len(vocab)}")
        return cls(vocab, merges)

    # ---------- encoding ----------
    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        seq = [ch if ch in self._tok2id else None for ch in word]
        parts = []
        for ch, ok in zip(word, seq):
            if ok is not None:
                parts.append(ch)
            else:
                parts.extend(f"<byte:{b}>" for b in ch.encode("utf-8"))
        seq = parts
        while len(seq) > 1:
            best, bi = None, -1
            for i in range(len(seq) - 1):
                r = self._rank.get((seq[i], seq[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if best is None:
                break
            seq = seq[:bi] + [seq[bi] + seq[bi + 1]] + seq[bi + 2:]
        ids = [self._tok2id[t] for t in seq]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        out = []
        for w in pretokenize(text):
            out.extend(self._encode_word(w))
        return out

    def decode(self, ids):
        parts = []
        pending = []
        for i in ids:
            t = self.vocab[i]
            if t.startswith("<byte:"):
                pending.append(int(t[6:-1]))
                continue
            if pending:
                parts.append(bytes(pending).decode("utf-8", errors="replace"))
                pending = []
            parts.append(t)
        if pending:
            parts.append(bytes(pending).decode("utf-8", errors="replace"))
        return "".join(parts)

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])
