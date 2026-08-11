"""TinyStories data: HTTP-range download, BPE tokenization to uint16 memmaps.

python -m spikelm.data  → downloads (train subset + full valid), trains the
4k BPE on a sample, tokenizes both splits to .bin. All cached in data/.
"""

import os
import urllib.request

import numpy as np

from .tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_BYTES = 200 * 1024 * 1024  # first 200MB of train — plenty for ~13M params
BPE_SAMPLE_BYTES = 4 * 1024 * 1024
VOCAB_SIZE = 4096


def _fetch(name, out, max_bytes=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, out)
    if os.path.exists(path):
        return path
    req = urllib.request.Request(f"{BASE}/{name}")
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    print(f"downloading {name}{f' (first {max_bytes >> 20}MB)' if max_bytes else ''}…")
    with urllib.request.urlopen(req) as r, open(path + ".tmp", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.rename(path + ".tmp", path)
    return path


def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    # a range-cut file may end mid-utf8-char / mid-story: trim to last newline
    text = raw.decode("utf-8", errors="ignore")
    return text[: text.rfind("\n") + 1]


def get_tokenizer():
    tok_path = os.path.join(DATA_DIR, f"bpe{VOCAB_SIZE}.json")
    if os.path.exists(tok_path):
        return BPETokenizer.load(tok_path)
    train_path = _fetch("TinyStories-train.txt", "train.txt", TRAIN_BYTES)
    sample = _read_text(train_path)[:BPE_SAMPLE_BYTES]
    print(f"training {VOCAB_SIZE}-token BPE on {len(sample) >> 20}MB sample…")
    tok = BPETokenizer.train(sample, VOCAB_SIZE, verbose=True)
    tok.save(tok_path)
    return tok


def _tokenize_split(tok, txt_path, bin_path):
    if os.path.exists(bin_path):
        return
    text = _read_text(txt_path)
    print(f"tokenizing {os.path.basename(txt_path)} ({len(text) >> 20}MB)…")
    ids = []
    step = 1 << 22
    for i in range(0, len(text), step):
        ids.extend(tok.encode(text[i : i + step]))
        print(f"  {min(i + step, len(text)) >> 20}MB → {len(ids)} tokens")
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"  wrote {bin_path}: {len(arr)} tokens")


def prepare():
    tok = get_tokenizer()
    train_txt = _fetch("TinyStories-train.txt", "train.txt", TRAIN_BYTES)
    valid_txt = _fetch("TinyStories-valid.txt", "valid.txt")
    _tokenize_split(tok, train_txt, os.path.join(DATA_DIR, "train.bin"))
    _tokenize_split(tok, valid_txt, os.path.join(DATA_DIR, "valid.bin"))
    return tok


def load_split(split):
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size, ctx, rng, device):
    import torch

    ix = rng.integers(0, len(data) - ctx - 1, size=batch_size)
    x = np.stack([data[i : i + ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + ctx + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


if __name__ == "__main__":
    prepare()
