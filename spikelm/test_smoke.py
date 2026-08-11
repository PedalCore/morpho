"""Smoke tests: shapes, tokenizer round-trip, tiny-model overfit. CPU, ~1 min."""

import torch

from spikelm.model import Config, RWKVMini
from spikelm.spiking import SpikeAct, rate_loss
from spikelm.tokenizer import BPETokenizer
from spikelm.evaluate import degeneration_metrics


def test_tokenizer_roundtrip():
    text = "Once upon a time, there was a little girl named Lily. She had 3 cats!\n"
    tok = BPETokenizer.train(text * 50, vocab_size=400)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert all(0 <= i < tok.vocab_size for i in ids)
    print(f"tokenizer roundtrip ok ({len(ids)} tokens, vocab {tok.vocab_size})")


def test_model_shapes(spiking):
    cfg = Config(vocab_size=100, n_layer=2, n_embd=64, ctx=32, spiking=spiking)
    m = RWKVMini(cfg)
    x = torch.randint(0, 100, (3, 32))
    logits, loss = m(x, x)
    assert logits.shape == (3, 32, 100)
    assert loss.isfinite()
    loss.backward()
    if spiking:
        rl = rate_loss(m)
        assert rl.isfinite()
    print(f"shapes ok (spiking={spiking}, {m.num_params()/1e3:.0f}k params)")


def test_overfit(spiking):
    torch.manual_seed(0)
    cfg = Config(vocab_size=20, n_layer=2, n_embd=64, ctx=32, spiking=spiking)
    m = RWKVMini(cfg)
    # learnable pattern: repeating sequence
    seq = torch.arange(32).remainder(7)
    x = seq.unsqueeze(0).repeat(8, 1)
    y = (x + 1).remainder(7)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(150):
        _, loss = m(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.35, f"no learning: {first:.3f} -> {loss.item():.3f}"
    print(f"overfit ok (spiking={spiking}): loss {first:.2f} → {loss.item():.3f}")


def test_metrics():
    loopy = "the in the come " * 30
    fresh = "once upon a time there was a little girl who went to the park and saw a dog"
    assert degeneration_metrics(loopy)["rep4"] > 0.8
    assert degeneration_metrics(fresh)["rep4"] == 0.0
    print("degeneration metrics ok")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_model_shapes(False)
    test_model_shapes(True)
    test_overfit(False)
    test_overfit(True)
    test_metrics()
    print("\nall smoke tests passed")
