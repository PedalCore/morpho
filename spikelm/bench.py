"""Throughput reality check — run BEFORE believing any training-time claim.

Measures fwd+bwd tokens/sec and peak memory at the milestone-0 config on
this machine (MPS/CUDA/CPU), prints an honest ETA for a 100M-token run.
"""

import time

import torch

from spikelm.model import Config, RWKVMini
from spikelm.train import pick_device


def bench(spiking=False, batch=16, steps=12, warmup=3):
    device = pick_device()
    cfg = Config(spiking=spiking)
    model = RWKVMini(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx), device=device)
    for i in range(warmup):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for i in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    tps = batch * cfg.ctx * steps / dt
    mem = ""
    if device == "mps":
        mem = f" · peak mem {torch.mps.driver_allocated_memory() / 1e9:.1f}GB"
    tag = "spiking" if spiking else "baseline"
    print(f"{tag}: {model.num_params()/1e6:.1f}M params · {device} · "
          f"{tps:.0f} tok/s fwd+bwd{mem}")
    print(f"  100M tokens ≈ {1e8 / tps / 3600:.1f} h · 20k steps × b{batch}×{cfg.ctx} "
          f"≈ {20000 * batch * cfg.ctx / tps / 3600:.1f} h")
    return tps


if __name__ == "__main__":
    bench(spiking=False)
    bench(spiking=True)
