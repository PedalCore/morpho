"""Milestone-0 training loop: RWKV-mini on TinyStories.

python -m spikelm.train [--spiking] [--steps N]
Checkpoints, JSONL logs, and fixed-prompt rollouts land in runs/<name>/.
Resume is automatic if a checkpoint exists.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import get_tokenizer, load_split, get_batch, prepare
from .evaluate import evaluate_rollouts
from .model import Config, RWKVMini


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiking", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--rate-lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    prepare()
    tok = get_tokenizer()
    train_data = load_split("train")
    valid_data = load_split("valid")

    cfg = Config(vocab_size=tok.vocab_size, spiking=args.spiking)
    model = RWKVMini(cfg).to(device)
    name = f"{'spike' if args.spiking else 'base'}-rwkv-d{cfg.n_embd}L{cfg.n_layer}-s{args.seed}"
    run_dir = os.path.join(os.path.dirname(__file__), "..", "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    log_path = os.path.join(run_dir, "log.jsonl")
    print(f"{name}: {model.num_params()/1e6:.1f}M params on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01)
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        rng = np.random.default_rng(args.seed + start_step)
        print(f"resumed at step {start_step}")

    warmup = 200

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    t0 = time.time()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, args.batch, cfg.ctx, rng, device)
        _, loss = model(x, y)
        total = loss
        if args.spiking:
            from .spiking import rate_loss

            total = loss + args.rate_lambda * rate_loss(model).to(loss.device)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            tps = args.batch * cfg.ctx * (step - start_step + 1) / (time.time() - t0)
            print(f"step {step}: loss {loss.item():.3f} · {tps:.0f} tok/s")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(valid_data, args.batch, cfg.ctx, rng, device)
                _, vloss = model(vx, vy)
            model.train()
            agg, rollouts = evaluate_rollouts(model, tok, device)
            rec = {
                "step": step + 1,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vloss.item(), 4),
                "val_ppl": round(math.exp(vloss.item()), 2),
                **{k: round(v, 4) for k, v in agg.items()},
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            with open(os.path.join(run_dir, f"rollouts-{step + 1}.json"), "w") as f:
                json.dump(rollouts, f, indent=1)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, ckpt_path)
            print(f"  eval: val_loss {rec['val_loss']} ppl {rec['val_ppl']} rep4 {rec['rep4']} "
                  f"distinct2 {rec['distinct2']}")
            print(f"  sample: {rollouts[0]['sample'][:120]!r}")


if __name__ == "__main__":
    main()
