"""Rollout evaluation — the headline metrics, ported from the reservoir
campaign's hard lessons: coherence and degeneration, not teacher-forced
score. Every checkpoint gets fixed-prompt rollouts and these numbers.
"""

import torch

FIXED_PROMPTS = [
    "Once upon a time, there was a little girl named",
    "Tom and his dog went to the park. Suddenly,",
    "The old man opened the box and found",
    "Lily wanted to bake a cake for her mom, so she",
    "One day, a small bird fell out of its nest.",
]


@torch.no_grad()
def generate(model, tok, prompt, max_new=256, temperature=0.8, top_k=40, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.cfg.ctx :])
        logits = logits[0, -1] / max(temperature, 1e-4)
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=g).item()
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(idx[0].tolist())


def degeneration_metrics(text):
    """Repetition + diversity numbers. High rep4 / low distinct = loops."""
    words = text.split()
    out = {}
    for n in (3, 4):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            out[f"rep{n}"] = 0.0
            continue
        seen, repeated = set(), 0
        for gm in grams:
            if gm in seen:
                repeated += 1
            seen.add(gm)
        out[f"rep{n}"] = repeated / len(grams)
    for n in (1, 2):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        out[f"distinct{n}"] = len(set(grams)) / max(1, len(grams))
    return out


def evaluate_rollouts(model, tok, device, max_new=256):
    model.eval()
    results = []
    for i, prompt in enumerate(FIXED_PROMPTS):
        text = generate(model, tok, prompt, max_new=max_new, device=device, seed=1000 + i)
        completion = text[len(prompt) :]
        m = degeneration_metrics(completion)
        m["prompt"] = prompt
        m["sample"] = completion[:200]
        results.append(m)
    model.train()
    agg = {
        k: sum(r[k] for r in results) / len(results)
        for k in ("rep3", "rep4", "distinct1", "distinct2")
    }
    return agg, results
