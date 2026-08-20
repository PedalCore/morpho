# whitebox — causal CRATE, and a spiking proximal step

An adaptation of **CRATE** (Yu et al., [arXiv:2306.01129](https://arxiv.org/abs/2306.01129);
causal GPT-style variant in the [JMLR version](https://arxiv.org/abs/2311.13110))
to this project's corpus, baselines, and measurement discipline — plus the
fusion that is ours: **the ISTA block's proximal step replaced by an integer
spike quantizer**.

## Why this fusion is principled, not bolted on

CRATE derives each layer from the sparse rate reduction objective:
attention = a gradient step compressing tokens toward subspaces (MSSA),
MLP = a proximal step of sparse coding (ISTA, a soft-threshold). The spike
operator `clamp(floor(v/thr), 0, L) * thr` is **also a proximal operator** —
of a sparsity penalty plus an integer-grid constraint. So a spiking MLP here
is not an approximation of the derived layer; it is a *different prox in the
same alternating scheme*, and the white-box methodology says exactly what to
verify: does each layer still reduce its coding-rate term, and does the code
stay sparse? Both are logged every 500 steps (`layer_metrics`).

Two further points of hygiene:

- **Thresholds receive gradient.** The spikelm SpikeAct never routed a
  gradient to its thresholds (all 9,216 sat frozen at init — a finding, now
  corrected here): `SpikeProx` trains them through the `n·thr` product.
- **Novelty was checked, not assumed** (2026-08-20): causal-LM CRATE exists
  (JMLR version); no spiking / integer-prox CRATE work found. The candidate
  contribution is the fusion *plus* what no one else has: gate-level cost of
  the resulting architecture counted by construction
  (`examples/rwkv/count_gates.py` methodology), since integer codes make the
  dictionary matmul multiply-free — and in CRATE the ISTA block is a larger
  share of the MACs than RWKV's channel-mix was.

## Matched-budget comparison table (fill as runs land)

Same corpus (TinyStories), tokenizer (4k BPE), optimizer recipe, and
5,500-step budget as the spikelm baselines:

| model | params | val ppl @5,500 |
|---|---|---|
| RWKV-mini (float) | ~14M | 6.42 |
| RWKV-mini spiking L4 / binary | ~14M | 6.77 / 6.39 |
| causal CRATE (d=384, L=12, tied) | 5.2M | *(training)* |
| causal CRATE + spiking prox | 5.2M | *(training)* |

CRATE blocks are ~3d² parameters against a transformer's ~12d² — the table
reports both param counts and shared step budget; neither normalization is
"the" fair one, so both are stated.

## Deviations from the strict derivation (stated, per house rules)

Pre-LN before each block (as the released CRATE code); the derived step-size
constant κ·p/(Nε²) folded into a learnable per-layer scale; causal masking
(compression restricted to past tokens — same move as the JMLR causal
variant). MSSA head aggregation is weight-tied to the input projection
(`tied=True`, the derivation's form; `--untied` ablates it).

## Run

```bash
python3 -m whitebox.train                 # M0: causal CRATE baseline
python3 -m whitebox.train --spike-prox    # M1: spiking proximal step
```

Logs: `whitebox/runs/<name>/log.jsonl` — val ppl + per-layer ΔR^c (the
compression each MSSA achieves; negative = working as derived) and ISTA
sparsity. Data comes from the spikelm checkout (machine-local path in
train.py).

## Roadmap

- **M0** causal CRATE at matched budget — the honest baseline.
- **M1** spiking prox — quality cost + white-box curves under integer codes.
- **M2** candidates, in order of ambition: threshold annealing 4→2→1 (the
  schedule that made binary RWKV *better*); spiking MSSA (signed spikes into
  the U projection); a rate-reduction *recurrence* (unrolled compression as
  a state update — CRATE×RWKV, a genuinely new architecture if it works).
- **Hardware close-out**: ISTA-block engines through the Morpho gate-count
  pipeline; the spiking-CRATE datapath number next to the RWKV ones.
