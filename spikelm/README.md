# spikelm — a spiking language model that can actually speak

The engineering track that follows the reservoir campaign
([why we changed tack](https://soundlark.studio/changing-tack.html)).
Goal: a small directly-trained spiking LM whose free-running rollouts are
recognizably language. Built on others' work — SpikeGPT (spiking RWKV +
surrogate gradients), BICLab's train-soft/infer-sparse lesson, CTM's
synchronization-as-representation as a later research arm — not reinvented.

**Success criterion (from day one): coherent, non-degenerate free-running
output on fixed prompts.** Teacher-forced loss is a training signal, not
the headline.

## Milestones (gated — see changing-tack page)

0. `rwkv-mini` conventional baseline on TinyStories (~13M params, 6 blocks,
   d=384, 4k BPE, ctx 256) → gate: coherent baseline rollouts
1. same model, spiking activations + surrogate gradients + firing-rate
   regularization → gate: rollouts within sight of milestone 0
2. integer-spike annealing (train soft, infer sparse) → gate: sparsity
   without rollout collapse
3. Morpho phase-wise growth/pruning of sparse structure; CTM-style
   synchronization readout as a research arm → gate: beats fixed
   architecture at matched params

## Quickstart

```bash
cd spikelm
python bench.py                 # MPS/CPU throughput reality check FIRST
python -m spikelm.data          # download TinyStories subset, train 4k BPE, tokenize
python -m spikelm.train         # milestone-0 baseline (checkpoints + rollouts to runs/)
python test_smoke.py            # tiny overfit test (~1 min, CPU)
```

Throughput honesty: training-time estimates are made by `bench.py` on this
machine, not assumed. MPS operator fallbacks are real; the wkv scan is a
Python-level loop until measured to be the bottleneck.

## Layout

```
spikelm/tokenizer.py   pure-python 4k BPE (no deps), save/load JSON
spikelm/data.py        TinyStories fetch (HTTP range), uint16 memmap, batches
spikelm/model.py       RWKV-mini: token-shift time-mix + channel-mix, stable wkv scan
spikelm/spiking.py     surrogate-gradient spike units + rate regularizer (milestone 1)
spikelm/evaluate.py    fixed-prompt rollouts, repetition/distinct-n/entropy metrics
spikelm/train.py       AdamW + cosine, checkpoints, val loss, rollouts every eval
bench.py               tokens/sec fwd+bwd + peak memory at the target config
test_smoke.py          shapes + tiny-overfit sanity (CPU, ~1 min)
```

Everything deterministic per seed; runs/ holds checkpoints, JSONL logs and
rollout samples. The browser SNN lab (`../snn/`) is untouched — separate track.

## Design notes from the literature (2026-08-11)

**SpikeDecoder** (Beger et al., TUM — fully-spiking GPT decoder, staged
"spike degrees"): parameterized LIFs (per-unit learnable thresholds/τ)
were their largest quality lever (+11pp fully-spiking) → our `SpikeAct`
now has per-block, per-channel learnable thresholds. Fully-spiking
residuals were catastrophic (→18–42%) → we keep float residuals through
milestone 2. Spiking the embedding cost nothing; spiking the mixing is
what hurts, and fewer/wider mixing channels help. Their gaps — train-set
accuracy as the metric, no sampling temperature, no degeneration
measurement — are exactly what our rollout harness covers; keep it.

**PSAC** (Nazari & Amiri, Sci Rep 2025 — STDP + spiking actor-critic):
their global learning factor is a reward-prediction error from a spiking
critic (value-baselined), not raw reward. Relevant here if we ever
fine-tune on non-differentiable rollout metrics (use an RPE baseline),
and retro-relevant to the reservoir campaign's v16d null (raw ±reward —
the untested variant is critic-baselined RPE).
