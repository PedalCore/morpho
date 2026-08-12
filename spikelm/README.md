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

## Results log

**Milestone 0 (baseline, step 5500):** val ppl 6.42, rep4 0.016, distinct2 0.872.
Coherent on-prompt TinyStories; learned the document separator and starts
fresh stories after an ending. ~2000 tok/s, 2.9GB, M3/MPS.

**Milestone 1 (spiking, matched 5500 steps):** val ppl 6.80 (+6% vs
baseline), rep4 0.010 and distinct2 0.882 (both BETTER than baseline),
firing rates per block 4/5/6/8/13/20% (mean 9.2%), 2158 tok/s. Curves
crossed 3× during training — the gap is within noise-of-seed territory.
Caveats: one seed, one budget, spike degree 1–2 (float residuals,
embeddings, wkv, head). Write-up: soundlark.studio/spikelm.html

**Milestone 2 (in progress):** anneal 4 integer levels → 2 → 1 (binary),
measuring ppl + rollout quality + firing rate at each level.

**Milestone 2 (annealing ladder, 500 fine-tune steps per rung):**
4 levels ppl 6.80 @9.2% firing · 2 levels 6.42 @9.8% · BINARY 6.62 @10.0%.
Quality survived binarization (~3% behind float baseline) and firing rate
did NOT inflate to compensate. Op-count proxy: ~30% of a block's matrix
energy, no wall-clock or memory gain on GPU/MPS (dense kernels, float
storage). Caveats: single seed, extra fine-tune steps vs comparison arms,
annealed-not-from-scratch.

**CTM arms (milestone 4, both null):** neuron-level temporal models (per-unit
history weights) tracked the plain spiking arm across 5 checkpoints
(21.28/14.95/10.97/10.44/8.39 vs 21.47/14.71/10.90/10.54/8.28) — plausibly
because RWKV already gives every channel its own learnable time-decay, unlike
CTM's feed-forward setting. The synchronization readout (512 pairwise
co-activation traces) led by 0.77 at step 500 and was level by step 1000.
Both arms started as exact no-ops by construction (zero-init projections),
so neither null is explained by the mechanism failing to engage.

**Fully-spiking channel-mix + channel-axis LIF (in progress):** signed input
spikes make BOTH channel-mix matmuls multiply-free (~54% of model arithmetic,
up from ~27%). Gap to plain spiking closing over training: +16.4% (500),
+10.1% (1000), +8.7% (1500), +6.4% (2000), +7.8% (3000). Confounded between
the two mechanisms — split them before drawing conclusions.

**Fixed-point wkv sweep:** perplexity 6.244 in every format (Q8.8/Q6.10/Q10.6,
LUT 32/64, interp on/off, exact vs restoring division) despite a 12x spread in
wkv-output RMS error. Published on soundlark.studio/wkv-cell.html.

**Export:** export/wkv-atlas.json (exact per-block decays, committed),
site/rwkv-export/model-int8.bin (14.9MB, committed; validated 6.4224 ->
6.4215, -0.01%), also mirrored as release spikelm-v0.1. Runs live in the
browser at soundlark.studio/rwkv-live.html.
