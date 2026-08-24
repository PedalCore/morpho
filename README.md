# whitebox-lm

White-box transformer language models, derived from objectives and held to
preregistered tests. A research campaign built on CRATE (Yu et al.,
arXiv:2306.01129), extended with spiking proximal operators, a
constant-state counter attention (CRSA), an overcomplete sparse-coding
feature block, and a gate-counted hardware path.

Small scale, deliberately: 5–26M params, TinyStories, one controlled seed
per claim until replicated. The contribution is findings plus method, not
a leaderboard entry.

## Headline results

**Best constant-state model** (matched 20k-step pair, one seed):

| model | params | val ppl @20k |
|---|---|---|
| CRSA + overcomplete dictionary | 14.0M | **8.40** |
| plain CRSA d=672 | 13.8M | 9.72 |

Neither had plateaued at schedule end. A reduced-LR extension of the
dictionary model reached 8.08 (28.5k steps, censored). The M4 program
(binding/retrieval; see M4.md and PAPER-M4.md) then produced the
current best: **slots-lm-v2 at 6.48** (20k from scratch, 17.1M, campaign close —
counters + overcomplete dictionary + owner-routed slots with a fully
learned conv write path + grouped basis), 0.44 above a true untied
QKV+MLP transformer reference at matched params (6.04 on its own 10k
schedule; matched-schedule control owed) — with ~194 KB constant
recurrent state versus a growing KV cache.

**The operator ladder** (matched d=448 + identical 4d MLP, 3,000-step
screens, one seed):

| attention operator | val ppl | nats vs leaky |
|---|---|---|
| softmax (pairwise) | 12.54 | −0.292 |
| CRSA, dyadic-leaky counters | 16.79 | — |
| CRSA-uniform (prefix measure ablation) | 24.57 | +0.381 |
| literal causal TSSA (verified reproduction) | 25.43 | +0.415 |

At matched width, feature block, data, seed, and schedule, dyadic-leaky
CRSA improves by 0.381 nats/token over its uniform-measure ablation and by
0.415 nats/token over a numerically verified reproduction of causal TSSA
(equation-level equivalence 6e-8), despite TSSA carrying 2.4M additional
parameters. This identifies **multiscale exponential temporal weighting**
— not routing simplification or parameter count — as the principal source
of CRSA's advantage over cumulative token-statistics attention in this
setting. Softmax retains a substantial retrieval advantage: CRSA trades
retrieval capacity for bounded recurrent state.

**The factorial** (overcomplete dictionary, fixed d=448): the proximal
nonlinearity alone is worth −0.080 nats at zero added parameters; 4×
width with the prox disabled is worth nothing (+0.004, as the rank
argument demands); the interaction is −0.046 nats. Overcompleteness has
no value as a linear factorization; it becomes useful only when the
threshold partitions inputs into selectively activated feature regions.

**The derivation–execution gap**: under aligned measurement, every
softmax-attention variant was repurposed by training — attention learned
to *expand* the coding-rate term it was derived to compress (12/12
layers, confirmed by directional derivatives and α-sweeps). The counter
operator is the only one whose trained dynamics match its derived sign
throughout training. Derivation supplies a falsifiable mechanistic
hypothesis, not a guarantee.

**Preregistered probes** (locked before CRSA existed): perfect induction
(1.00) at every delay up to 89 tokens from 384 counters/layer, tied with
full attention; the measured cost is selective retention under
distractors (0.56 vs 0.91) — the operator forgets by *collision*, not by
time. This deficit predicted the ladder's softmax result before any
perplexity surfaced it, and appears in generated text as referent drift.

**Hardware**: one CRSA counter coordinate = 58–61 gates + 14 registers
(dyadic decay = shift; price 1/(1+c) = 2-comparator staircase); a
16-wide bank places and routes at 690 LCs, 103.7 MHz on iCE40. No
exponential, no divider, no softmax anywhere in the datapath.

## Repo map

| file | contents |
|---|---|
| `model.py` | Config; MSSA/ISTA (CRATE); SpikeProx/SignedProx; CRSA + uniform ablation + literal TSSA; overcomplete dictionary blocks; MLP control; per-layer instrumentation (aligned ΔR^c, sparsity, frame/coherence/dead-atom) |
| `train.py` | trainer (AdamW, cosine, matched to the spikelm baseline); all arms reachable by flag; logs `runs/<name>/log.jsonl` + checkpoint |
| `probes.py`, `probe_train.py` | locked probe suite: 4 tasks × delays {6..89}, fixed seeds, preregistered predictions |
| `calibrate.py`, `autopsy.py`, `hard_autopsy.py` | spike conversion calibration; derivation–execution gap instruments (g_dir, α-sweeps, Gram decomposition) |
| `M2.md` | spike-driven weight paths: prox propositions, results, scope freeze, the gap finding |
| `M3.md` | CRSA: derivation via exponentially weighted measures, ToST prior-art position, validation battery, judgment protocol |
| `DICTIONARY.md` | overcomplete fork: formalization, factorial, MLP three-way + literal-TSSA close, plateau protocol, claim freezes |
| `PROBES.md` | preregistration + results scorecard |
| `PLFP.md` | forward-only learning formalization (not yet run) |

## Running

```
python -m whitebox.train --steps 20000 --width 448 --crsa \
    --dict 4 --dict-local --m2 b --m2-identity --name my-run
```

Flags select arms: `--crsa` counters, `--tost` uniform ablation,
`--tssa-lit` literal TSSA, `--mlp` transformer MLP control, `--dict N
--dict-local` overcomplete dictionary. Data/tokenizer come from the
spikelm sibling repo (path in `train.py`); checkpoints and logs are
machine-local and untracked.

## Method, briefly

Formalize before training. Lock probes and predictions before the
operator exists. Screens (3k steps) eliminate catastrophes only —
verdicts require plateaus (<0.5% over 2–3k steps after an LR reduction).
Decompositions in nats. One seed until replicated, and every claim
carries its caveats. Corrections are published as findings, not patched:
this repo's history includes a misaligned metric, a naming collision
with prior art (ToST/TSSA, arXiv:2412.17810 — cited as CRSA's direct
ancestor), and the reinterpretation of an early headline once a real
feature block entered the comparison.

## Open work

Replication seeds; two-size scaling law; true plateaus on longer
schedules; spike conversion of the best CRSA model via the calibration
protocol; the PLFP forward-only ladder; hardware costing of the full
model; a representative memory between counters and a full cache
(the collision repair).
