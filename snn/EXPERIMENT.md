# Experiment: Is "Morpho × SNN" a good idea?

**Date started:** 2026-08-07
**Status:** Experiments 1–5 of the SNN brief implemented in miniature, plus
LIVE-mode concepts imported from the C++/JUCE plugin briefs (v2). Findings below.

## v2 — plugin-brief concepts folded in (same day)

Feedback from first listening session: little structural/register variation,
chords landing as blocks. Fixes, all borrowed from the plugin briefs so the
browser doubles as their prototyping lab:

- **Structural pitch (plugin §12):** region depth → octave, sector + birth
  order → scale degree, fixed at birth. The grammar now recurses to *stochastic*
  depths (2–3 branching, early termination), so registers spread 4–5 octaves.
- **Region subdivision + ⑂ branch button:** leaves divide during lifetime
  (automatic when large + healthy, or manually via the branch button, which
  sprouts brand-new sibling populations one register deeper). Development
  literally climbs in pitch as structure elaborates.
- **Strum:** near-simultaneous notes share a scheduling cursor and fan out
  into fast arpeggios (adjustable spacing).
- **Walkers (plugin LIVE mode, §16–19):** weighted stochastic graph traversal
  with variation/momentum/repetition-penalty on a musical grid — a melodic
  voice distinct from the spike texture.
- **Music affects survival (plugin §36–37):** neuron energy decays each epoch,
  restored by firing AND by walker visits; pruning is energy-based and
  walker-occupied neurons are protected. The music the organism plays now
  helps decide which structures live.
- **Afferent repair:** silent regions may grow a long-range afferent from an
  active region instead of another doomed local neuron (fixes the v1 churn
  finding).

Headless check (50 epochs): octave spread grows 2–3 → 4–5; leaf regions 6 → 14–25;
5–11 region divisions per run; structural churn roughly tripled while staying
within budgets.

Later same session:
- **⑂ branch button** — manual recursive fan-out: sprouts fresh sibling
  populations (new neurons, one register deeper) off existing leaves.
- **Modulator nodes** — rare gold-diamond neurons (~5% of non-output
  excitatory, grown neurons roll the same dice). When one spikes it may move
  the key around the circle of fifths — adjacent step (up a fifth / down a
  fourth) or skip-over (two positions), either direction — gated by low
  probability + 12 s cooldown. Part of the deterministic sim, so each seed has
  a reproducible harmonic journey; some organisms never leave C, others
  wander (e.g. seed 42: C→D→A→E→A→B→…). Logged as KeyChanged events.
- **FX sends** — generated-impulse reverb, feedback delay tracking the pulse
  at a dotted interval, LFO chorus. Audio-side only.
- Scales added: phrygian dominant, harmonic minor. Structural sounds
  (birth/prune/division) got their own toggle.

All 25 tests pass.

## v3 — performance steering + STDP

Feedback: wanted tighter timing and ways to steer/vary the behaviour.

- **Quantize** — note onsets pulled toward a musical grid (strength 0–1, grid
  = pulse ÷ 1/2/3/4), composing with strum. Audio-side only.
- **Drive patterns** — input rhythm selectable: steady / euclidean / bursts /
  sparse, on an 8-slot half-pulse grid. Deterministic, sim-side.
- **Attractor steering** (plugin §23) — with "steer" on, hovering the network
  pulls walker traversal toward the cursor (exp-distance weight multiplier).
  Hand-steered runs are intentionally not seed-reproducible.
- **Walker momentum + step rate** exposed on the UI alongside variation.
- **STDP** (brief §13, toggle) — trace-based pair STDP on excitatory
  synapses: pre-before-post potentiates, post-before-pre depresses, slight
  depression bias, weights clamped [0.05, 1.1]. Deterministic; with it on,
  correlated (played) pathways strengthen over minutes — the prerequisite for
  Experiment 6's weight-to-structure rule.

All 29 tests pass.

## v4 — duet mode: human-in-the-loop training (duet.html)

The collaboration experiment: the human replaces the metronome as the
organism's environment.

- **MIDI/pads/keyboard → spike encoding** — one sensory input neuron per
  scale degree, wired *tonotopically* to excitatory neurons whose birth-fixed
  pitch sounds the same degree class. Incoming notes snap to the current
  scale+key; velocity sets burst length (1–3 spikes). Pads + computer keys
  (a–l / q–p rows) are scale-locked by construction.
- **No pulse drive** — only a whisper of background (0.25 Hz) so the organism
  "dreams" when idle. What it plays back is provoked by what you play.
- **STDP on by default** — the pathways you exercise strengthen; development
  (afferent growth, region division) reorganizes around your playing.
- **✚ reinforce** — deposits survival energy on everything that fired in the
  last ~2.5 s: a reward button for responses you like.
- **MIDI out** — the organism can play external hardware/DAW.
- Engine fix found by tests: force-fired input spikes were invisible to the
  step's returned spike list (cleared at step start); now they are included.

All 34 tests pass. Open question to explore by playing: does reinforcement +
STDP + development make its responses converge toward the player's material,
and over what timescale?

## v5 — measuring learning; Q&A becomes walker-answered

Two headless experiments (`npm run experiment:learning|dialogue`) settled the
"is it actually learning?" question with controls:

**Finding 1 — usage, not sequences** (`experiments/learning.mjs`, motif
completion probe with STDP-off and scrambled-motif controls): trained Δlift
0.171 ≈ scrambled Δlift 0.186, both ≫ stdp-off ≈ 0. Pair-STDP here learns
WHAT you play (which material/anatomy), not what-follows-what. For
conversational relevance that's the useful half; sequence memory would need
different machinery (eligibility traces / reservoir readout).

**Finding 2 — raw reverberation cannot answer** (first dialogue run): with no
drive, network activity dies ~100 ms after the call ends (STDP-on answered 0%
of calls; STDP-off occasionally "answered" with 300-note runaway bursts).
The substrate either goes silent or seizes — there is no natural gap-answer.

**Fix — walkers are the answer voice**: when a call ends (600 ms silence),
walkers are teleported onto the neurons the call activated most; their
weighted traversal during the response window IS the answer. Result: 100% of
calls answered (~30 notes), immediate relatedness ~0.5 (tonotopic seeding),
and session drift +0.057 with STDP vs +0.020 without — the dialogue
measurably converges toward the player's idiom, though the effect is modest.

Duet additions: q&a toggle (model holds voice while you play, answers in the
gap), Dialogue panel (exchanges, call→response counts, relatedness score =
degree-histogram cosine, rolling average), walkers default 2.

All 35 tests pass. Next: reward-modulated STDP (eligibility traces would make
the reinforce button act on *synapses*, not just survival, and could crack
sequence learning); response density/length shaping; MIDI-file calls.

## v6 — attention (attention.html): MA-SNN's result reproduced musically

Third experiment page, leaving lab and duet as untouched baselines. Borrowed
from the literature:

- **Regional attention** (MA-SNN, arXiv:2209.13929, gradient-free): leaf
  regions = channels; per-region gain from cosine similarity between the
  region's pitch profile and a recency-weighted histogram of recently heard
  notes; gains modulate synaptic delivery (`engine.modulation`).
- **Temporal attention**: recency weighting in the heard-note histogram +
  recency-weighted walker seeding (answers pick up the tail of the phrase).
- **Attention as morphogen** (novel here): strongly attended regions receive
  survival energy each update — sustained attention shapes development.
- **Energy accounting** (SGNNBench discipline): spikes-per-exchange logged in
  the dialogue benchmark and UI.

**Ablation** (`npm run experiment:attention`, 60-exchange idiom sessions,
4 seeds, dev+STDP on in all arms):

```
attn-off:            relatedness 0.544   spikes/exchange 485
balanced (1±s/2):    relatedness 0.561   spikes/exchange 534   ← no benefit
suppress-only:       relatedness 0.610   spikes/exchange  73   ← the result
```

Suppress-only attention (best-matching region keeps full voice, others
damped, nothing amplified) reproduces MA-SNN's headline: **sparser AND
better** — +12% relatedness at **85% fewer spikes** (MA-SNN reported 84.9%
spike reduction on DVS Gesture — coincidentally close). Symmetric boosting
failed: amplification destabilizes a recurrent net (one seed ran away);
suppression only removes off-material noise. `bias: 'suppress'` is now the
default.

What Morpho added: the regions attention operates over ARE the developmental
units, so attention-guided survival plugs straight into growth/pruning — an
attention-shapes-structure loop none of the referenced papers have (they all
use fixed architectures). What Morpho would impede: chasing the papers'
learned-attention results at scale — no gradient path here, tiny graphs,
topology changes under you. That work belongs in a separate fixed-topology
branch if ever wanted.

All 42 tests pass.

## v7 — research sprint: controls, R-STDP, persistence, MIDI, STSA

Full write-up with tables: https://soundlark.studio/research.html

- **Rhythm relatedness** added to the dialogue metric (IOI-histogram cosine).
- **Specialization control** (`experiment:specialization`): NULL — train on
  idiom A, probe A vs unseen B: the index does not grow, and pre-training
  A-advantage is architectural. The organism mirrors; it does not specialize.
- **Attention-as-morphogen** (`experiment:morphogen`): coverage of the played
  idiom rises in BOTH arms (0.56→0.64 — activity-driven development is itself
  a morphogen); the attention energy trickle adds a modest sharpening
  (Δcoverage +0.077 vs +0.062, late rel 0.596 vs 0.566) and keeps organisms
  ~50% larger. (Also fixed: trickle threshold was unreachable in suppress
  mode — both arms had been bit-identical.)
- **R-STDP** (`experiment:rstdp`): eligibility traces + reward implemented
  (reinforce button now delivers targeted synaptic credit; lab.reward()).
  Sequence learning still NULL: contingent reward beats random reward but not
  immediate STDP. Order has nowhere to live in this anatomy.
- **Organism persistence**: full deterministic snapshots (RNG streams,
  in-flight spikes, traces, walkers, attention) — restored organisms continue
  spike-for-spike identically (tested). Save/load/export/import on all pages.
  Required refactor: per-graph id counters (module globals broke coexisting
  organisms).
- **MIDI-file training**: minimal SMF parser; "train from midi" replays a
  score into the sensory layer on sim time (speed control = training speed).
- **Style capture** (`experiment:style`, Ode to Joy ×30 vs untrained twin):
  pitch style leans trained (0.81 vs 0.76; one seed 0.87 vs 0.63); rhythm
  style NULL — nothing learns rhythm yet. Next mechanism: learned walker IOI
  vocabulary.
- **STSA-inspired temporal mixing** (SpikeVoice, ACL 2024): attention over
  context depth (3 timescales, sharpness-weighted) before structure. Ties
  with plain suppression under a static idiom (0.610 vs 0.602); fair test
  (idiom-switch sessions) queued. On by default on attention.html.

All 48 tests pass.

## v8 — language sideline: tiny shakespeare as a liquid state machine

(`experiment:language`, dataset fetched+cached on first run)

Characters → input spikes → frozen recurrent organism as reservoir →
closed-form ridge readout (backprop-free) predicting the next character.
Arms: exact bigram, char-only readout (sanity ≈ bigram ✓), fresh reservoir,
reservoir after developmental exposure (dev+STDP while "listening" to 15k
chars, then frozen).

```
uniform 5.95 bpc · bigram 28.8% / 3.54 bpc
char-only readout 28.6% · fresh reservoir 29.2% · EXPOSED reservoir 31.7%
(seed 7 replication: fresh 29.3% → exposed 31.4%; organisms grew ~150→260
 neurons during exposure)  · char transformer reference ≈ 58% / 1.5 bpc
```

Honest verdict: not remotely transformer-comparable (expected — no gradient
path, tiny graphs), but the replicated +2.5pp from developmental exposure is
the interesting result: development+STDP acts as backprop-free
representation learning on the reservoir. Related reading now in repo docs:
Forward-Forward SNNs (arXiv:2502.20411 — the natural next step for a
backprop-free readout stack), Spyx (2402.18994) and Training Deep SNNs
(2006.04436) — the surrogate-gradient track we deliberately keep out of this
substrate.

## Hypothesis (from the brief, §42)

> Can a compact recursive developmental grammar generate a spiking neural
> structure whose ongoing neural activity influences which parts of that
> structure grow, survive, collapse and regrow — and does that produce
> musically interesting dynamics that a fixed SNN or a conventional generative
> graph would not?

## What was built

A self-contained browser lab (no build step, no dependencies) implementing the
brief's three-layer architecture on three timescales:

| Layer | Module | Timescale |
|---|---|---|
| Developmental grammar ("morpho-lite") | `js/morpho/grammar.js` | slow — epochs |
| Development feedback (grow/prune/homeostasis) | `js/morpho/development.js` | slow — epochs |
| Neural graph (neurons, synapses, regions) | `js/neural/graph.js` | — |
| Spiking engine (LIF, delays, recurrence) | `js/neural/engine.js` | fast — 1 ms steps |
| Activity statistics (EMA rates) | `js/neural/activity.js` | medium — per epoch |
| Sonification (spikes → scale notes) | `js/ui/audio.js` | — |

The grammar is not a parser for real Morpho syntax; it is the developmental
*semantics* (recursive region → population → neuron expansion with seeded
stochastic choices). A real Morpho front-end would compile to the same
`growNetwork` / `wireNeuronIntoRegion` calls — that is the documented
integration point for the existing Morpho web repo.

**The musical mapping is structural, which is the actual experiment:** each
output neuron receives a stable scale-degree slot *at birth*. Growth adds
pitches to the texture; pruning removes them. You are not hearing a
visualization of the music — the network's anatomy *is* the note pool, its
spike timing *is* the rhythm (input pulses + synaptic delays + refractory
periods), and its development *is* the arrangement changing.

## Protocol

1. `npm test` — 16 headless tests covering LIF dynamics, delays, recurrence,
   determinism, grammar expansion, budgets, pruning safety, long-run
   boundedness.
2. `npm run serve` → http://localhost:8765 — enable **sound**, press **run**.
3. Compare conditions:
   - **development off** (frozen organism) vs **on** — does development add
     musical interest, or just noise?
   - multiple **seeds** — do different genotypes produce recognizably
     different organisms/music?
   - **pulse period** and **scale** — musical exploration.

## Judgment criteria

The idea is *promising* if, over ~5 simulated minutes:

- [x] the network neither dies (silence) nor saturates (seizure) — homeostatic
      band mostly holds
- [x] growth AND pruning both occur repeatedly (structural churn, not one-way
      growth)
- [x] different seeds → different structural trajectories (genotype matters)
- [x] deterministic replay (same seed = same spikes = same music)
- [ ] development-ON sounds *noticeably* different from development-OFF after a
      few minutes (listen and judge — subjective)
- [ ] at least one emergent behaviour not explicitly programmed

## Initial findings (headless, 40 epochs, seeds 42 / 7 / 99)

```
seed 42: 65 neurons, 467 synapses, 22 grown /  3 pruned, mean 5.8 Hz
seed  7: 56 neurons, 357 synapses, 44 grown / 34 pruned, mean 2.2 Hz
seed 99: 54 neurons, 335 synapses, 21 grown / 13 pruned, mean 3.3 Hz
```

Observations:

1. **Alternating quiet/burst regimes emerged without being programmed.** Rate
   trajectories oscillate (e.g. seed 7: 54 Hz → 2 Hz → 0.9 Hz → 16 Hz → 3 Hz).
   The growth/prune loop plus recurrent delays produces slow structural
   "breathing" — the expansion → scarcity → pruning → recovery cycle §21 of
   the brief hoped for, even without an explicit energy model. Musically this
   reads as sections: sparse passages then dense flurries.
2. **Churn cycles in silent regions.** A region that goes silent grows
   excitatory neurons (homeostatic pressure), which stay silent (no input
   path), get pruned at min-age, and regrow. This is "development, forgetting,
   redevelopment" in miniature — but it also shows the survival metric is too
   naive (brief §16 predicted this): activity alone doesn't measure
   *usefulness*. A silent region may need a new *long-range afferent*, not
   more local neurons. → Experiment idea below.
3. **Genotype legibility:** different seeds are audibly different organisms
   (different voice counts, register spreads, echo patterns from long-range
   delays). The compact-grammar → distinct-phenotype claim holds at this scale.
4. **Budgets hold.** 60+ epochs, bounded memory, no invalid graph states.

## Verdict so far

The core loop (grammar → structure → spikes → statistics → structural change →
new dynamics) closes, stays stable, and is *audible*. The mechanism is worth
pursuing. The weak point is exactly where the brief predicted: the survival /
growth heuristics are too local. Structure responds to activity, but not yet
to *usefulness*.

## Next experiments (in order of expected information gain)

1. ~~**Long-range growth rule**~~ — done in v2 (afferent repair).
2. **Weight-to-structure (brief §14)** — add simple STDP, then let
   persistently strong synapses trigger pathway duplication with a delayed
   echo path. Musically: motifs that reinforce themselves get elaborated.
3. **Collapse/regrowth (brief §18–19)** — fold low-value leaf regions into
   proxy nodes retaining their developmental rule + pitch identity; regrow
   later, optionally mutated. Musically: motif recurrence with variation.
4. **Walker ecology** — collisions, spawning on branch, probability attractor
   (plugin §23: an XY attractor pulling melodic activity around the graph —
   easy to prototype with the mouse in the browser).
5. **A real Morpho front-end** — replace `DEFAULT_GRAMMAR` params with parsed
   Morpho source from the existing repo, compiling to the same grow calls.
6. **MIDI out** (Web MIDI) so the organism can play hardware/DAW instruments.

## Reproducibility

Everything is seeded (`js/core/rng.js`, three independent streams: build /
sim / development). Same seed + same parameters = identical topology and
spike-for-spike identical history (covered by test). Changing tempo/scale/
density sliders mid-run affects audio only, not simulation determinism —
except the pulse slider, which changes the input drive and thus the sim.

## v9 — error-driven growth (`experiment:growth`)

The capacity hypothesis: grow when unable to answer correctly, stop when
capacity suffices. Online delta-rule readout streams tiny shakespeare
(100k chars); rolling error sets development's growth budget; correct
predictions deposit survival energy on just-active neurons. Controls:
frozen, and always-grow (max pressure regardless of error).

```
frozen 143n → 31.9% | always-grow 600n(cap) → 32.3% | ERROR-DRIVEN 600n → 33.3%
ceiling 1400: error arm self-limits at 834n (growth decelerates as error
falls, +100/10k chars early → +30-40 late), accuracy saturates ~600n → 33.3%
```

Error-gated growth beats frozen AND blind growth at equal size — error
gating + correctness-survival direct capacity usefully. The
progressive-growth-to-adequacy trajectory is visible; the residual ceiling
is the linear readout, not the organism. Honest negative: minimal FF head
(1 layer, 64 hidden) badly underperforms ridge (12.1% / 3.7%) — needs
normalization+depth before FF is competitive here.

Separate write-up: https://soundlark.studio/language.html

## v10 — big brain (`experiment:bigbrain`): 120k neurons, prune down

New SoA engine (typed arrays, CSR, event-driven, lazy decay): 120,000-neuron
4-layer structured spiking net (FF+recurrence+skips+feedback+15% inhibition),
full tiny shakespeare corpus, ~75 chars/s. Stability recipe that made deep
spiking compute possible: ~2 Hz sparse targets, CONTINUOUS per-layer
homeostatic thresholds, recurrent gain damped (×0.45 rec, ×0.3 feedback),
membrane clamp at 3×thr. First attempts were silent (weights too weak) then
seizing (200 Hz) — both preserved in git history as measured failures.

Results:
- 120k structured brain: 33.0% held-out — MATCHES the 834-neuron grown
  organism → the ~33% ceiling is the linear readout, not capacity.
- Naive error-credit pruning kills inhibition first → seizure → collapse
  (2030 → 5233 → 197 spikes/char across rounds; acc → 25%).
- ROLE-AWARE pruning (E and I pruned separately, bottom 25% by credit per
  round): 120k → 50k with accuracy intact (32.7%); floor at ~38k.

The overprovision-then-prune regime is validated with a measured boundary
and a legible failure law: pruning must preserve the stabilizing scaffolding.
Write-up: https://soundlark.studio/language.html

## v11 — readout v2 + the Mamba mechanism (`experiment:readout2`)

- Previous-char context → **34.2%, new best** (from 33.0%).
- Multi-τ trace banks (20/80/320ms = diagonal SSM memory): at matched dims,
  **selective (bigram-surprise-gated) state beats plain slow state 30.0 vs
  28.5** — Mamba-Spike/SpikingMamba's core mechanism (input-dependent state
  write) transfers to our gradient-free substrate, directionally.
- Scaling the bank to 3203 dims at 16k fit samples collapsed the ridge
  (14–18%) — sample-budget artifact (5:1 ratio, correlated slow traces).
  Lever: more fit data / λ tuning before bigger banks.
- FF head v2 (2-layer, standardized, goodness rerank): 10.1% — third failure;
  shallow FF on reservoir features is a robust negative in our hands.

## v12 — autoregression (`experiment:autoreg`)

Already autoregressive in factorization (teacher-forced next-char). Tested
the missing senses at the corrected sample budget (40k fit):
- same features as v11 best: 34.2% → **38.0%** (data alone)
- + 2nd previous char: **39.1% — new best** (backprop-free, vs bigram 28.8,
  transformer ≈58)
- belief feedback (two-stage posterior stacking): 39.2% — honest null under
  teacher forcing
- free-running generation (brain hears its own sampled output): gibberish —
  exposure bias made vivid; generation, not held-out accuracy, is the honest
  distant yardstick.

## v13 — evolution of development (`experiment:evolve`) — design pre-registered

The missing experiment from the original Morpho discussion: evolve the
**rule for producing networks**, never a network. An 11-gene genome
parameterizes the same per-neuron statistical wiring law that built the
v10–v12 structured brains (E/I ratio, ff/rec/skip/fb/inh fan-outs, weight
bases, recurrent/feedback gains, delay spread) — genome length constant in
N, so one genome instantiates at any size. Readout frozen at v12's arm E
(fast taps + cur/prev/prev2 one-hots, ridge): if fitness moves, development
moved it, not the learner. Layer count stays fixed at 4 — structural genes
are v14's question.

Pre-registered protocol (before results were known):

- **Fitness** `F(g) = mean_N acc − 0.5·std_N acc − 0.02·(syn/neuron)/100`
  over N ∈ {2k, 4k, 8k} — selecting for scale *robustness* plus a
  connectivity cost, not accuracy at one size.
- **Optimizer** deliberately boring (μ+λ) ES: pop 16, 12 generations, top
  25% survive, uniform crossover (p=0.3) + gaussian mutation (σ=0.15,
  p=0.5/gene) in normalized gene space. Common random numbers per
  generation (shared build seed + corpus window, rotated each generation);
  elites re-evaluated every generation. The novelty budget is spent on
  *what* is evolved, not the optimizer.
- **Baselines** v12 hand genome · random genomes (= gen-0 population) ·
  best of gen 0 — identical pipeline, one shared fresh window at the end.
- **Transfer** frozen winners instantiated at held-out 16k/32k/60k with the
  identical (small) readout budget; **120k stays genuinely held out** behind
  an explicit `--120k` flag until everything else is done.
- **Per-phenotype metrics logged**, not just accuracy: synapses/neuron,
  realized E/I, recurrent + long-range fractions, spikes/char, dead-neuron
  fraction, build/sim time — so "evolution found accuracy by tripling
  connectivity" is distinguishable from a genuinely better law.

Deterministic and resumable (checkpoint per generation in
`experiments/results/`); pipeline validated end-to-end by 5 new tests
(53 total) + smoke run. Results below when the runs complete.

### v13 results (3 independent evolutionary seeds; 120k unlocked last)

Evolution ran at N ∈ {2k, 4k, 8k}; every genome below was then FROZEN and
instantiated at held-out 16k/32k/60k/120k (15–60× beyond evolution).
Numbers are held-out next-char accuracy under the frozen small readout
budget (256 taps, 8k fit — NOT comparable to v12's 39.1%, which used 1024
taps / 40k fit).

```
                 evo scales (2/4/8k)      held-out (16/32/60/120k)     syn/n
seed 42  v12     36.3 / 37.3 / 37.4    37.2 / 38.4 / 38.3 / 39.0      19.9
         gen0    36.0 / 37.7 / 36.9    39.2 / 40.2 / 38.8 / 40.7      11.2
         evolved 36.6 / 37.3 / 37.3    38.5 / 38.7 / 38.4 / 38.9      10.6
seed 7   v12     36.3 / 36.3 / 36.9    37.6 / 37.3 / 36.9 / 37.8      19.9
         gen0    36.3 / 36.3 / 37.5    38.2 / 39.3 / 39.1 / 38.9      16.9
         evolved 36.0 / 36.1 / 36.6    38.5 / 38.6 / 39.1 / 38.7      12.0
seed 99  v12     36.3 / 36.3 / 35.8    38.5 / 37.3 / 37.3 / 36.1      19.9
         gen0    36.8 / 37.2 / 37.0    38.9 / 39.7 / 39.3 / 39.3      12.9
         evolved 36.2 / 36.5 / 37.5    37.9 / 38.5 / 37.7 / 38.1       5.9
```

Findings, in order of confidence:

1. **Scale transfer is real.** All six small-scale-selected genomes
   (gen0-best + evolved-best × 3 seeds) hold or improve at 120k
   (38.1–40.7%, mean 39.1%) vs the hand law's 36.1–39.0% (mean 37.6%),
   at 1.2–3.3× fewer synapses (0.71–2.03M vs 2.38M). Nothing collapsed.
   The developmental representation — 11 numbers — carries to sizes it
   never saw. The hand law is also the *least robust* across build seeds
   at scale (36.1% with 90%-dead layers on one seed, 33k spikes/char
   near-seizure on another): selection at small scale implicitly selected
   for physiology that survives the frozen homeostatic protocol.
2. **Convergent developmental signature across independent lineages:**
   all three winners pushed inhibitory fraction to the gene's UPPER BOUND
   (0.34–0.35 vs hand 0.15), cut feedforward fan (14 → 2–7), kept or
   raised long-range skips, and nearly eliminated intra-layer recurrence
   (6 → 0–2). Sparse, inhibition-rich, skip-dominated, barely recurrent.
   The bound-pinning means the true optimum may lie outside the legal
   range — widen it in v14. And recurrence being selected out says that
   under THIS readout, reservoir recurrence does not earn its synapses.
3. **Honest null on the optimizer: accuracy-wise, evolution ≈ random
   search.** At evolution scales all arms sit at ~36–37%; gen-0's best
   random genome transfers as well as (often better than) the evolved
   one. 12 generations of (μ+λ) bought connectivity reduction (19.9 →
   5.9–12.0 syn/n at matched accuracy — a real Pareto move the β-penalty
   only partly explains), not accuracy. The paper question "do laws
   selected small deploy large?" answers YES, but the selection pressure
   that matters so far is *screening the parameter space*, not iterating
   on it. More generations, bigger populations, or lower eval noise are
   the v14 levers.
4. **Protocol caveat:** the frozen 1500-char calibration under-adapts
   thresholds at ≥32k (large dead fractions in all arms). Held-out
   accuracy at scale therefore leans on sparse activity + char one-hots.
   All arms share the handicap, so comparisons stand, but absolute
   numbers at scale are depressed.

Reproduce: `npm run experiment:evolve [seed]`, then
`npm run experiment:evolve -- transfer results/evolve-seed<S>.json --120k`.
Full per-phenotype metrics in `experiments/results/evolve-seed{42,7,99}.json`.

## v14 — structural evolution + the ladder attempt — design pre-registered

Two pushes to move v13's results higher, protocols fixed before results:

**Full-budget ladder eval** (`experiment:fullbudget`): v13's accuracies used
a deliberately small frozen readout (256 taps, 8k fit) — fair, but not
ladder-comparable. The best small-scale-selected genomes (seed-42 gen-0,
seed-42 evolved, seed-99 gen-0) are instantiated at 120k under the FULL v12
budget (1024 taps, 40k fit, 3k test, 5k calib, cur/prev/prev2), v12 hand
genome as same-window control. Ladder reference to beat: 39.1%.

**v14 evolution** (`experiment:evolve2`): v13 with its three identified
constraints removed —
1. wider bounds on every gene v13 pinned (inhib_frac hi 0.35→0.55,
   delay_scale lo 0.3→0.15, inh_fan lo 4→2, skip hi 8→12, w_exc, w_inh,
   rec_gain widened);
2. first STRUCTURAL gene: n_layers ∈ {2..6} enters the genome — depth is
   no longer the designer's choice;
3. a fair fight for the optimizer vs v13's evolution≈random-search null:
   pop 24 (was 16), 20 generations (was 12), 3k test chars (was 1.5k —
   halves per-scale eval noise to ~0.9pp). Same boring (μ+λ), same fitness,
   same frozen readout/physiology. Transfer-mode calibration fixed at 5000
   chars (v13's 1500 under-adapted at ≥32k — all arms share the fix).
   Final-report arms include v13's winning genome re-encoded in the v14
   gene space, so v14-vs-v13 progress is measured on a shared window.
   3 independent seeds; 120k held out behind --120k as before.

3 new tests (56 total); smoke-validated. Results below when runs complete.

### v14 results, part 1 — the ladder falls: 42.5% (full-budget eval)

The three best v13-selected genomes at 120k under the FULL v12 readout
budget (1024 taps, 40k fit, same corpus windows as v12; build seed differs):

```
v12-hand (control)   36.1%   syn/n 19.9   SEIZED (120k spikes/char ≈ 100 Hz)
gen0-best seed42     42.5%   syn/n 11.2   ← NEW BEST (prior best 39.1%)
evolved-best seed42  39.8%   syn/n 10.6
gen0-best seed99     40.4%   syn/n 12.9
```

- **All three selected genomes beat the prior 39.1% best**, at 40–70%
  fewer synapses; the winner by +3.4pp is the seed-42 gen-0 genome
  (inhibition-rich, ff_fan 7, skip-heavy, w_exc low). Backprop-free
  throughout, as ever.
- The v12 hand law's control run SEIZED on this build seed (100 Hz,
  36.1%) — its published 39.1% rode a favorable build. Third independent
  line of evidence (after v13 transfer + v13 final report) that the
  selected developmental laws are more physiologically robust than the
  hand design, not just sparser.
- The stable sparse phenotypes also evaluated ~40× faster than the
  seizing control (1–2 min/arm vs 68 min): spikes × synapses is the
  simulation cost, and evolution minimized both.
- Caveat: one build seed per arm (the protocol's deterministic seed);
  v13's transfer data says selected genomes vary less across builds than
  the hand law, but 42.5% carries single-build uncertainty.

## v15 — generation as a measurement + the linear ceiling — design pre-registered

Motivated by the scaling analysis (substrate size N: measured zero slope;
trained readout params × data: the live axis; long context: the structural
gap to the ~10.7M-param transformer reference). Two instruments, protocols
fixed before results:

**Generation benchmark** (`experiment:genbench`): make generation quality a
number. (1) Metric: bits/char of generated text under an interpolated 1–5
gram model trained on 900k corpus chars; real held-out text = floor,
uniform noise = ceiling. (2) Closed-form scheduled sampling: drive the
brain with a mixture of true and self-sampled chars (p ∈ {0, 0.25, 0.5}),
collect ridge rows under that mixed distribution (targets stay true),
refit closed-form — exposure-bias repair with zero backprop. Substrate:
the 42.5% ladder-best genome @120k, full budget, temp 0.8.
PREDICTION (memory-gap hypothesis): scheduled sampling improves gen-bpc
somewhat; teacher-forced accuracy and generation quality stay far apart.

**Scaling sweep** (`experiment:scaling`): where does the linear readout
saturate? Taps axis 256/512/1024/2048 @40k fit; data axis 10k/20k/40k/80k
@1024 taps, same genome/protocol. Trained params = (taps+196)×65.
PREDICTION: both axes bend toward a mid-40s plateau (a linear map over
spike traces + 3 chars of context is n-gram-class); past it the lever is
mechanism (deep readout / memory-rewarded substrate), not size.

Both per-config checkpointed. 2 new tests (58 total). Results below.

## v16 — beyond the linear form — program pre-registered

Three attacks on the (predicted) linear ceiling, in order of increasing
depth of change; all on the 42.5% ladder-best genome @120k, full budget,
so every arm is directly comparable:

- **v16a `experiment:readout3` — richer features, linear solve.** Five
  arms, only the feature map changes: A0 rate-trace control; A1 multi-τ
  banks (20/80/320ms — v11's mechanism at the 40k sample budget v11
  lacked); A2 second-order tap-pair products; A3 fixed random ReLU
  projections (ELM); A4 char-gated bilinear taps. All closed-form ridge.
  PREDICTION: A1/A4 win if the ceiling is a feature-map problem; five-way
  tie at ~42% means the reservoir's readable information is exhausted.
- **v16c — Forward-Forward done properly** (to be built): normalization +
  depth + goodness design per the FF-SNN literature, replacing the three
  casual shallow attempts that failed (12.1%, 10.1%, —). The readout-depth
  axis, gradient-free.
- **v16d — three-factor plasticity** (to be built): reward-modulated STDP
  (local eligibility × global correctness scalar) shaping the substrate's
  1.3M weights during a training phase, then frozen readout as usual.
  Converts untrained parameters into slowly-trained ones — the deepest
  change, and the return of the substrate's biology.

Substrate depth is settled (v14: 2 layers, 3/3 seeds); v16c explores
readout depth instead. Queue: after v14 transfers + v15 genbench/scaling.

### v14 results, part 2 — structural evolution: depth 2, unanimously

3 independent seeds, 20 generations each, layer count genomic (2–6),
bounds widened everywhere v13 pinned. Shared-window final reports and
held-out transfer (calib 5000; all numbers within-protocol only):

```
shared window (2/4/8k)   seed42   seed7    seed99
  v12-hand               0.319    0.319    0.317      L=4, syn/n 19.9
  v13-winner             0.331    0.331    0.331      L=4, syn/n 11.1
  v14-gen0-best          0.336    0.327    0.339      L=2/3/2
  v14-evolved            0.338    0.330    0.336      L=2, syn/n 7.9/2.9/12.2

held-out 120k            v12-hand   gen0-best   evolved-best
  seed42                 34.6%      35.9%       36.8%
  seed7                  35.1%      37.1%       36.5%
  seed99                 35.3%      37.6%       37.1%
```

Findings:

1. **Depth 2 beats depth 4, unanimously.** All three lineages' winners
   chose n_layers=2 — the first structural gene contradicted the designer
   in every independent run, and the choice survives transfer to 120k
   (every selected genome beats its v12-hand control there).
2. **The v13 signature was conditional on imposed depth.** With depth
   free, inhibitory fraction abandoned v13's 0.35 pin (seed42 0.05,
   seed7 0.08, seed99 0.22): the "inhibition-rich" law was compensation
   for forced 4-layer stability, not a universal principle. Depth-2 is
   the convergent discovery; within it, many wirings work (ff_fan spans
   5–28 across winners). A caution for interpreting any evolved genome:
   signatures are relative to the constraint set.
3. **Sparsity extreme:** seed7's winner computes at 2.9 syn/neuron
   (~350k synapses at 120k — 6.8× fewer than the hand law) and still
   beats it at every held-out size. All v14 arms beat v13-winner on the
   shared windows.
4. **Evolution vs gen-0: 2/3 seeds, small margins.** Better than v13's
   0/3, still not decisive. The noise reduction helped; the honest
   statement remains "screening dominates, iteration adds a little."
5. v12-hand seized during evaluation twice more (seed99 60k: 71k
   spikes/char; seed42 120k: 49k) — the fragility result is now
   systematic across v13, v14 and the full-budget control.

### v15 results, part 1 — generation measured (`experiment:genbench`)

Instrument: bits/char under a 1–5 gram corpus model; floor = real held-out
text 2.52 bpc, ceiling = uniform noise 8.25 bpc. Substrate: ladder-best
genome @120k, full budget (teacher-forced 43.2% on this window).

**Protocol amendment (flat-sampler artifact, discovered mid-run):** ridge
scores approximate probabilities in [0,1]; softmaxing them at T≈1 (the
pre-registered decoder, inherited from v12's generation demo) is
near-uniform — first run generated at 8.23 bpc ≈ noise BY CONSTRUCTION.
Amended to power-law sampling on clamped scores (greedy = argmax);
flat-sampler run preserved in results/genbench-seed42-flatsampler.*.

```
                     gen bpc    teacher-forced acc
p=0    sampled T0.8    4.38        43.2%
p=0    greedy          1.93*       43.2%
p=0.25 sampled         4.35        41.0%
p=0.5  sampled         4.57        36.5%
```

1. **v12's "gibberish" verdict retroactively revised:** decoded properly,
   free-running generation produces word-fragment English ("…thankeyz n:
   the lhod magnI't sart…") at 4.38 bpc — far from noise. The model
   generates at roughly trigram quality; the old demo's decoder was
   flattening it to uniform.
2. **(*) Metric caveat, caught immediately: gen-bpc is gameable.** Greedy
   scores BELOW the real-text floor by looping "the in the come" — the
   n-gram scorer rewards degenerate blandness. bpc needs a
   repetition/diversity companion before "beats the floor" means
   anything. Honest operating number: sampled 4.38.
3. **Scheduled sampling: null, as the memory-gap hypothesis predicted.**
   Generation improves ≤0.03 bpc at p=0.25 and worsens at p=0.5, while
   teacher-forced accuracy degrades monotonically. Exposure bias is not
   the binding constraint; the teacher-forced ↔ generation gap
   (43.2% vs 4.38-bpc trigram-grade text) is a memory gap. The
   pre-registered prediction stands confirmed.

### v15 results, part 2 + v16a — the ceiling was samples-per-parameter

Scaling sweep (ladder-best genome @120k, single axes then joint):

```
taps  @40k fit:  256→38.2  512→39.8  1024→42.5  2048→43.2   (bending)
fit  @1024 taps: 10k→30.4  20k→41.0  40k→42.5  80k→42.4     (saturated)
joint:           2048×80k→44.5   4096×80k→45.6               (re-opened!)
```

The single-axis "mid-40s plateau" prediction was WRONG in the useful way:
data saturates only at fixed parameter count. Growing taps × data jointly
keeps climbing (~+1.1pp per taps doubling at 80k fit). The binding
constraint all along was samples-per-parameter, not the linear form per
se — the "networks too small" instinct was right, about the READOUT.

v16a feature arms (matched d where noted):

```
@1024 taps, 40k fit:  rates 43.2 · pairs 43.9 · ELM 44.5 · CHAR-GATE 44.9
@2048 taps, 80k fit:  rates 44.7 · CHAR-GATE 45.8  ← NEW BEST
@1024 taps, 80k fit:  multi-τ 42.2 (fair test, 25 samples/param)
```

1. **Nonlinearity beats size at matched parameters:** char-gated bilinear
   taps (traces read differently per current char) add +1.1–1.7pp over
   rate controls at equal d; ELM +1.3. The pre-registered A4 prediction
   confirmed. 45.8% = new ladder best (session arc 39.1 → 45.8, all
   backprop-free, ~277k trained params vs transformer's 10.7M).
2. **Multi-τ is now a clean null.** 35.8% at 13 samples/param (starved,
   as pre-flagged) but only 42.2% at 25:1 — slow trace banks add nothing
   over fast rates even with adequate data. The reservoir's long-τ
   dynamics carry no additional readable long-context information:
   independent corroboration of the memory gap, and consistent with
   evolution deleting recurrence.
3. Implementation note: evaluate()-based numbers (scaling.mjs) and
   readout3 numbers differ by a systematic ~+0.2–0.7pp for identical
   configs (e.g. 44.5 vs 44.7 at 2048×80k) — separate engine
   implementations; comparisons are made within-file only.

**v16a addendum:** char-gate at 4096 taps × 80k fit = **47.2%** (+1.4 over
45.8; 407k trained params; 28.9m solve). The joint nonlinear axis is still
climbing. Session arc 39.1 → 47.2, all closed-form. Practical note: ridge
solve time is now the binding cost (O(d³) — 29 min at d=6263); pushing
this axis further means streaming accumulation + a better solver, or depth
(v16c) instead of width.

### v16c results — Forward-Forward: a rigorous negative, the fourth and final

The serious attempt (standardized inputs, 3×512 layers with inter-layer
L2 normalization, label embedding, ridge-confusable hard negatives, 8
epochs) against ridge on IDENTICAL standardized features:

```
ridge control 42.5% · ff-depth1 16.9% · ff-depth3 11.5%
```

Depth HURT (depth-1 beats depth-3 by 5.4pp), inverting the FF-literature
expectation, and the gap to ridge is 26pp with every previously-missing
ingredient present. Caveat: one hyperparameter point (θ=2, embed 4, lr
0.03) — but the depth inversion plus the margin make tuning unpromising.
Verdict after 4 attempts (12.1% → 10.1% → casual → 11.5%/16.9% rigorous):
goodness-based FF classification is mismatched to dense 65-way reservoir
readout in our hands. The readout-depth lever, if it exists, is stacked
closed-form fits — not FF. v16d (substrate plasticity) is the remaining
deep lever.

### v16d results — three-factor plasticity: a stable null

Reward-modulated STDP (potentiation-only eligibility via pre-trace ×
post-spike, global correct/wrong scalar, multiplicative sign-preserving
updates clamped to [0.25×, 2.5×] birth magnitude, η=0.0015, 60k chars)
vs an identical-stream frozen control:

```
plastic  ridge 41.5% · 669,171 synapses changed (mean rel Δ 10.7%) · stable
control  ridge 41.1% · 0 changed
online during training: plastic 36.0% vs control 39.9% (non-stationarity)
```

The substrate absorbed reward-gated ~10% perturbation of half its
synapses with NO readability consequence in either direction — striking
robustness, zero sculpting. Caveats: one η, within-char eligibility
horizon, potentiation-only. Consistent with v7's R-STDP sequence null:
this local rule does not convert reward into better features here.

**v16 program ledger:** features +4.0pp (47.2% best, char-gated bilinear)
· FF depth: rigorous negative (depth inverts, 26pp behind ridge) ·
substrate plasticity: null. The reservoir's readable information is what
development made it; the remaining transformer gap is long-context
memory, which none of the readout-side or local-plasticity levers touch.
Next lever with a mechanism behind it: memory-REWARDING task families
(v15 plan) so evolution stops deleting recurrence — select the substrate
for memory, then read language out of it.
