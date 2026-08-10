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
