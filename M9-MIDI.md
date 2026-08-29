# M9 — ARIA-MIDI continuator: first four-arm results (2026-08-28)

Data: aria-midi unique subset, 32,522 performances -> 292.7M event
tokens (219-vocab: NOTE_ON/OFF[88+88], TIME_SHIFT[32 log buckets],
VEL[8]). Models: whitebox stack (CausalCRATEM2 + MLP blocks), d448
x 12L, ctx 512, 20k steps batch 32 (~330M tokens ~ 1 epoch), one
seed, L4 + FLA. wandb: morpho-whitebox-midi.

| arm | params | val ppl |
|---|---|---|
| counter (CRSA) | 22.0M | 7.29 |
| WKV (RWKV4-form) | 29.3M | 5.94 |
| hyena (param-matched, 2nd sweep) | 15.9M | 5.55 |
| mixed (half counter / half delta) | 22.7M | 5.53 |
| qkv attention (control) | 29.3M | 5.39 |
| LONGHORN | 29.4M | **5.20** |

Second-sweep notes: WKV lands between counters and the binding arms
(its normalized-counter anatomy, in its place); hyena at the
smallest size reaches the hybrid's level but does NOT repeat its
DNA-splice dominance — music's structure rewards associative state
over global convolution here. Longhorn remains champion. All six
continuations share one held-out prompt (runs/midi/*/continuation).

FINDINGS (one seed, prereg-consistent but unreplicated):
1. **Longhorn > attention on music at equal params** — the first
   domain in the program where the delta memory beats softmax
   attention outright (language: attention led). Music's repetition
   structure suits decaying associative state.
2. Association gap over counters is large (2.09 ppl, 29%) — bigger
   than language's; music joins splice as a binding domain.
3. The HYBRID pattern holds a third time: mixed closes 84% of the
   counter->longhorn gap at counter-class params, and comes within
   0.14 of attention at 77% of its size.
4. Params are NOT matched across arms (mixer-intrinsic, as in M7);
   the counter->mixed comparison IS matched (22.0 vs 22.7M).

Artifacts: whitebox/runs/midi/<arm>-s0/{ckpt.pt,continuation.mid}
(+ local .wav renders, MuseScore soundfont). All four continuations
share one held-out prompt — a controlled listening comparison.

NEXT: musical metrics (pitch-class/key consistency, IOI histogram
match, motif reuse — the "measured, musically relevant" criteria);
seeds; hyena arm on the same tokens; M8 oscillatory heads for meter
(the designed follow-on); interactive play-in/respond demo;
longer-ctx runs (512 events ~ 30-60s of music — motif memory beyond
that needs either state or length).
