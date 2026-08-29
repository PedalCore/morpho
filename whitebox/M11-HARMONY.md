# M11 — harmony vertical x horizontal: the factorized music
# process (design, preregistered 2026-08-29; user concept)

USER FRAMING: harmony runs on two axes — VERTICAL (chords,
simultaneity) and HORIZONTAL (melody/progression, sequence). The
M10-P boundary result splits exactly along it: statistics own the
vertical (pc-traces won the ladder; within-tick coupling = chord
machinery), binding owns the horizontal (the audible randomness).

THE ENTROPY ARGUMENT: flat 88-key x 50 Hz = 4,400 coin flips/s.
Factored: progressions ~0.5-2 decisions/s over a ~50-chord
vocabulary; melody ~2-3/s over ~12 chord-constrained candidates.
A few bits/s of real decision -> sequence memory over a TINY
alphabet at musical event rate = small-machinery problem. The 29M
LM was needed by the formulation, not the music.

ARCHITECTURE (each level small, whitebox, mechanism already
validated somewhere in the program):
1. HARMONIC RHYTHM (when chords change): clock + counters
   (M10-drums class).
2. PROGRESSION (chord -> chord): small sequence memory over the
   chord vocabulary (slot-chains or tiny Longhorn) — the horizontal
   binding, bought cheaply.
3. VOICING (chord -> keys): pc-traces + within-tick bass->treble
   coupling + hand kernels (M10-P vertical machinery).
4. MELODY (line over scaffold): next note | chord + recent line;
   small candidate set; slots-or-tiny-longhorn.

BUILD LADDER (preregistered):
a. Chord lattice extraction from ARIA: segment at harmonic-change
   points (pc-profile shifts), cluster simultaneity windows ->
   chord vocabulary (inspect: do clusters = triads/7ths?). Validate
   coverage.
b. Progression models over the lattice: Markov-1/2 vs decayed
   traces vs tiny-longhorn (the counters-vs-binding question at the
   RIGHT timescale). Metric: bits/chord.
c. Melody-over-chords: candidate-set accuracy + bits/note; does a
   slot memory of recent line fragments beat order-free context?
d. Full stack render vs flat-GLM render vs M9-longhorn render —
   the three-way listening test.
PREREGISTERED BET: a sub-10k-parameter hierarchy audibly beats the
flat GLM, because decisions finally live at music's own rate and
vocabulary. Failure mode to watch: chord segmentation quality
bounds everything above it (garbage lattice -> garbage hierarchy).
