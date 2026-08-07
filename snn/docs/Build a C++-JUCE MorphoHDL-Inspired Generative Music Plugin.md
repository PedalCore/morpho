# Build a C++/JUCE MorphoHDL-Inspired Generative Music Plugin

I want to build a **C++-only JUCE audio/MIDI plugin** inspired by the open-source MorphoHDL project:

https://github.com/paradigms-of-intelligence/morpho

The goal is not to embed JavaScript, Python, WebAssembly, a browser, or a web view. I want a **native C++ implementation** that reproduces the important MorphoHDL concepts and turns graph/circuit growth into a generative musical system.

The reference MorphoHDL project is Apache 2.0 licensed.

## Core idea

MorphoHDL describes structures using recursive graph/circuit rewriting.

A cell can recursively divide into smaller cells. Those cells can divide again. Growth continues until no cells can expand.

The original demo sonifies this process:

- every cell division produces a note
- recursion depth determines register
- spatial/layout position influences melody
- once growth finishes, a signal wave moves from inputs to outputs
- cells fire according to their logic/topological depth
- the structure being built is itself the musical event source

I want to recreate this idea inside JUCE and extend it into a full generative MIDI instrument.

---

# High-level architecture

Please keep the project divided into two layers.

## 1. MorphoCore

A pure C++ library with **no JUCE dependencies**.

Suggested structure:

```text
MorphoCore/
    Tokenizer
    Parser
    AST / SSA representation
    SourceSpan / source metadata
    Compiler
    Primitive operations
    CompiledGraph
    CellStore / PinStore / NetStore
    GraphGrower
    LogicTiming
    LayoutEngine
    GrowthEvent
    GraphSnapshot
```

## 2. MorphoJUCE

JUCE-specific plugin/application code.

Suggested structure:

```text
MorphoJUCE/
    PluginProcessor
    PluginEditor
    GraphComponent
    CodeEditor
    GrowthController
    GrowthScheduler
    GenerativePlayer
    ScaleMapper
    MidiGenerator
    optional SynthEngine
```

The Morpho graph/compiler should remain reusable independently of JUCE.

---

# Reference implementation to study

Use the existing Morpho repository as a behavioral reference.

Important files include roughly:

```text
js/parser.js
js/compiler.js
js/primitives.js
js/grower.js
js/force_layout.js
js/viewer.js
js/layout_renderer.js

graphs_engine/src/main.c
```

Important behavior from the original implementation:

- small Python-like MorphoHDL syntax
- parser creates a compact SSA-like representation
- recursive cell specialization based on input bus widths
- flat Struct-of-Arrays graph representation
- cells, pins and nets stored separately
- hierarchical Morpho cells expand into child cells
- multi-wire cells can divide into smaller cells
- LUT cells and constants can be optimized
- dead-code elimination exists
- graph growth can occur breadth-first or by largest expandable cell
- logic arrival times are calculated by topological traversal
- the existing force layout already has a C backend used through WASM

Do not translate JavaScript line-for-line unnecessarily. Reimplement the semantics idiomatically in modern C++.

---

# Language support

Initially reproduce the useful subset of MorphoHDL.

Example:

```python
Xor3 = LUT(3, 0b1001_0110)
Maj3 = LUT(3, 0b1110_1000)

@morpho
def full_adder(a, b, c_in):
    sum = Xor3(a, b, c_in)
    c_out = Maj3(a, b, c_in)
    return sum, c_out

@morpho(fallback=full_adder)
def ripple_adder(a, b, c):
    a0, a1 = SPLIT(a)
    b0, b1 = SPLIT(b)

    s0, c_mid = ripple_adder(a0, b0, c)
    s1, c_out = ripple_adder(a1, b1, c_mid)

    sum = CAT(s0, s1)
    return sum, c_out
```

Support, approximately:

- decimal integers
- binary integers
- hexadecimal integers
- identifiers
- function calls
- assignments
- tuple/multiple assignments
- `def`
- `return`
- `@morpho`
- `@morpho(fallback=...)`
- indexing
- slicing
- LUT declarations
- keyword overrides where required by Morpho semantics

Primitive operations should initially include:

```text
SPLIT
CAT
REPEAT
LSLICE
HSLICE
INDEX
SLICE
```

---

# Preserve source locations

Unlike a minimal parser, every instruction should retain its source location.

For example:

```cpp
struct SourceSpan
{
    int startLine {};
    int startColumn {};
    int endLine {};
    int endColumn {};
};
```

And:

```cpp
struct Instruction
{
    OpCode op;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    SourceSpan source;
};
```

This is important because the JUCE UI should eventually be able to highlight the exact MorphoHDL statement currently being executed during graph expansion.

---

# Graph representation

Prefer a flat cache-friendly graph representation similar to the original Struct-of-Arrays design.

Conceptually:

```cpp
struct CellStore
{
    std::vector<uint8_t> active;
    std::vector<CellType> type;

    std::vector<int32_t> parent;

    std::vector<int32_t> pinStart;
    std::vector<int32_t> inputCount;

    std::vector<int32_t> netStart;
    std::vector<int32_t> outputCount;

    std::vector<float> firstTime;
    std::vector<float> lastTime;

    std::vector<std::string> name;
};
```

Also maintain flat stores for:

```text
Pins
Nets
LUT metadata
cell definitions/templates
```

Cell types should include concepts equivalent to:

```text
INPUT
OUTPUT
LUT
MORPHO
```

---

# Graph growth

Expose graph growth as explicit discrete events.

Something conceptually like:

```cpp
bool GraphGrower::expandNext();
bool GraphGrower::expandLargest();
```

Each successful expansion should produce enough metadata for visualization and music.

For example:

```cpp
struct GrowthEvent
{
    int parentCell {};
    std::vector<int> createdCells;

    int recursionDepth {};

    SourceSpan source;

    uint64_t generation {};
};
```

The graph itself should remain deterministic for a given source, input signature, traversal strategy and seed.

Support at least:

```text
Breadth-first growth
Largest-expandable-cell growth
```

---

# Layout

Port or reuse the existing C graph-layout algorithm from:

```text
graphs_engine/src/main.c
```

as native C/C++.

Do not use WASM.

The JUCE visualizer should eventually support:

- node positions
- edges
- pan
- zoom
- smooth interpolation when cells divide
- highlighting active cells
- highlighting ancestry/descendants
- signal-wave visualization

For the first implementation, visual polish is secondary to graph correctness.

---

# Logic timing

After the graph finishes growing, calculate topological signal arrival times.

Each active cell should have values conceptually equivalent to:

```text
firstTime
lastTime
logicDepth
```

Use an iterative topological/Kahn-style traversal rather than recursion.

This timing information will later drive both animation and music.

---

# MIDI and sonification

The JUCE plugin should primarily be a **MIDI generator** at first.

Internal synthesis can be added later.

There should be two distinct musical phases.

## Phase 1: BUILD

The graph grows.

Every cell division or newly created cell can produce MIDI.

Initial mapping:

```text
event               = cell division
time                = growth clock
register            = recursion depth
pitch               = spatial or structural position
velocity            = fanout / child count / graph feature
pan or MIDI CC      = X position
timbre/channel      = cell type
accent/articulation = rule or SSA instruction
```

Users should be able to select:

```text
root note
scale
octave range
growth rate
growth traversal mode
pitch mapping
velocity mapping
note duration
```

---

# Scale system

Do not hardcode only major/minor.

Use a generic interval-based scale representation.

For example:

```cpp
class Scale
{
public:
    int rootMidi = 60;

    std::vector<int> intervals {
        0, 2, 3, 5, 7, 9, 10
    };

    int noteForDegree (int degree) const;
};
```

Support presets such as:

```text
Chromatic
Major
Natural minor
Dorian
Phrygian
Mixolydian
Pentatonic
Whole tone
custom interval set
```

Eventually make microtonal mapping possible, but not required initially.

---

# Deterministic pitch mapping

The topology should have a recognizable musical identity.

Prefer:

```text
recursion depth → register
X/layout position → scale degree
logic depth → rhythmic position
cell type → channel/timbre
fanout → velocity/accent
```

If using force-layout coordinates for pitch, sample or quantize the position at an event boundary instead of allowing constantly moving node positions to continuously alter pitch.

Also consider a deterministic "structural position" independent of force layout.

---

# Phase 2: LIVE / Living Graph

Once the graph is fully grown, do NOT stop.

The finished graph should become a reusable generative musical structure.

This phase should produce **non-deterministic performances from the exact same fixed graph**.

The graph itself should not mutate.

Instead, introduce stochastic graph traversal and interpretation.

---

# GenerativePlayer

Create a separate musical engine:

```cpp
class GenerativePlayer
{
public:
    void prepare (...);
    void reset (...);
    void advance (...);

private:
    const Graph* graph = nullptr;

    Scale scale;

    RandomGenerator rng;

    std::vector<Walker> walkers;

    float entropy {};
    float momentum {};
    float repetition {};
    float density {};
    float branchBias {};
};
```

The graph engine itself should know nothing about music.

---

# Random-walk playback

A core LIVE playback mode should use one or more random walkers.

Example:

```text
             A
          /     \
         B       C
       /  \     / \
      D    E   F   G
```

Different runs might produce:

```text
A → B → E
A → C → F
A → B → D
A → C → G
```

Each visited node produces a MIDI event.

Do not use completely uniform randomness unless requested.

Use weighted stochastic decisions.

---

# Weighted traversal

Candidate edges should have scores influenced by structural and musical properties.

Conceptually:

```text
weight =
    baseWeight
    × topologyWeight
    × directionWeight
    × momentumWeight
    × spatialWeight
    × repetitionPenalty
    × cellTypeWeight
    × userBias
```

Then normalize weights and perform weighted random sampling.

---

# Entropy / Variation parameter

Expose a global parameter controlling how strongly the player follows the preferred structural route versus exploring alternatives.

Call it something like:

```text
Variation
Entropy
Chaos
```

Behavior:

```text
0%
almost deterministic / highest-scoring path

30%
small variations

60%
strong exploration

100%
near-uniform stochastic choice
```

A softmax-temperature style implementation is appropriate.

---

# Musical memory

Pure randomness usually sounds poor.

The player should maintain history.

Possible state:

```text
recent node IDs
recent pitch degrees
previous movement direction
recent intervals
recent rhythms
```

Provide behavior such as:

```text
repetition penalty
momentum
novelty preference
direction bias
```

Example:

```cpp
if (recentlyVisited (candidate))
    weight *= repetitionPenalty;
```

Momentum should allow a walker to prefer continuing approximately in its previous spatial or graph direction.

---

# Random seed

Use seeded PRNGs.

The user should be able to:

```text
generate a new seed
enter a seed manually
lock the seed
unlock the seed
```

Requirements:

```text
same graph
+ same seed
+ same parameters
= reproducible performance
```

Unlocked seed:

```text
same graph
= new performance on every run
```

This is essential in a DAW so that a good generative result can be recalled.

---

# Multiple walkers

Support more than one walker.

Suggested initial range:

```text
1–8 voices
```

Each walker should maintain independent state but use the same graph.

Interesting interactions:

- walkers may produce independent melodic voices
- simultaneous node arrivals can form chords
- walker collisions can produce accents
- collisions could optionally merge, reverse or spawn walkers later

Keep the first implementation simple but design the architecture so these behaviors are possible.

---

# Additional LIVE playback modes

Design the system so these modes can eventually coexist:

## Walk

One or more stochastic agents follow graph edges.

## Wave

Activation progresses according to logic/topological depth, but each eligible node can probabilistically fire.

## Pulse

Choose a graph region/node and propagate outward.

## Scatter

Probabilistically sample nodes globally.

Initial implementation only needs Walk and deterministic Signal Wave, but do not architect in a way that prevents the others.

---

# Probability attractor

Eventually allow the user to place an XY attractor over the graph.

Nodes closer to it gain higher traversal probability.

Conceptually:

```cpp
float distance = node.position.getDistanceFrom (attractor);
weight *= std::exp (-distance * attractionStrength);
```

This lets the performer "pull" the melody through different parts of the graph without mutating the graph.

Please prepare the probability system so such a multiplier can be added easily.

---

# Rhythm variation

Randomness should not only affect pitch.

Allow rhythm to vary probabilistically as well.

Potential rhythmic values:

```text
1/4
1/8
dotted 1/8
1/16
triplets
rests
ties
```

Graph features can bias rhythm.

For example:

```text
high fanout → shorter notes
low fanout → longer notes
deep recursion → faster subdivisions
merge points → longer sustain
```

Keep topology influential so different graphs have different rhythmic personalities.

---

# Important realtime rule

Never compile, parse, grow, allocate graph nodes, or run force-layout simulation from the audio thread.

Do NOT perform operations like:

```cpp
graph.expandCell();
parser.parse();
layout.relax();
```

inside `processBlock()`.

Preferred architecture:

```text
UI / worker thread
        |
        | graph events / immutable musical events
        v
lock-free FIFO
        |
        v
audio thread
        |
        +--> MIDI
        +--> optional synth later
```

The audio callback should consume small preallocated immutable events only.

No:

```text
heap allocation
locks
filesystem calls
graph mutation
parser work
layout work
```

inside the audio callback.

Use JUCE-safe realtime design.

---

# DAW timing

Musical scheduling should eventually use host timing / PPQ.

Examples:

```text
growth rate:
1/4
1/8
1/16
1/32
triplets
```

Logic wave:

```text
logic depth 0 → beat N
logic depth 1 → beat N + step
logic depth 2 → beat N + 2*step
```

Keep internal musical timing separate from GUI animation timing.

Sample-accurate MIDI event offsets should be generated in `processBlock()` from already prepared state/events.

---

# JUCE UI concept

Eventually aim for something like:

```text
+----------------------------------------------------------+
| Morpho                                                   |
+-------------------------+--------------------------------+
|                         |                                |
| @morpho                 |              O                 |
| def tree(a):            |            /   \               |
|   x,y = SPLIT(a)        |          O       O             |
|   ...                   |        /  \     /  \           |
|                         |       O    O   O    O           |
|                         |                                |
+-------------------------+--------------------------------+
| Root C3     Scale Dorian      Octaves 4                 |
| Pitch: X Position       Register: Recursion Depth        |
| Velocity: Fanout        MIDI Channel: Cell Type          |
|                                                          |
| Growth 1/16   Traversal BFS     [GROW] [RESET]           |
|                                                          |
| LIVE: Walk                                                |
| Voices 3    Variation 42%    Momentum 65%                |
| Novelty 70%  Density 55%      Seed 381729 [Lock]         |
+----------------------------------------------------------+
```

Do not prioritize beautiful GUI work until the engine and MIDI behavior are tested.

---

# State serialization

All important plugin state should eventually serialize through JUCE:

```text
Morpho source code
root design
input signature
root note
scale
mapping parameters
growth mode
growth rate
LIVE mode
voices
entropy
momentum
density
seed
seed lock state
```

---

# Suggested implementation milestones

Please work incrementally.

## Milestone 1 — Core parser

Create:

```text
Tokenizer
Parser
AST/SSA
SourceSpan
basic test cases
```

Parse a simple MorphoHDL source file successfully.

No JUCE required.

## Milestone 2 — Graph compiler

Implement:

```text
input/output cells
LUT cells
Morpho cells
bus width inference
SPLIT
CAT
basic slicing
recursive specialization
fallback
```

Be able to compile something like `ripple_adder`.

## Milestone 3 — Growth engine

Create a root graph containing one hierarchical Morpho cell.

Expand it incrementally.

Expose:

```cpp
expandNext()
expandLargest()
isFullyExpanded()
```

and emit `GrowthEvent`s.

Add deterministic unit tests.

## Milestone 4 — Logic timing

Implement topological arrival-time calculation.

Verify logic depth on known small circuits.

## Milestone 5 — Native layout

Port the existing C layout backend to native C/C++.

Generate stable node coordinates.

Do not focus on polished drawing yet.

## Milestone 6 — Minimal JUCE plugin

Build a JUCE MIDI-effect/instrument plugin.

Allow:

```text
load/enter Morpho source
compile
grow
reset
```

During growth:

```text
one division/new-cell event → one MIDI note
```

Basic mapping:

```text
recursion depth → octave
structural or X position → scale degree
```

## Milestone 7 — Host synchronization

Add:

```text
tempo sync
PPQ scheduling
1/4, 1/8, 1/16 etc.
sample-accurate MIDI offsets
```

Keep graph work off the audio thread.

## Milestone 8 — Finished graph LIVE mode

Implement a single random walker.

Add:

```text
seed
seed lock
variation/entropy
repetition penalty
momentum
```

Same graph should generate multiple performances while preserving a recognizable topology-derived musical identity.

## Milestone 9 — Polyphonic walkers

Add:

```text
1–8 walkers
density
optional collision handling
```

## Milestone 10 — Visual editor

Render graph and source.

Add:

```text
pan
zoom
node animation
currently executing source statement
active walker positions
logic wave
```

---

# Development priorities

Please prioritize in this order:

1. Correct graph semantics
2. Clean architecture
3. Testability
4. Realtime safety
5. Musical usefulness
6. Performance
7. UI polish

Avoid premature UI complexity.

---

# Tests

Create automated tests wherever practical.

Important test cases:

```text
tokenizer/parser correctness
numeric literals
SPLIT
CAT
slice/index operations
bus-width behavior
recursive fallback
graph expansion
parent-child ancestry
deterministic graph growth
logic-depth calculation
same seed = same random walk
different seed = different random walk
repetition penalty behavior
scale mapping
negative scale degrees
audio-thread code performs no dynamic allocation where avoidable
```

Compare selected graph outputs against the original Morpho implementation when useful.

---

# Code quality

Use modern C++17 or C++20.

Prefer:

```text
RAII
strong types/enums
small focused classes
clear ownership
const correctness
std::span where appropriate
preallocation in realtime paths
unit tests
minimal global state
```

Avoid creating an overly abstract framework before the first graph successfully grows and produces MIDI.

---

# First concrete task

Start by inspecting the reference repository and documenting the minimum behavioral subset required to compile and incrementally expand one existing recursive design such as `ripple_adder`.

Then implement the first vertical slice:

```text
Morpho source
    ↓
C++ parser
    ↓
C++ compiler
    ↓
root graph
    ↓
incremental growth
    ↓
GrowthEvent
    ↓
simple note mapping
    ↓
JUCE MIDI output
```

Do not attempt the complete polished plugin at once.

The first meaningful success condition is:

> A native C++ JUCE plugin can compile one MorphoHDL design, grow it incrementally, and emit musically quantized MIDI notes for each structural division, with recursion depth affecting register and a selectable scale controlling pitch.

After that, implement the fixed-graph stochastic LIVE player.

The overall product concept is:

> BUILD creates the graph and lets the user hear the structure being born. LIVE keeps the completed graph fixed and continuously explores it using reproducible stochastic traversal.

The key musical principle is:

> The sound is not merely accompanying the graph. The graph structure itself determines the probability space from which the music emerges.