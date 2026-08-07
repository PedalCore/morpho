# Build a Native C++/JUCE Morpho-Inspired Generative Music System

## Goal

Build a **native C++ JUCE plugin** inspired by MorphoHDL:

https://github.com/paradigms-of-intelligence/morpho

Do not embed JavaScript, Python, WebAssembly, Chromium, a browser, or a web view.

Reimplement the important Morpho concepts in native C++ and extend them into a generative musical instrument.

The system should eventually support three related behaviours:

1. **BUILD** — hear a graph/circuit being recursively constructed.
2. **LIVE** — once grown, keep the graph fixed and explore it stochastically to generate music.
3. **LIVING GRAPH** — optionally never finish growing at all; continuously grow, age, collapse, prune and regenerate while remaining bounded in memory.

The core artistic principle is:

> The sound should not merely accompany the graph. The graph structure itself should determine the musical probability space.

---

# 1. Core Morpho concept

MorphoHDL describes circuits/graphs using recursive structural rules.

A hierarchical cell is replaced by smaller cells. Those cells can themselves be expandable. The process repeats until nothing remains expandable.

The original audiovisual idea maps this process to sound:

- a cell division creates a note
- recursion depth affects register
- position in the layout affects melodic pitch
- once growth completes, a signal wave travels through the final graph
- logic/topological depth determines when cells are activated
- the audible result represents the actual construction and topology

Recreate these ideas natively in JUCE, then extend them substantially.

---

# 2. Architecture

Keep the system separated into two major layers.

## MorphoCore

Pure C++, with no JUCE dependencies.

Suggested modules:

```text
MorphoCore/
    Tokenizer
    Parser
    AST
    SSA / IR
    SourceSpan

    Compiler
    PrimitiveOperations

    CompiledGraph
    CellStore
    PinStore
    NetStore

    GraphGrower
    LogicTiming

    LayoutEngine

    GrowthEvent
    LifecycleEvent
    GraphSnapshot

    LivingGraph
```

## MorphoJUCE

JUCE-specific code:

```text
MorphoJUCE/
    PluginProcessor
    PluginEditor

    CodeEditor
    GraphComponent

    GrowthController
    GrowthScheduler

    ScaleMapper
    MidiGenerator

    GenerativePlayer
    Walker
    ProbabilityModel

    optional internal SynthEngine
```

The graph/compiler must remain usable independently of JUCE.

---

# 3. Reference implementation

Study the original repository carefully.

Important files include approximately:

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

Important existing concepts include:

- a deliberately small Python-like custom language
- custom parser rather than a Python runtime
- SSA-like representation
- recursive graph materialisation
- specialization based on bus widths
- flat Struct-of-Arrays graph representation
- hierarchical Morpho cells
- LUT cells
- input/output cells
- nets and pins
- incremental graph expansion
- breadth-first and largest-first growth
- optimization / constant propagation / DCE
- logic timing via topological traversal
- force-directed layout
- existing C layout backend currently compiled through WASM

Do not translate JavaScript mechanically. Reimplement behaviour idiomatically in C++17/20.

---

# 4. Morpho language subset

Initially support enough Morpho syntax to compile existing recursive examples.

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

Support approximately:

```text
decimal integers
binary integers
hex integers

identifiers
function calls

single assignment
multiple assignment

def
return

@morpho
@morpho(fallback=...)

LUT declarations

indexing
slicing

keyword overrides where required
```

Initial primitive operations:

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

# 5. Preserve source locations

Every parsed/compiled instruction should retain its source location.

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

This is required so the JUCE editor can later highlight the rule/statement being executed while the structure grows.

Growth events should be able to point back to their originating source instruction.

---

# 6. Graph representation

Prefer a flat cache-efficient Structure-of-Arrays design similar to the original implementation.

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

Maintain equivalent stores for:

```text
Pins
Nets
LUT metadata
hierarchical cell templates
```

Core cell types:

```text
INPUT
OUTPUT
LUT
MORPHO
```

---

# 7. Incremental graph growth

Expose growth as explicit discrete operations.

For example:

```cpp
bool expandNext();
bool expandLargest();
bool isFullyExpanded();
```

A successful expansion should create a `GrowthEvent`.

Example:

```cpp
struct GrowthEvent
{
    CellId parentCell;

    std::vector<CellId> createdCells;

    int recursionDepth {};

    SourceSpan source;

    uint64_t generation {};
};
```

Growth should be deterministic for identical:

```text
source
input signature
growth mode
random seed, where randomness is used
```

Initially support:

```text
breadth-first expansion
largest-expandable-cell expansion
```

---

# 8. Layout

Port/reuse the existing layout backend from:

```text
graphs_engine/src/main.c
```

as native C or C++.

Do not use WASM.

The layout should eventually support:

```text
node positions
edges

incremental layout relaxation

pan
zoom

smooth interpolation when a node divides

highlighting active cells
highlighting ancestry

walker positions

signal-wave animation
```

Layout correctness/performance matters before visual polish.

---

# 9. Logic timing

Once a graph reaches a given stable state, calculate signal arrival timing.

Each active cell should expose values equivalent to:

```text
firstTime
lastTime
logicDepth
```

Use an iterative Kahn/topological algorithm rather than recursive traversal.

This information drives both visual animation and music.

---

# 10. BUILD musical mode

During normal growth, every structural division can create music.

Initial mapping idea:

```text
event               → cell division
time                → growth clock
register            → recursion depth
pitch               → spatial or structural position
velocity            → fanout / child count
pan / MIDI CC       → X position
channel / timbre    → cell type
accent/articulation → source operation/rule
```

Provide configurable mappings.

---

# 11. Scale system

Use a generic interval-based scale model.

Example:

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
Natural Minor
Dorian
Phrygian
Mixolydian
Pentatonic
Whole Tone
Custom
```

Allow negative and positive degrees to extend naturally across octaves.

Design so microtonal mapping can be added later.

---

# 12. Deterministic topology-to-music mapping

Try to keep each graph musically recognisable.

Suggested mappings:

```text
recursion depth → octave/register
X position → scale degree
structural sibling position → alternate pitch source
logic depth → rhythmic placement
fanout → velocity/accent
cell type → channel/timbre
```

Do not continuously derive pitch from a force-layout position that is still moving.

Instead either:

- snapshot/quantize the layout position when the musical event occurs, or
- provide a deterministic structural position independent of force layout.

---

# 13. Host timing

Eventually synchronize growth and LIVE playback to the DAW.

Support subdivisions such as:

```text
1/4
1/8
1/16
1/32
triplets
dotted values
```

Keep:

```text
musical timing
GUI animation timing
graph simulation timing
```

separate.

MIDI events should ultimately be sample-accurate.

---

# 14. Critical realtime requirement

Never parse, compile, expand the graph, mutate complex graph structures, run layout simulation, or allocate heavily inside `processBlock()`.

Do not perform:

```cpp
parser.parse();
graph.expandCell();
layout.relax();
```

from the audio thread.

Use an architecture similar to:

```text
UI / worker / graph thread
        |
        | prepared immutable events
        v
lock-free FIFO / realtime-safe queue
        |
        v
audio thread
        |
        +--> MIDI
        +--> optional synth
```

The audio callback should avoid:

```text
locks
heap allocation
filesystem access
graph mutation
parsing
layout work
```

---

# 15. LIVE mode: fixed graph, non-deterministic performance

When standard BUILD mode finishes, the graph should optionally remain fixed.

Do not alter its topology.

Instead, create a separate stochastic musical interpreter.

Conceptually:

```cpp
class GenerativePlayer
{
public:
    void reset();
    void advance();

private:
    const Graph* graph {};

    Scale scale;

    RandomGenerator rng;

    std::vector<Walker> walkers;

    float variation {};
    float momentum {};
    float repetition {};
    float novelty {};
    float density {};
};
```

The graph should know nothing about music.

---

# 16. Random walkers

A core LIVE playback mode should use stochastic graph traversal.

Example:

```text
             A
          /     \
         B       C
       /  \     / \
      D    E   F   G
```

Different performances might traverse:

```text
A → B → D
A → C → G
A → B → E
A → C → F
```

Each visited node can emit MIDI.

The same graph should be capable of generating many different performances.

---

# 17. Weighted stochastic traversal

Do not default to naive uniform randomness.

Each candidate edge should receive a score.

Conceptually:

```text
weight =
    base topology weight
  × direction preference
  × spatial preference
  × momentum
  × cell-type preference
  × recent-history penalty
  × user bias
```

Then perform weighted stochastic sampling.

---

# 18. Variation / entropy control

Expose a parameter such as:

```text
Variation
Entropy
Chaos
```

Behaviour:

```text
0%
mostly deterministic/highest scoring route

30%
small deviations

60%
strong variation

100%
near-uniform exploration
```

A softmax/temperature approach is appropriate.

The goal is controlled uncertainty rather than arbitrary randomness.

---

# 19. Musical memory

Pure randomness tends to sound incoherent.

Each walker/player should remember things such as:

```text
recent node IDs
recent pitches
recent intervals
previous graph direction
recent rhythmic values
```

Use this memory to support:

```text
repetition penalty
novelty
momentum
direction bias
phrase continuity
```

Example:

```cpp
if (recentlyVisited(candidate))
    weight *= repetitionPenalty;
```

Momentum can favour continuing in approximately the same geometric or topological direction.

---

# 20. Random seed and reproducibility

Use seeded PRNGs.

Allow:

```text
New Seed
Manual Seed
Lock Seed
Unlock Seed
```

Requirement:

```text
same graph
+ same seed
+ same parameters
= same performance
```

Unlocked seed should permit a new performance each run.

This is crucial for DAW workflows.

---

# 21. Multiple walkers / polyphony

Support multiple simultaneous walkers.

Suggested initial range:

```text
1–8 voices
```

Each walker gets independent state.

Possible later behaviours:

```text
walker collisions
chords
accents
voice merging
walker splitting
walker reversal
walker spawning
```

Architect for these without requiring all of them initially.

---

# 22. Other LIVE playback modes

Design the player so additional traversal modes can eventually exist:

```text
Walk
Wave
Pulse
Scatter
```

Definitions:

### Walk

Stochastic agents traverse graph edges.

### Wave

Activity moves according to logic/topological depth, with probabilistic triggering.

### Pulse

A selected graph region/node emits an outward activation pulse.

### Scatter

Random/probabilistic nodes are selected globally.

Initial implementation only needs:

```text
Walk
Deterministic Signal Wave
```

---

# 23. Probability attractors

Eventually allow a user-controlled XY attractor over the rendered graph.

Nodes nearer the attractor receive additional traversal probability.

Conceptually:

```cpp
float distance = node.position.getDistanceFrom(attractor);

weight *= std::exp(
    -distance * attractionStrength
);
```

This creates a performance control for pulling melodic activity through different areas of the graph.

Prepare the probability architecture so this can be added as just another weight multiplier.

---

# 24. Rhythmic stochasticity

Variation should not exist only in pitch/path choice.

Allow probabilistic rhythm selection.

Examples:

```text
1/4
1/8
dotted 1/8
1/16
triplet
rest
tie
```

Topology can bias rhythm.

For example:

```text
high fanout → shorter notes
low fanout → longer notes
deep recursion → faster subdivisions
merge points → longer sustain
```

Different graphs should develop recognisable rhythmic tendencies.

---

# 25. New major feature: LIVING GRAPH

Do not require every Morpho process to eventually finish.

Add a second graph-growth model:

> The process can continue indefinitely while the resident graph stays bounded.

This is an **infinite growth process with finite memory**.

Conceptually:

```text
grow
 ↓
age
 ↓
prune / collapse
 ↓
regrow
 ↓
age
 ↓
prune
 ↓
...
forever
```

Do not call this an actually infinite in-memory graph.

Call the concept something like:

```text
Living Graph
Bounded Infinite Growth
Continuous Morpho
```

---

# 26. Bounded resident graph

The Living Graph should maintain a configurable resource/cell budget.

Example:

```cpp
size_t maxLivingCells = 2048;
```

The growth process may continue indefinitely, but the currently resident graph should remain under fixed limits.

Conceptually:

```cpp
growSomeCells();

while (livingCellCount > maxLivingCells)
    pruneLowestPriorityRegion();
```

Support configurable limits for:

```text
cells
nets
history
possibly layout particles
```

---

# 27. Progressive pruning

Do not simply delete the oldest cell.

Each node/region should have a survival score.

Possible factors:

```text
recency
age
recent musical activity
connectivity
fanout
distance from growth frontier
walker presence
structural importance
random survival term
```

Example:

```cpp
float survivalScore (const LivingCell& c)
{
    return
          recencyWeight      * c.recency
        + activityWeight     * c.recentActivity
        + connectivityWeight * c.connectivity
        + frontierWeight     * c.frontierProximity
        + walkerWeight       * c.walkerPresence
        + randomWeight       * rng.nextFloat();
}
```

Prune/collapse lowest-scoring eligible cells or regions.

---

# 28. Memory parameter

Expose a musical/biological control named something like:

```text
Memory
```

Low Memory:

```text
old structures disappear quickly
graph reinvents itself rapidly
less thematic persistence
```

High Memory:

```text
old regions persist longer
new growth accumulates around older structures
greater motif continuity
```

This should influence decay/pruning behaviour.

---

# 29. Safe pruning

Morpho is not just an arbitrary tree: it is a graph/circuit with nets and dependencies.

Never casually delete a node if doing so leaves invalid connectivity.

Initially support safe operations such as:

### Leaf pruning

Delete structurally irrelevant leaves.

### Dead region removal

Remove regions that no longer contribute to active topology/music.

### Subtree/region collapse

Prefer collapsing an old detailed region back into a proxy/summary cell rather than destructively deleting arbitrary internal nodes.

---

# 30. Reversible collapse / folding

A key Living Graph idea:

> Pruning can often mean folding rather than destruction.

A complex region:

```text
            O
          / | \
         O  O  O
        / \   / \
       O   O O   O
```

may collapse to:

```text
            ●
```

The summary/proxy node should preserve enough information to maintain external graph connectivity.

Possible retained metadata:

```text
external input mapping
external output mapping
historical depth
previous size
cell type statistics
musical statistics
source definition
random/mutation state
```

This region can later be re-expanded.

---

# 31. Regrowth

Collapsed cells/regions should be capable of growing again.

Initial behaviour may simply regenerate the original structure.

Later add mutation:

```text
original expansion

    A
   / \
  B   C

regrowth

    A
   /|\
  B D C
```

The mutation probability should be configurable.

This moves the system toward a stochastic L-system / graph grammar rather than a one-shot compiler visualisation.

---

# 32. Growth-rule mutation

Eventually allow more than one possible expansion outcome.

Example:

```text
Rule A 50%
Rule B 30%
Rule C 15%
Rule D  5%
```

Rule choice may be affected by:

```text
cell type
cell age
energy
location
walker activity
parent state
musical state
MIDI input
seed
```

Do not require this for the earliest Living Graph version, but design the lifecycle architecture to permit it.

---

# 33. Cell lifecycle

Add lifecycle metadata to graph cells.

Conceptually:

```cpp
struct LivingCell
{
    CellId id;

    float age {};
    float energy { 1.0f };

    float recentActivity {};

    bool frontier {};

    uint64_t protectedUntilGeneration {};
};
```

Potential states:

```text
Alive
Protected
Dying
Collapsed
Dead
```

---

# 34. Energy / metabolic model

Cells may gradually lose energy.

Example:

```cpp
cell.energy -= decayRate;
```

Energy can be restored when:

```text
a walker visits
signal passes through
the cell contributes children
the cell is near an attractor
the cell participates in currently active music
```

Example:

```cpp
if (walkerVisited)
    cell.energy += walkerEnergy;

if (signalPassed)
    cell.energy += signalEnergy;
```

When energy falls below a threshold:

```cpp
if (cell.energy <= 0.0f)
    markForCollapseOrPrune(cell);
```

This creates a graph ecology rather than a simple timeout.

---

# 35. Resource competition

Optionally implement a global resource/energy budget.

Example:

```text
totalGraphEnergy = 1000
```

More living cells mean less available resource per cell.

This can create emergent cycles:

```text
growth explosion
→ resource pressure
→ collapse
→ recovery
→ growth
```

This may be musically valuable because it creates long-form macrostructure without explicitly sequencing sections.

---

# 36. Music affects survival

One particularly important feedback idea:

> Musical activity should be capable of influencing which graph structures survive.

If walkers repeatedly traverse a region, its nodes can receive survival energy.

This creates a feedback loop:

```text
graph topology
     ↓
music probability
     ↓
walker activity
     ↓
cell survival
     ↓
future topology
     ↓
future music
```

The graph therefore develops according to its own musical behaviour.

---

# 37. Protect actively used cells

Never prune a node while an active musical walker/event depends on it.

For example:

```cpp
cell.protectedUntilGeneration =
    currentGeneration + protectionTicks;
```

Pruning must skip protected nodes.

Protect:

```text
current walker positions
recently traversed nodes
currently sounding event sources
temporarily important signal paths
```

---

# 38. Lifecycle sonification

Living Graph lifecycle events should themselves be available musically.

Define events such as:

```cpp
enum class LifecycleEventType
{
    Born,
    Expanded,
    Activated,
    Deactivated,
    Collapsing,
    Collapsed,
    Regrown,
    Died
};
```

Possible musical interpretations:

| Lifecycle event | Musical interpretation |
|---|---|
| Born | note onset |
| Expanded | chord / arpeggio |
| WalkerVisit | melodic note |
| Activated | accent |
| Collapsing | descending interval |
| Collapsed | low-energy note or release |
| Regrown | motif variation |
| Died | final release/percussive event |

The mappings should eventually be user-configurable.

---

# 39. BUILD vs LIVE vs LIVING GRAPH

Treat these as related but distinct operating models.

## BUILD

```text
finite recursive construction
→ sonify birth/division
→ graph eventually reaches final state
```

## LIVE

```text
fixed completed graph
→ topology remains unchanged
→ stochastic walkers reinterpret it forever
```

## LIVING GRAPH

```text
graph continually grows
→ walkers traverse it
→ old regions age
→ pruning/collapse occurs
→ regions can regrow
→ topology continually evolves
```

The user should eventually be able to choose among these behaviours.

---

# 40. Suggested Living Graph controls

Potential plugin parameters:

```text
Growth Rate
Decay Rate
Memory
Mutation
Resource Level
Competition
Max Cells

Variation
Momentum
Novelty
Density

Walker Count

Attractor X
Attractor Y

Seed
Seed Lock
```

These should be automatable through JUCE where practical.

---

# 41. DAW automation as graph ecology

Host automation should eventually be capable of driving:

```text
Growth
Decay
Memory
Mutation
Resources
Entropy
Density
```

Example musical behaviour:

```text
Growth ↑
→ structure rapidly expands
→ note density increases

Decay ↑
→ regions collapse
→ graph thins
→ musical texture becomes sparse
```

This should create intuitive audiovisual macro-control.

---

# 42. LivingGraph architecture

Add a layer above the basic graph grower.

Conceptually:

```cpp
class LivingGraph
{
public:
    void tick();

private:
    void grow();
    void updateAge();
    void distributeEnergy();
    void updateActivity();

    void scoreSurvival();

    void choosePruningCandidates();
    void collapseRegions();
    void removeDeadCells();

    void regrowEligibleRegions();

    CompiledGraph graph;
    GraphGrower grower;

    size_t maxLivingCells = 2048;
};
```

`LivingGraph` should produce immutable lifecycle events that the musical system can consume.

---

# 43. JUCE UI concept

Eventually aim toward a UI containing:

```text
+-----------------------------------------------------------+
| Morpho                                                    |
+--------------------------+--------------------------------+
|                          |                                |
| @morpho                  |             O                  |
| def tree(a):             |           /   \                |
|   x,y = SPLIT(a)         |         O       O              |
|   ...                    |       /  \     /  \            |
|                          |      O    O   O    O            |
|                          |                                |
+--------------------------+--------------------------------+
| Root C3       Scale Dorian       Octaves 4               |
| Pitch X       Register Depth      Velocity Fanout          |
|                                                           |
| Mode: LIVING GRAPH                                        |
|                                                           |
| Growth 65%     Decay 40%       Memory 70%                 |
| Mutation 20%   Resources 55%   Max Cells 2048             |
|                                                           |
| Walkers 3      Variation 42%   Momentum 65%               |
| Novelty 70%    Density 55%     Seed 381729 [Lock]         |
|                                                           |
| [GROW] [PAUSE] [RESET] [NEW SEED]                         |
+-----------------------------------------------------------+
```

Do not prioritize visual polish until the graph/compiler/music engines work correctly.

---

# 44. Plugin state serialization

Eventually serialize:

```text
Morpho source
selected root design
input signature

scale
root note
pitch mapping
velocity mapping
growth rate
growth mode

LIVE mode settings
walker count
variation
momentum
novelty
density

Living Graph enabled state
growth
decay
memory
mutation
resource level
max cell count

seed
seed lock
```

---

# 45. Development milestones

Work incrementally.

Do not attempt the complete plugin in one pass.

## Milestone 1 — Parser

Implement:

```text
Tokenizer
Parser
AST
SSA/IR
SourceSpan
tests
```

Parse a small existing Morpho example.

---

## Milestone 2 — Compiler

Implement enough graph compilation for:

```text
input/output cells
LUTs
Morpho cells
bus width inference
SPLIT
CAT
basic slicing
fallback
recursive specialization
```

Compile a design such as `ripple_adder`.

---

## Milestone 3 — Incremental growth

Implement:

```text
expandNext()
expandLargest()
isFullyExpanded()
GrowthEvent
```

Verify ancestry and resulting graph topology.

---

## Milestone 4 — Logic timing

Implement iterative topological timing.

Test logic depth on known small circuits.

---

## Milestone 5 — Native layout

Port the existing C backend.

Produce stable positions.

---

## Milestone 6 — Minimal JUCE MIDI plugin

Vertical slice:

```text
Morpho source
    ↓
C++ parser
    ↓
compiler
    ↓
graph
    ↓
incremental growth
    ↓
GrowthEvent
    ↓
scale mapping
    ↓
JUCE MIDI
```

Success criterion:

> A native JUCE plugin compiles one Morpho design and emits quantized MIDI notes as its cells divide.

---

## Milestone 7 — Host sync

Add:

```text
tempo sync
PPQ handling
note subdivisions
sample-accurate MIDI offsets
```

---

## Milestone 8 — Fixed-graph LIVE mode

Implement one random walker.

Add:

```text
seed
seed lock
variation
momentum
repetition penalty
novelty
```

Requirement:

> Same graph, same seed and same parameters produce the same performance.

---

## Milestone 9 — Polyphonic LIVE mode

Add:

```text
1–8 walkers
density
independent walker state
```

---

## Milestone 10 — Living Graph prototype

Add a bounded lifecycle model:

```text
continuous growth
age
energy
safe leaf pruning
cell protection
fixed max cell count
LifecycleEvent
```

Do not implement mutation yet.

Success criterion:

> The system can run indefinitely while staying within a bounded cell budget.

---

## Milestone 11 — Reversible collapse

Add:

```text
subtree/region collapse
summary proxy cells
safe external input/output preservation
possible later re-expansion
```

---

## Milestone 12 — Musical survival feedback

Allow walker activity to restore node energy and affect survival.

Requirement:

> Regions used by the generated music are statistically more likely to persist.

---

## Milestone 13 — Regrowth and mutation

Add:

```text
collapsed-region regrowth
multiple growth alternatives
mutation probability
```

---

## Milestone 14 — Visual editor

Add:

```text
graph rendering
pan
zoom
smooth growth animation

current source-line highlighting

walker visualization

growth/frontier state
dying/collapsing state

logic signal wave
```

---

# 46. Testing priorities

Automate tests for:

```text
tokenizer correctness
parser correctness

decimal/binary/hex literals

SPLIT
CAT
INDEX
SLICE

bus-width behaviour
fallback behaviour

recursive specialization

graph expansion
parent/child ancestry

logic-depth calculation

deterministic graph growth

same seed = same traversal
different seeds = different traversal

weighted stochastic choice
repetition penalty
momentum

scale degree mapping
negative scale degree mapping

Living Graph remains under max cell count

protected cells are not pruned

leaf pruning preserves graph validity

collapse preserves external connections

regrowth restores valid topology

long-running Living Graph does not grow memory indefinitely
```

Where useful, compare the C++ implementation against the original Morpho repository.

---

# 47. Development priorities

Prioritize:

1. Graph semantic correctness
2. Clear ownership and architecture
3. Automated tests
4. Realtime audio safety
5. Musical usefulness
6. Bounded memory behaviour
7. Performance
8. UI polish

Avoid premature abstraction.

Get one complete vertical slice functioning before building generalized frameworks.

---

# 48. Code quality

Use modern C++17 or C++20.

Prefer:

```text
RAII
strong enum/classes
clear ownership
const correctness
small focused modules

std::span where appropriate

flat contiguous stores for graph hot paths

preallocation for realtime paths

minimal shared mutable state
```

Avoid unnecessary runtime allocation in realtime systems.

---

# 49. First task for the coding agent

First inspect the original Morpho repository and document the smallest behavioural subset required to compile and incrementally expand an existing recursive example such as `ripple_adder`.

Then implement the first vertical slice:

```text
MorphoHDL source
      ↓
native C++ parser
      ↓
native C++ compiler
      ↓
root hierarchical graph
      ↓
incremental graph expansion
      ↓
GrowthEvent stream
      ↓
simple recursion-depth / scale mapping
      ↓
JUCE MIDI output
```

Do not begin with the Living Graph or polished visualization.

Once this basic pipeline works and is covered by tests:

1. implement fixed-graph stochastic LIVE playback
2. then implement bounded continuous Living Graph behaviour
3. then reversible collapse/regrowth
4. then musical-survival feedback and mutation

At the end of every milestone:

- build the project
- run all available tests
- document what is implemented
- document deviations from Morpho semantics
- do not proceed while known foundational tests are failing

---

# Product concept

The finished system should feel like an evolving audiovisual instrument with three temporal states.

### BUILD

> Hear a structure being born.

### LIVE

> Freeze the structure and continuously explore its musical possibilities.

### LIVING GRAPH

> Let the structure remain alive indefinitely: growing, consuming resources, being played, forgetting old regions, collapsing and regrowing.

The most important conceptual feedback loop is:

```text
graph structure
      ↓
musical probabilities
      ↓
generated activity
      ↓
cell energy / survival
      ↓
future graph structure
      ↓
future music
```

The result should therefore be more than a random sequencer.

It should behave like a **bounded generative graph ecology whose musical activity helps determine its own future structure**.