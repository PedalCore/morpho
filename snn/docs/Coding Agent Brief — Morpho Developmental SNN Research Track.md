# Coding Agent Brief — Morpho Developmental SNN Research Track

## Project context

We are already pursuing a separate project to build a native C++/JUCE generative music plugin inspired by MorphoHDL:

https://github.com/paradigms-of-intelligence/morpho

That plugin project should continue independently.

This task is a **parallel experimental research project** exploring the relationship between:

- Morpho-style recursive graph development
- Spiking Neural Networks
- structural plasticity
- recursive developmental encoding
- iterative lifetime development
- progressive pruning
- bounded continuous growth
- activity-dependent survival
- generative/evolutionary systems
- eventual musical applications

For this research track, **retain the web/browser architecture**.

Do NOT port this experiment to JUCE or C++ at this stage.

JavaScript/WebGL/WebAssembly/browser tooling is acceptable and desirable because the goal is rapid experimentation and visual inspection.

The research findings may later inform the C++ plugin.

---

# 1. Core hypothesis

Treat Morpho not merely as a circuit description language, but as a possible **developmental encoding for neural structures**.

The central analogy is:

```text
Morpho program
     ↓
developmental rules / genotype
     ↓
recursive structural growth
     ↓
spiking neural graph / phenotype
     ↓
neural activity
     ↓
development feedback
     ↓
growth / pruning / structural change
```

Morpho describes **how the network develops**.

The SNN describes **how the developed network behaves dynamically**.

These systems should remain conceptually separate but communicate through an explicit feedback layer.

---

# 2. Do not replace the existing Morpho web implementation

Use the existing Morpho repository as the experimental starting point where practical.

Important existing concepts include:

```text
parser
compiler
recursive graph expansion
cell hierarchy
parent relationships
graph layout
incremental growth
signal visualization
force layout
C/WASM graph backend
```

Do not rewrite the whole project merely to begin the experiment.

Prefer extending it experimentally.

Create new modules rather than deeply entangling neural simulation with the existing circuit compiler.

---

# 3. Separate the system into three layers

The architecture should conceptually become:

```text
┌─────────────────────────────┐
│ Morpho Developmental System │
│                             │
│ grow                        │
│ divide                      │
│ collapse                    │
│ prune                       │
│ regrow                      │
└──────────────┬──────────────┘
               │
       structural changes
               │
               ▼
┌─────────────────────────────┐
│ Neural Graph                │
│                             │
│ neurons                     │
│ synapses                    │
│ delays                      │
│ populations                 │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Spiking Engine              │
│                             │
│ membrane state              │
│ spikes                      │
│ refractory state            │
│ synaptic events             │
│ learning                    │
└──────────────┬──────────────┘
               │
       activity statistics
               │
               ▼
┌─────────────────────────────┐
│ Development Feedback        │
│                             │
│ survival                    │
│ growth pressure             │
│ pruning pressure            │
│ homeostasis                 │
│ structural plasticity       │
└──────────────┬──────────────┘
               │
               └────────────► Morpho
```

Avoid making one giant `Graph` class responsible for everything.

---

# 4. Three timescales

The system should explicitly operate on different timescales.

## Fast timescale — neural dynamics

Approximately milliseconds.

Contains:

```text
spikes
membrane integration
refractory periods
synaptic transmission
synaptic delays
```

## Medium timescale — plasticity

Contains:

```text
firing-rate statistics
STDP or Hebbian updates
homeostatic adaptation
recent correlation
activity traces
```

## Slow timescale — development

Contains:

```text
Morpho expansion
new neurons
new synapses
structural pruning
subgraph collapse
regrowth
structural mutation
```

Do not modify topology on every individual spike.

Instead run many neural simulation steps before evaluating development.

Conceptually:

```text
simulate SNN
     ↓
measure activity
     ↓
development epoch
     ↓
grow/prune
     ↓
simulate SNN
     ↓
...
```

---

# 5. Recursive in space, iterative in time

This distinction is central to the experiment.

## Recursive development

Morpho recursion describes hierarchy in space:

```text
network
   ↓
region
   ↓
population
   ↓
microcircuit
   ↓
neuron
```

A developmental rule may recursively generate increasingly fine structure.

## Iterative development

The resulting organism continues changing over time:

```text
epoch 1
↓
epoch 2
↓
epoch 3
↓
epoch 4
↓
...
```

During those epochs the graph may:

```text
grow
prune
collapse
regrow
rewire
change synaptic weights
```

The desired system is therefore:

> recursively generated in space and iteratively developed in time.

---

# 6. Initial neuron model

Do not begin with a biologically complicated model.

Implement a simple deterministic spiking neuron first.

A basic Leaky Integrate-and-Fire neuron is sufficient.

Conceptually maintain:

```text
membrane potential
threshold
reset voltage
leak
refractory period
```

Suggested structure:

```javascript
class Neuron {
    id;

    membrane;
    threshold;
    resetPotential;

    refractoryUntil;

    incoming;
    outgoing;

    activity;
}
```

Keep neural state separate from visualization state.

---

# 7. Explicit synapses

Represent synapses independently.

For example:

```javascript
class Synapse {
    source;
    target;

    weight;
    delay;

    enabled;
}
```

Support:

```text
positive/excitatory weights
negative/inhibitory weights
transmission delays
```

Recurrent connections must be legal.

This is important because unlike combinational circuits, an SNN can contain:

```text
A → B
↑   ↓
D ← C
```

Therefore do not depend upon a global DAG/topological ordering for neural simulation.

---

# 8. Event-driven spike propagation

Use explicit spike events.

Conceptually:

```javascript
{
    targetNeuron,
    arrivalTime,
    weight,
    sourceNeuron
}
```

When neuron A spikes:

```text
A fires
  ↓
inspect outgoing synapses
  ↓
schedule spike arrival events
  ↓
delay expires
  ↓
target neuron receives impulse
```

Use a priority queue or suitable time-ordered event structure if practical.

For early prototypes, fixed timestep simulation is also acceptable if substantially simpler.

Correct architecture matters more than early optimization.

---

# 9. Map Morpho leaves to neural structures

Instead of every terminal Morpho cell becoming a LUT/gate, experiment with terminal types such as:

```text
ExcitatoryNeuron
InhibitoryNeuron
InputNeuron
OutputNeuron
LIFNeuron
Synapse
NeuralPopulation
```

Later, Morpho cells may represent reusable neural motifs:

```text
ExcitatoryInhibitoryPair
RecurrentPool
Oscillator
WinnerTakeAll
FeedForwardLoop
DelayLoop
CoincidenceDetector
Integrator
```

Do not require all of these initially.

---

# 10. Developmental hierarchy

Explore whether recursion depth can carry semantic meaning.

For example:

```text
depth 0 → whole organism/network
depth 1 → neural region
depth 2 → population
depth 3 → microcircuit
depth 4 → neuron group / neuron
```

Avoid assuming every recursion level must literally correspond to biology.

Treat this as an experimental abstraction.

The important capability is hierarchical developmental encoding.

---

# 11. Initial developmental experiment

Begin with a very simple recursive neural grammar.

For example:

```text
NeuralRegion
    ↓
ExcitatoryPopulation
+
InhibitoryPopulation
```

And recursively:

```text
ExcitatoryPopulation
    ↓
smaller excitatory population
+
smaller excitatory population
+
inhibitory regulator
```

Eventually the recursion reaches terminal neurons.

The exact syntax does not initially need to be beautiful.

First prove that Morpho-style recursive rules can instantiate an operational recurrent SNN.

---

# 12. Structural plasticity

This is a major research objective.

Most neural learning changes:

```text
synaptic weight
```

We also want to explore learning that changes:

```text
network topology
```

Examples:

```text
create synapse
delete synapse

grow neuron
grow microcircuit

collapse region

duplicate pathway

add inhibitory pathway

regrow previously collapsed structure
```

This should occur on the slow developmental timescale.

---

# 13. Synaptic plasticity and structural plasticity should coexist

Do not replace ordinary weight learning.

Support a conceptual hierarchy:

```text
short-term evidence
      ↓
synaptic weight change
      ↓
persistent evidence
      ↓
structural change
```

For example:

```text
A→B repeatedly correlates
        ↓
STDP strengthens A→B
        ↓
connection stays strong for many epochs
        ↓
development system considers expanding pathway
```

Conversely:

```text
connection remains weak
+
rarely participates
        ↓
structural pruning candidate
```

---

# 14. Explore "weight becomes structure"

This is an important experimental idea.

Suppose:

```text
A ──0.97──► B
```

remains strongly useful across many developmental epochs.

Instead of allowing its weight to increase forever, Morpho may eventually transform that relationship into richer anatomy:

```text
        C
       / \
A ───►D───►B
       \ /
        E
```

Possible structural expansion could add:

```text
parallel pathways
inhibitory regulation
delay paths
recurrent loops
redundancy
specialized microcircuits
```

Treat this as an experiment rather than a required biological claim.

---

# 15. Activity statistics

Track slow neural statistics independently from instantaneous membrane state.

For each neuron or region consider:

```text
mean firing rate
recent spike count
activity EMA
input activity
output activity
correlation
synaptic utilization
age
energy
structural cost
```

Suggested conceptual structure:

```javascript
class ActivityStats {
    meanFiringRate;
    activityEMA;

    recentSpikes;

    inputCorrelation;
    outputInfluence;

    lastActiveTime;
}
```

---

# 16. Activity-dependent survival

Use neural activity to influence structural survival.

Conceptually:

```text
survival =
    useful neural activity
  + downstream influence
  + connectivity contribution
  + correlation
  + developmental bias
  - metabolic cost
```

Do not simply preserve the neurons with the highest firing rate.

Excessive firing may be pathological or redundant.

The metric should eventually consider whether activity is useful relative to the experiment.

Initially, however, a simpler activity-based score is acceptable.

---

# 17. Progressive pruning

Reuse the Living Graph concept from the music-plugin research.

The developmental graph should be capable of running indefinitely while remaining bounded.

Conceptually:

```text
growth
  ↓
activity
  ↓
aging
  ↓
survival evaluation
  ↓
pruning / collapse
  ↓
regrowth
  ↓
...
```

Maintain a configurable limit such as:

```javascript
maxLivingNeurons
maxSynapses
maxMorphoCells
```

The process may continue indefinitely, but resident memory must remain finite.

---

# 18. Prefer collapse over arbitrary destruction

Do not casually delete arbitrary internal neural structures if that invalidates the developmental hierarchy.

Experiment with reversible collapse.

For example:

```text
       region
      / | | \
     many neurons
```

may fold into:

```text
       proxy region
```

The proxy should retain enough developmental information for possible later expansion.

Potential retained metadata:

```text
developmental rule
external inputs
external outputs
previous population size
activity statistics
historical firing behaviour
random seed/state
structural role
```

---

# 19. Regrowth

Collapsed structures should eventually be capable of re-expanding.

Initially:

```text
collapse
→ later restore approximately the same structure
```

Later:

```text
collapse
→ later regrow a related but mutated structure
```

This creates:

```text
development
forgetting
redevelopment
```

rather than irreversible one-way growth.

---

# 20. Energy / metabolic pressure

Explore a simple energy model.

Each neural unit or developmental region may have:

```text
energy
maintenance cost
activity cost
```

For example:

```javascript
energy -= maintenanceCost;
energy -= spikeCost * spikeCount;
```

Useful activity may also replenish or justify resources.

The exact metaphor is experimental.

The purpose is to create competition for finite structural resources.

---

# 21. Global resource constraint

Optionally maintain a global resource budget.

Example:

```text
total neural resources = fixed
```

Then:

```text
more neurons
→ fewer resources per neuron
→ greater pruning pressure
```

This could naturally produce cycles:

```text
expansion
→ resource scarcity
→ pruning
→ recovery
→ expansion
```

Observe whether interesting dynamical regimes emerge without manually sequencing developmental phases.

---

# 22. Homeostatic morphogenesis

Explore whether growth itself can help regulate firing statistics.

For example:

```text
population firing too high
        ↓
increase probability of inhibitory growth
        ↓
reduce excitatory expansion
```

Or:

```text
population firing too low
        ↓
increase excitatory connectivity
        ↓
reduce pruning pressure
```

This is slow structural homeostasis rather than merely adjusting neuron gain.

Start with simple rules and make them inspectable.

---

# 23. Stochastic development

Morpho growth need not remain perfectly deterministic.

Allow developmental choices such as:

```text
60% → excitatory branch
25% → inhibitory branch
10% → recurrent connection
 5% → terminate
```

Probabilities may eventually depend upon:

```text
local firing rate
energy
age
development depth
region type
activity history
global resources
```

Use explicit seeds so experiments can be reproduced.

---

# 24. Developmental feedback loop

The central research loop should eventually become:

```text
Morpho developmental grammar
            ↓
     network topology
            ↓
      SNN dynamics
            ↓
     spike statistics
            ↓
 structural evaluation
            ↓
 growth / pruning decision
            ↓
     changed topology
            ↓
      changed dynamics
            ↓
            ...
```

Make this feedback visible in the UI.

---

# 25. Development epochs

Implement explicit developmental epochs.

Conceptually:

```javascript
while (running) {
    simulateNeuralDynamics(neuralStepsPerEpoch);

    activityTracker.update();

    const decisions =
        developmentController.evaluate(
            neuralGraph,
            activityTracker
        );

    livingMorphoGraph.apply(decisions);
}
```

Possible configuration:

```text
1 developmental epoch
=
1000 neural simulation steps
```

Exact values should be configurable.

---

# 26. Keep graph modification out of the spike loop

Do not mutate complex topology in the middle of processing one spike.

Prefer:

```text
simulate epoch
↓
collect statistics
↓
stop / reach safe boundary
↓
apply structural changes
↓
continue simulation
```

This simplifies correctness and future migration to higher-performance runtimes.

---

# 27. Visualization

The browser version is valuable precisely because we can visualize all of this.

Extend the existing graph viewer to show:

```text
neurons
synapses
excitatory/inhibitory distinction

spiking neurons

membrane activity if useful

recent firing-rate heat

growth frontier

newly born nodes

aging nodes

pruning candidates

collapsing regions

regrowing regions
```

Avoid visual overload.

Add toggles for separate overlays.

---

# 28. Timescale visualization

It should be possible to distinguish:

```text
SPIKE ACTIVITY
fast flashes

SYNAPTIC LEARNING
slow weight/edge changes

DEVELOPMENT
large structural birth/death events
```

This is important for understanding whether the system is behaving as intended.

---

# 29. Source-rule visualization

Where possible, preserve the existing Morpho concept of showing the source rule currently responsible for growth.

Eventually a development event should be traceable to:

```text
Morpho rule
source line
parent cell
development epoch
activity condition
random decision
```

This will be extremely useful for debugging emergent structures.

---

# 30. Instrumentation and inspectability

This is a research system.

Prioritize inspectability over clever abstraction.

Expose statistics such as:

```text
neuron count
synapse count

excitatory/inhibitory ratio

mean firing rate

spikes per second

active populations

growth events per epoch

pruning events per epoch

collapsed regions

resource usage

average synaptic weight

network activity distribution
```

Provide the ability to pause and inspect individual nodes.

---

# 31. Record developmental history

Maintain a bounded event history.

Example events:

```text
NeuronBorn
SynapseCreated
NeuronSpiked
WeightChanged
RegionExpanded
RegionCollapsed
NeuronPruned
SynapsePruned
RegionRegrown
DevelopmentRuleSelected
```

Do not retain every spike forever.

Use bounded buffers or aggregated histories.

---

# 32. Experiment reproducibility

Every experiment should have an explicit seed.

Record:

```text
Morpho source
initial parameters
random seed
neuron parameters
development parameters
simulation timestep
epoch size
```

Same configuration + same seed should reproduce the same experiment where practical.

---

# 33. Initial experiment sequence

Do not jump immediately to a self-evolving neural organism.

Implement progressively.

## Experiment 1 — Static SNN on existing graph

Build a small recurrent neural graph manually.

Verify:

```text
LIF dynamics
synaptic weights
delays
recurrent spikes
visual spike animation
```

No development.

Success:

> The browser can simulate and visualize a small recurrent SNN correctly.

---

## Experiment 2 — Morpho generates SNN once

Create a simple developmental grammar.

Morpho expands recursively into:

```text
neurons
synapses
populations
```

After growth ends, run the SNN.

No pruning.

Success:

> The same compact developmental rule can generate a substantially larger operational SNN.

---

## Experiment 3 — Iterative development

Alternate:

```text
SNN simulation
→ activity measurement
→ another Morpho growth step
→ SNN simulation
```

Success:

> Neural activity continues correctly while the network develops across epochs.

---

## Experiment 4 — Activity-dependent pruning

Add basic activity traces.

Prune low-activity safe structures.

Success:

> A bounded network can grow and prune over many epochs without invalid graph state.

---

## Experiment 5 — Activity-dependent growth

Allow activity to affect which regions expand.

For example:

```text
low firing
→ excitatory growth bias

high firing
→ inhibitory growth bias
```

Observe whether basic homeostatic behaviour emerges.

---

## Experiment 6 — Weight-to-structure

Track persistent strong connections.

Allow a persistent relationship to trigger structural expansion.

Success:

> Long-lived synaptic evidence can produce a topological change.

---

## Experiment 7 — Reversible collapse/regrowth

Collapse old low-value regions into proxies.

Later allow them to expand again.

Success:

> Network development can forget structural detail without completely destroying developmental potential.

---

## Experiment 8 — Bounded Living SNN

Combine:

```text
continuous SNN activity
growth
pruning
collapse
regrowth
finite resource budget
```

Run for a large number of epochs.

Success:

> The system remains computationally bounded while its structure continues changing indefinitely.

---

# 34. Do not optimize prematurely

Early priorities:

1. correct spike dynamics
2. inspectable state
3. correct structural changes
4. reproducible experiments
5. stable visualization
6. bounded memory
7. interesting emergent behaviour
8. performance

Only optimize hot paths once profiling demonstrates a need.

The browser is explicitly being retained for rapid development.

---

# 35. Relationship to the C++/JUCE plugin

This project should NOT block the JUCE plugin.

Treat them as parallel tracks.

```text
TRACK A

Morpho → native C++ → JUCE
              ↓
generative MIDI instrument


TRACK B

Morpho web environment
        ↓
developmental SNN research
        ↓
structural plasticity experiments
        ↓
validated concepts
```

Periodically identify successful concepts from Track B that could benefit Track A.

Possible transferable concepts include:

```text
activity-dependent pruning
energy/resource systems
structural survival
stochastic growth
collapse/regrowth
probability models
developmental event streams
```

Do not port experimental SNN machinery into the plugin merely because it exists.

Only migrate mechanisms that demonstrate clear value.

---

# 36. Potential later musical bridge

Music is not the immediate requirement of this research track, but preserve the ability to sonify neural events later.

Possible mappings:

```text
spike → note / trigger

population burst → chord/accent

spatial spike propagation → melodic movement

synchronization → harmonic event

new neuron → birth sound

new neural region → new voice/timbre

pruning → musical thinning

regrowth → motif recurrence/variation
```

An eventual hybrid may allow:

```text
SNN activity
      ↓
MIDI / audio
```

But do not make musical output a dependency for early SNN experiments.

---

# 37. Potential evolutionary layer

Do not implement this initially, but preserve the conceptual distinction between:

## Evolution

Changes the developmental program.

```text
Morpho source / growth grammar
```

## Development

Changes network topology during an individual's lifetime.

```text
neurons
synapses
regions
```

## Learning

Changes neural state/weights during operation.

```text
synaptic strength
activity dynamics
```

Eventually:

```text
Morpho genotype
     ↓
development
     ↓
SNN phenotype
     ↓
learning / lifetime activity
     ↓
evaluation
     ↓
mutate developmental rules
     ↓
next generation
```

This may become a separate experiment later.

---

# 38. Code organization

Prefer new modules with clear responsibilities.

Possible web-side structure:

```text
js/neural/
    neuron.js
    synapse.js
    neural_graph.js

    spike_engine.js
    spike_queue.js

    activity_tracker.js
    plasticity.js

    development_controller.js
    structural_plasticity.js

    living_neural_graph.js

    neural_renderer.js
    neural_experiments.js
```

Names may differ if the repository architecture suggests better integration.

Keep neural simulation independent enough to unit test without rendering.

---

# 39. Tests

Add automated tests for at least:

```text
single LIF neuron threshold/reset

refractory period

excitatory synapse

inhibitory synapse

synaptic delay

recurrent connection

deterministic simulation with seed

Morpho-generated neuron count

Morpho-generated connectivity

activity statistics

development epoch boundary

safe structural addition

safe structural deletion

protected active structure is not deleted

maximum neuron budget respected

maximum synapse budget respected

collapse preserves external connectivity

regrowth restores valid connectivity

same seed reproduces development

long-running simulation remains bounded
```

---

# 40. Experimental logging

For each experiment, provide a compact result report.

Record:

```text
experiment name

hypothesis

initial topology

Morpho rule

seed

simulation parameters

development parameters

number of epochs

final neuron count

final synapse count

growth count

prune count

mean firing rate

observed behaviour

problems

next experiment
```

We want to learn from failed experiments rather than only produce demos.

---

# 41. First coding task

Begin by inspecting the current Morpho web architecture and identify the cleanest integration points for an independent SNN runtime.

Then implement **Experiment 1 only**:

> A small recurrent Leaky Integrate-and-Fire network running inside the existing browser environment, with excitatory/inhibitory weighted synapses, configurable transmission delays, deterministic simulation, and simple spike visualization.

Requirements:

- do not alter Morpho compilation semantics yet
- do not implement recursive neural development yet
- do not implement pruning yet
- do not implement STDP yet
- do not implement evolutionary algorithms yet
- keep neural simulation independent of rendering
- add automated tests
- expose a small deterministic demo network
- document how the neural runtime could later receive topology from Morpho

Once this works, proceed to Experiment 2:

> Let one simple Morpho developmental rule generate the neural topology, then run the same SNN engine on the resulting network.

Only after those two foundations are stable should iterative development, structural plasticity, pruning and continuous Living SNN behaviour be attempted.

---

# 42. Research objective

The eventual question this project should help answer is:

> Can a compact recursive developmental grammar generate a spiking neural structure whose ongoing neural activity influences which parts of that structure grow, survive, collapse and regrow?

And beyond that:

> Can recursively generated structure plus iterative lifetime development create useful or musically interesting dynamics that would be difficult to obtain from either a fixed SNN or a conventional generative graph alone?

The desired long-term feedback loop is:

```text
developmental grammar
        ↓
neural structure
        ↓
spike dynamics
        ↓
activity / learning
        ↓
structural survival
        ↓
new neural structure
        ↓
new dynamics
        ↓
...
```

This research track should remain exploratory, visual, measurable and reproducible.

Do not rush to move it into C++.

Use the web implementation as the laboratory.