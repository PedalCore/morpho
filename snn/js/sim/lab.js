// Lab: assembles genotype → phenotype → dynamics → feedback into one
// deterministic, headless experiment object. The browser UI and the node
// tests both drive experiments through this module, so what you hear is
// exactly what the tests measure.

import { mulberry32, makeStreams } from '../core/rng.js';
import { resetNeuronIds } from '../neural/neuron.js';
import { resetSynapseIds } from '../neural/synapse.js';
import { NeuralGraph } from '../neural/graph.js';
import { SpikeEngine } from '../neural/engine.js';
import { ActivityTracker } from '../neural/activity.js';
import { growNetwork, sproutRegion, DEFAULT_GRAMMAR } from '../morpho/grammar.js';
import { DevelopmentController } from '../morpho/development.js';
import { WalkerSystem } from '../music/walker.js';
import { STDP } from '../neural/plasticity.js';

export const DEFAULT_SIM = {
  epochSteps: 2000, // 2 s of simulated time per development epoch
  pulsePeriodMs: 340, // rhythmic input drive — the "tempo" of the organism
  pulseFireProb: 0.55, // each input neuron fires with this prob per pulse
  backgroundHz: 1.0, // Poisson background per input neuron
  developmentEnabled: true,
  modProb: 0.06, // chance a modulator spike actually changes key — biased low
  modCooldownMs: 12000, // minimum time between key changes
  drivePattern: 'steady', // rhythm of the input drive (see DRIVE_PATTERNS)
  stdpEnabled: false, // spike-timing-dependent plasticity on excitatory synapses
};

// Input-drive rhythms on an 8-slot grid; one slot = half the pulse period,
// so a full cycle spans four pulses. Deterministic, part of the sim.
export const DRIVE_PATTERNS = {
  steady: [1, 0, 1, 0, 1, 0, 1, 0],
  euclidean: [1, 0, 1, 1, 0, 1, 1, 0],
  bursts: [1, 1, 1, 0, 0, 0, 0, 0],
  sparse: [1, 0, 0, 0, 0, 0, 1, 0],
};

// circle of fifths, by position; offset semitones = (7 × position) mod 12
export const KEY_NAMES = ['C', 'G', 'D', 'A', 'E', 'B', 'F♯', 'D♭', 'A♭', 'E♭', 'B♭', 'F'];

export class Lab {
  constructor({ seed = 42, sim = {}, grammar = {}, dev = {}, walk = {} } = {}) {
    this.seed = seed;
    this.simParams = { ...DEFAULT_SIM, ...sim };
    this.grammarParams = { ...DEFAULT_GRAMMAR, ...grammar };

    resetNeuronIds();
    resetSynapseIds();
    this.streams = makeStreams(seed);
    this.streams.walk = mulberry32(seed ^ 0x27d4eb2f);
    this.graph = new NeuralGraph();
    growNetwork(this.graph, this.grammarParams, this.streams.build);

    this.engine = new SpikeEngine(this.graph, { maxDelaySteps: 64 });
    this.activity = new ActivityTracker();
    this.dev = new DevelopmentController(dev);
    this.walkers = new WalkerSystem(this.graph, this.streams.walk, walk);
    this.stdp = new STDP();

    this.epoch = 0;
    this.inputIds = [...this.graph.neurons.values()]
      .filter((n) => n.role === 'input')
      .map((n) => n.id);
    this.pulseCount = 0;
    this.lastEpochChanges = { born: [], pruned: [], subdivided: [] };
    this.onEpoch = null; // (lab) => void

    // harmonic state: position on the circle of fifths, moved by rare
    // modulator nodes (deterministic — part of the sim, not the audio layer)
    this.key = { fifths: 0, offset: 0 };
    this.keyChangesTotal = 0;
    this.lastModStep = -Infinity;
    this.onKeyChange = null; // ({neuronId, name, rule, fifths, offset}) => void

    // externally injected input spikes (e.g. MIDI → spike encoding), with
    // optional per-spike delay so bursts can be scheduled
    this.pendingInputFires = [];

    // optional attention component (attention.html experiment)
    this.attention = null;
  }

  attachAttention(attention) {
    this.attention = attention;
    this.engine.modulation = (n) => attention.gainOf(n);
  }

  fireInput(neuronId, delaySteps = 0) {
    this.pendingInputFires.push({ id: neuronId, at: this.engine.stepCount + delaySteps });
  }

  // One 1 ms step: external drive, walkers, then neural dynamics.
  step() {
    const { pulsePeriodMs, pulseFireProb, backgroundHz, drivePattern } = this.simParams;
    const t = this.engine.stepCount;

    if (this.pendingInputFires.length) {
      const rest = [];
      for (const f of this.pendingInputFires) {
        if (f.at <= t) this.engine.forceFire(f.id);
        else rest.push(f);
      }
      this.pendingInputFires = rest;
    }

    const sub = Math.max(1, Math.round(pulsePeriodMs / 2));
    if (t % sub === 0) {
      const pattern = DRIVE_PATTERNS[drivePattern] ?? DRIVE_PATTERNS.steady;
      if (pattern[Math.floor(t / sub) % pattern.length]) {
        this.pulseCount++;
        for (const id of this.inputIds) {
          if (this.streams.sim() < pulseFireProb) this.engine.forceFire(id);
        }
      }
    }
    const pBg = backgroundHz / 1000;
    for (const id of this.inputIds) {
      if (this.streams.sim() < pBg) this.engine.forceFire(id);
    }

    if (this.attention && t % this.attention.periodMs === 0) this.attention.update();

    this.walkers.tick(t, pulsePeriodMs);

    const spikes = this.engine.step();
    if (spikes.length && this.simParams.stdpEnabled) {
      const now = this.engine.stepCount;
      for (const id of spikes) {
        const n = this.graph.neurons.get(id);
        if (n) this.stdp.onSpike(this.graph, n, now);
      }
    }
    if (spikes.length) this.maybeModulate(spikes);

    if (this.engine.stepCount % this.simParams.epochSteps === 0) {
      this.runEpochBoundary();
    }
    return spikes;
  }

  // Rare modulator nodes nudge the key around the circle of fifths. Two
  // rules: move to the adjacent key (up a fifth / down a fourth), or skip
  // over a position (two steps) — either direction. Heavily biased low:
  // few modulator nodes exist, each spike rarely triggers, and a cooldown
  // prevents thrashing.
  maybeModulate(spikes) {
    const t = this.engine.stepCount;
    if (t - this.lastModStep < this.simParams.modCooldownMs) return;
    for (const id of spikes) {
      const n = this.graph.neurons.get(id);
      if (!n || !n.isModulator) continue;
      if (this.streams.sim() >= this.simParams.modProb) continue;
      const r = this.streams.sim();
      let steps;
      let rule;
      if (r < 0.7) {
        steps = this.streams.sim() < 0.5 ? 1 : -1; // adjacent on the circle
        rule = steps > 0 ? 'up a fifth' : 'down a fourth';
      } else {
        steps = this.streams.sim() < 0.5 ? 2 : -2; // skip over
        rule = steps > 0 ? 'skip up' : 'skip down';
      }
      this.key.fifths = (((this.key.fifths + steps) % 12) + 12) % 12;
      this.key.offset = (7 * this.key.fifths) % 12;
      this.keyChangesTotal++;
      this.lastModStep = t;
      const name = KEY_NAMES[this.key.fifths];
      this.dev.log({ epoch: this.epoch, type: 'KeyChanged', id, rule, key: name });
      if (this.onKeyChange) {
        this.onKeyChange({ neuronId: id, name, rule, ...this.key });
      }
      break; // at most one modulation per step
    }
  }

  runEpochBoundary() {
    this.epoch++;
    this.activity.update(this.graph, this.simParams.epochSteps);
    if (this.simParams.developmentEnabled) {
      const decisions = this.dev.evaluate(
        this.graph,
        this.activity,
        this.epoch,
        this.streams.dev,
        this.walkers.occupiedIds()
      );
      this.lastEpochChanges = this.dev.apply(
        this.graph,
        decisions,
        this.epoch,
        this.streams.dev,
        this.grammarParams
      );
    } else {
      this.lastEpochChanges = { born: [], pruned: [], subdivided: [] };
    }
    if (this.onEpoch) this.onEpoch(this);
  }

  // Manual recursive fan-out ("branch" button): sprout fresh sibling
  // populations off up to `count` leaf regions. Uses the dev RNG stream, so
  // pressing it changes the developmental future (it is a real intervention).
  branchOut(count = 2) {
    const p = this.dev.params;
    const changes = { born: [], moved: [], sprouted: [] };
    const eligible = this.graph
      .leafRegions()
      .filter((r) => r.depth < p.maxDepth)
      .sort((a, b) => (a.path < b.path ? -1 : 1));
    for (let i = 0; i < count && eligible.length; i++) {
      if (this.graph.neurons.size >= p.maxNeurons - 8) break;
      if (this.graph.synapses.size >= p.maxSynapses - 50) break;
      const idx = Math.floor(this.streams.dev() * eligible.length);
      const region = eligible.splice(idx, 1)[0];
      const res = sproutRegion(this.graph, region, this.grammarParams, this.streams.dev, this.epoch);
      changes.born.push(...res.bornIds);
      changes.moved.push(...res.movedIds);
      changes.sprouted.push({ parent: region.path, children: res.children.map((c) => c.path) });
      this.dev.grownTotal += res.bornIds.length;
      this.dev.subdividedTotal++;
      this.dev.log({
        epoch: this.epoch,
        type: 'RegionSprouted',
        region: region.path,
        children: res.children.map((c) => c.path),
        newNeurons: res.bornIds.length,
      });
    }
    return changes;
  }

  runSteps(n) {
    for (let i = 0; i < n; i++) this.step();
  }

  runEpochs(n) {
    this.runSteps(n * this.simParams.epochSteps);
  }

  report() {
    const c = this.graph.counts();
    const octaves = new Set();
    for (const n of this.graph.neurons.values()) {
      if (n.role !== 'input') octaves.add(n.octave);
    }
    return {
      seed: this.seed,
      epoch: this.epoch,
      ...c,
      leafRegions: this.graph.leafRegions().length,
      octaveSpread: octaves.size,
      meanRateHz: +this.activity.networkRateHz.toFixed(3),
      grownTotal: this.dev.grownTotal,
      prunedTotal: this.dev.prunedTotal,
      subdividedTotal: this.dev.subdividedTotal,
      key: KEY_NAMES[this.key.fifths],
      keyChanges: this.keyChangesTotal,
    };
  }
}
