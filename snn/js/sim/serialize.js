// Organism persistence: full deterministic snapshot of a Lab. A restored
// organism continues *spike-for-spike identically* — every neuron field,
// synapse, region, in-flight delivery, RNG stream state, plasticity trace
// and walker position is captured. That property is what makes lifetime
// studies (raise a brain over days, hand it to someone else) meaningful.

import { Lab } from './lab.js';

export const FORMAT = 'morpho-snn-organism';
export const VERSION = 1;

export function serializeLab(lab) {
  const neurons = [...lab.graph.neurons.values()].map((n) => ({ ...n }));
  const synapses = [...lab.graph.synapses.values()].map((s) => ({ ...s }));
  const regions = [...lab.graph.regions.values()].map((r) => ({
    ...r,
    members: [...r.members],
  }));
  return {
    format: FORMAT,
    version: VERSION,
    savedAt: undefined, // stamped by the UI, not the sim
    seed: lab.seed,
    simParams: { ...lab.simParams },
    grammarParams: { ...lab.grammarParams },
    devParams: { ...lab.dev.params },
    walkParams: { ...lab.walkers.params },
    stdpParams: { ...lab.stdp.p },
    epoch: lab.epoch,
    pulseCount: lab.pulseCount,
    key: { ...lab.key },
    keyChangesTotal: lab.keyChangesTotal,
    lastModStep: Number.isFinite(lab.lastModStep) ? lab.lastModStep : null,
    inputIds: [...lab.inputIds],
    devTotals: {
      grown: lab.dev.grownTotal,
      pruned: lab.dev.prunedTotal,
      subdivided: lab.dev.subdividedTotal,
    },
    devEvents: lab.dev.events.slice(),
    activity: {
      epoch: lab.activity.epoch,
      networkRateHz: lab.activity.networkRateHz,
      rateHistory: lab.activity.rateHistory.slice(),
    },
    rng: {
      build: lab.streams.build.getState(),
      sim: lab.streams.sim.getState(),
      dev: lab.streams.dev.getState(),
      walk: lab.streams.walk.getState(),
    },
    engine: {
      stepCount: lab.engine.stepCount,
      ringSize: lab.engine.ringSize,
      ring: lab.engine.ring.map((slot) => slot.slice()),
    },
    walkers: lab.walkers.walkers.map((w) => ({ ...w, history: w.history.slice() })),
    eligibility: [...lab.stdp.eligibility.entries()].map(([id, rec]) => [id, { ...rec }]),
    pendingInputFires: lab.pendingInputFires.map((f) => ({ ...f })),
    attention: lab.attention
      ? {
          params: { ...lab.attention.params },
          inputHist: lab.attention.inputHist.slice(),
          hists: lab.attention.hists.map((h) => h.slice()),
        }
      : null,
    // live counters — they also count pruned ids, so max(living)+1 is wrong
    nextIds: { neuron: lab.graph.nextNeuronId, synapse: lab.graph.nextSynapseId },
    neurons,
    synapses,
    regions,
    // adjacency array ORDER is behavior (walkers index into it; it encodes
    // the graph's mutation history) — preserve it exactly
    outgoingOrder: [...lab.graph.outgoing.entries()].map(([id, list]) => [
      id,
      list.map((s) => s.id),
    ]),
    incomingOrder: [...lab.graph.incoming.entries()].map(([id, list]) => [
      id,
      list.map((s) => s.id),
    ]),
  };
}

export function deserializeLab(data, { AttentionClass = null } = {}) {
  if (data.format !== FORMAT) throw new Error('not a morpho-snn organism file');
  const lab = new Lab({
    seed: data.seed,
    sim: data.simParams,
    grammar: data.grammarParams,
    dev: data.devParams,
    walk: data.walkParams,
  });

  // rebuild the graph exactly (discard the freshly grown one). Neuron and
  // Synapse are pure data bags, so plain-object restoration is faithful.
  lab.graph.neurons.clear();
  lab.graph.synapses.clear();
  lab.graph.outgoing.clear();
  lab.graph.incoming.clear();
  lab.graph.regions.clear();
  for (const r of data.regions) {
    lab.graph.regions.set(r.path, { ...r, members: new Set(r.members) });
  }
  for (const nd of data.neurons) {
    lab.graph.neurons.set(nd.id, { ...nd });
  }
  for (const sd of data.synapses) {
    lab.graph.synapses.set(sd.id, { ...sd });
  }
  // rebuild adjacency in the exact serialized order
  for (const [id, synIds] of data.outgoingOrder) {
    lab.graph.outgoing.set(
      id,
      synIds.map((sid) => lab.graph.synapses.get(sid)).filter(Boolean)
    );
  }
  for (const [id, synIds] of data.incomingOrder) {
    lab.graph.incoming.set(
      id,
      synIds.map((sid) => lab.graph.synapses.get(sid)).filter(Boolean)
    );
  }

  // sim state
  lab.epoch = data.epoch;
  lab.pulseCount = data.pulseCount;
  lab.key = { ...data.key };
  lab.keyChangesTotal = data.keyChangesTotal;
  lab.lastModStep = data.lastModStep ?? -Infinity;
  lab.inputIds = [...data.inputIds];
  lab.dev.grownTotal = data.devTotals.grown;
  lab.dev.prunedTotal = data.devTotals.pruned;
  lab.dev.subdividedTotal = data.devTotals.subdivided;
  lab.dev.events = data.devEvents.slice();
  lab.activity.epoch = data.activity.epoch;
  lab.activity.networkRateHz = data.activity.networkRateHz;
  lab.activity.rateHistory = data.activity.rateHistory.slice();
  lab.streams.build.setState(data.rng.build);
  lab.streams.sim.setState(data.rng.sim);
  lab.streams.dev.setState(data.rng.dev);
  lab.streams.walk.setState(data.rng.walk);
  lab.engine.stepCount = data.engine.stepCount;
  lab.engine.ringSize = data.engine.ringSize;
  lab.engine.ring = data.engine.ring.map((slot) => slot.slice());
  lab.walkers.walkers = data.walkers.map((w) => ({ ...w, history: w.history.slice() }));
  lab.walkers.params.count = lab.walkers.walkers.length;
  lab.stdp.p = { ...data.stdpParams };
  lab.stdp.eligibility = new Map(data.eligibility.map(([id, rec]) => [id, { ...rec }]));
  lab.pendingInputFires = data.pendingInputFires.map((f) => ({ ...f }));

  if (data.attention && AttentionClass) {
    const scaleLen = data.attention.inputHist.length;
    const attn = new AttentionClass(lab.graph, scaleLen, data.attention.params);
    attn.inputHist = data.attention.inputHist.slice();
    if (data.attention.hists) attn.hists = data.attention.hists.map((h) => h.slice());
    lab.attachAttention(attn);
  }

  // id counters continue past the restored population
  lab.graph.nextNeuronId = data.nextIds.neuron;
  lab.graph.nextSynapseId = data.nextIds.synapse;
  return lab;
}
