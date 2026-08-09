// NeuralGraph: topology container (neurons, synapses, regions). The spiking
// engine reads it; the development layer mutates it — only at epoch
// boundaries, never mid-step.

import { Neuron } from './neuron.js';
import { Synapse } from './synapse.js';

export class NeuralGraph {
  constructor() {
    this.nextNeuronId = 1;
    this.nextSynapseId = 1;
    this.neurons = new Map(); // id -> Neuron
    this.synapses = new Map(); // id -> Synapse
    this.outgoing = new Map(); // neuronId -> Synapse[]
    this.incoming = new Map(); // neuronId -> Synapse[]
    // regions: developmental hierarchy. path like "R.0.1"; leaf regions hold neurons.
    this.regions = new Map(); // path -> { path, depth, kind, members:Set<neuronId>, parent }
  }

  // a0/a1: angular sector of the developmental hierarchy, subdivided as
  // regions divide — gives every region a stable structural position that
  // layout and pitch mapping both read.
  addRegion(path, depth, kind, parent = null, a0 = 0, a1 = Math.PI * 2) {
    if (!this.regions.has(path)) {
      const degreeOffset = Math.round((((a0 + a1) / 2) / (Math.PI * 2)) * 7) % 7;
      this.regions.set(path, {
        path,
        depth,
        kind,
        members: new Set(),
        parent,
        a0,
        a1,
        degreeOffset,
        bornCount: 0,
      });
    }
    return this.regions.get(path);
  }

  addNeuron(opts) {
    const n = new Neuron(opts);
    n.id = this.nextNeuronId++;
    this.neurons.set(n.id, n);
    this.outgoing.set(n.id, []);
    this.incoming.set(n.id, []);
    const region = this.regions.get(n.region);
    if (region) region.members.add(n.id);
    return n;
  }

  addSynapse(opts) {
    if (!this.neurons.has(opts.source) || !this.neurons.has(opts.target)) {
      throw new Error(`synapse endpoints must exist: ${opts.source}->${opts.target}`);
    }
    const s = new Synapse(opts);
    s.id = this.nextSynapseId++;
    this.synapses.set(s.id, s);
    this.outgoing.get(s.source).push(s);
    this.incoming.get(s.target).push(s);
    return s;
  }

  removeSynapse(id) {
    const s = this.synapses.get(id);
    if (!s) return;
    this.synapses.delete(id);
    const out = this.outgoing.get(s.source);
    if (out) out.splice(out.indexOf(s), 1);
    const inc = this.incoming.get(s.target);
    if (inc) inc.splice(inc.indexOf(s), 1);
  }

  removeNeuron(id) {
    const n = this.neurons.get(id);
    if (!n) return;
    for (const s of [...this.outgoing.get(id)]) this.removeSynapse(s.id);
    for (const s of [...this.incoming.get(id)]) this.removeSynapse(s.id);
    this.outgoing.delete(id);
    this.incoming.delete(id);
    const region = this.regions.get(n.region);
    if (region) region.members.delete(id);
    this.neurons.delete(id);
  }

  leafRegions() {
    return [...this.regions.values()].filter((r) => r.kind === 'leaf');
  }

  counts() {
    let excitatory = 0;
    let inhibitory = 0;
    let inputs = 0;
    let outputs = 0;
    for (const n of this.neurons.values()) {
      if (n.role === 'input') inputs++;
      else if (n.role === 'inhibitory') inhibitory++;
      else excitatory++;
      if (n.isOutput) outputs++;
    }
    return {
      neurons: this.neurons.size,
      synapses: this.synapses.size,
      excitatory,
      inhibitory,
      inputs,
      outputs,
    };
  }
}
