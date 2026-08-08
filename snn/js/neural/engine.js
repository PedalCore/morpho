// Fixed-timestep spiking engine (dt = 1 ms). Delayed synaptic transmission is
// handled with a ring buffer of pending deliveries indexed by arrival step —
// event-driven in spirit, simple and deterministic in practice.
//
// The engine never mutates topology. Development applies structural changes
// between epochs; pending deliveries to pruned neurons are dropped on arrival.

export class SpikeEngine {
  constructor(graph, { maxDelaySteps = 64 } = {}) {
    this.graph = graph;
    this.stepCount = 0;
    this.ringSize = maxDelaySteps + 1;
    this.ring = Array.from({ length: this.ringSize }, () => []);
    this.onSpike = null; // (neuron, step) => void — audio/renderer hook
    this.lastStepSpikes = [];
  }

  scheduleDelivery(targetId, weight, delaySteps) {
    const d = Math.min(Math.max(1, delaySteps), this.ringSize - 1);
    const slot = (this.stepCount + d) % this.ringSize;
    this.ring[slot].push(targetId, weight);
  }

  // Externally drive an input neuron (rhythmic pulses, Poisson noise, tests).
  forceFire(neuronId) {
    const n = this.graph.neurons.get(neuronId);
    if (!n) return;
    this._fire(n);
  }

  _fire(n) {
    n.membrane = n.resetPotential;
    n.refractoryUntil = this.stepCount + n.refractorySteps;
    n.spikeCount++;
    n.lastSpikeStep = this.stepCount;
    this.lastStepSpikes.push(n.id);
    for (const s of this.graph.outgoing.get(n.id)) {
      if (s.enabled) this.scheduleDelivery(s.target, s.weight, s.delaySteps);
    }
    if (this.onSpike) this.onSpike(n, this.stepCount);
  }

  step() {
    // note: lastStepSpikes may already hold spikes force-fired since the
    // previous step (input drive, MIDI encoding) — they belong to this step
    const { neurons } = this.graph;

    // 1. deliver spikes arriving now
    const slot = this.stepCount % this.ringSize;
    const arrivals = this.ring[slot];
    for (let i = 0; i < arrivals.length; i += 2) {
      const n = neurons.get(arrivals[i]);
      if (n && n.role !== 'input' && this.stepCount >= n.refractoryUntil) {
        n.membrane += arrivals[i + 1];
      }
    }
    arrivals.length = 0;

    // 2. leak + threshold
    for (const n of neurons.values()) {
      if (n.role === 'input') continue;
      n.membrane *= n.decay;
      if (n.membrane < -2) n.membrane = -2; // clamp runaway inhibition
      if (this.stepCount >= n.refractoryUntil && n.membrane >= n.threshold) {
        this._fire(n);
      }
    }

    this.stepCount++;
    const spikes = this.lastStepSpikes;
    this.lastStepSpikes = [];
    return spikes;
  }
}
