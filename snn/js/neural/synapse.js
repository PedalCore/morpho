// Explicit synapse: signed weight (excitatory > 0, inhibitory < 0) and an
// integer transmission delay in simulation steps. Recurrence is legal —
// nothing here assumes a DAG.

// ids are assigned by NeuralGraph (per-graph counters)

export class Synapse {
  constructor({ source, target, weight, delaySteps = 1, enabled = true }) {
    this.id = 0; // assigned by NeuralGraph.addSynapse
    this.source = source; // neuron id
    this.target = target; // neuron id
    this.weight = weight;
    this.delaySteps = Math.max(1, delaySteps | 0);
    this.enabled = enabled;
  }
}
