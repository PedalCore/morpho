// Leaky Integrate-and-Fire neuron. Pure simulation state — no visualization
// or audio state lives here (renderer/audio keep their own maps keyed by id).

// ids are assigned by NeuralGraph (per-graph counters — organisms may coexist)

export const ROLES = {
  INPUT: 'input', // externally driven, never integrates
  EXCITATORY: 'excitatory',
  INHIBITORY: 'inhibitory',
};

export class Neuron {
  constructor({
    role = ROLES.EXCITATORY,
    region = 'R',
    threshold = 1.0,
    resetPotential = 0.0,
    tauMs = 20, // membrane leak time constant
    refractoryMs = 5,
    dtMs = 1,
    isOutput = false,
    isModulator = false, // rare node: firing may nudge the key around the circle of fifths
    // Structural pitch, fixed at birth (plugin brief: depth → register,
    // structural position → scale degree). Never derived from moving layout.
    octave = 1,
    structDegree = 0,
    bornEpoch = 0,
  } = {}) {
    this.id = 0; // assigned by NeuralGraph.addNeuron
    this.role = role;
    this.region = region;
    this.isOutput = isOutput;
    this.isModulator = isModulator;
    this.octave = octave;
    this.structDegree = structDegree;
    this.bornEpoch = bornEpoch;

    this.membrane = 0;
    this.threshold = threshold;
    this.resetPotential = resetPotential;
    this.decay = Math.exp(-dtMs / tauMs);
    this.refractorySteps = Math.round(refractoryMs / dtMs);
    this.refractoryUntil = -1;

    // per-epoch spike counter + slow activity trace (updated by ActivityTracker)
    this.spikeCount = 0;
    this.activityEMA = 0;
    this.lastSpikeStep = -1;

    // survival ecology (plugin brief: energy restored by activity and by
    // walker visits — musical use keeps structure alive)
    this.energy = 1.0;
    this.walkerVisits = 0;
  }
}
