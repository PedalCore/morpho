// Sensory layer for the duet experiment: human notes become spikes.
//
// One input neuron per scale degree, wired tonotopically — each projects to
// excitatory neurons whose birth-fixed pitch sounds that same degree class.
// Playing a phrase therefore excites exactly the anatomy that would echo it;
// STDP strengthens the pathways you actually play, and (with no metronome
// drive) homeostatic development grows around what it hears. The human is
// the environment the organism develops in.

import { SCALES } from '../ui/audio.js';

export const NOTE_NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];

export function wireSensoryInputs(graph, scaleName, rng, { fanout = 10, weight = [0.65, 1.0] } = {}) {
  const scale = SCALES[scaleName];
  graph.addRegion('IN', 0, 'input', null);
  const region = graph.regions.get('IN');
  const inputs = [];
  for (let d = 0; d < scale.length; d++) {
    const n = graph.addNeuron({ role: 'input', region: 'IN' });
    n.degreeClass = d;
    region.members.add(n.id);
    const targets = [...graph.neurons.values()].filter(
      (t) => t.role === 'excitatory' && t.structDegree % scale.length === d
    );
    const wired = new Set();
    for (let k = 0; k < Math.min(fanout, targets.length); k++) {
      const t = targets[Math.floor(rng() * targets.length)];
      if (wired.has(t.id)) continue;
      wired.add(t.id);
      graph.addSynapse({
        source: n.id,
        target: t.id,
        weight: weight[0] + rng() * (weight[1] - weight[0]),
        delaySteps: 1 + Math.floor(rng() * 3),
      });
    }
    inputs.push(n);
  }
  return inputs;
}

// MIDI note → snapped scale degree → burst of input spikes (velocity sets
// burst length: louder playing drives the network harder).
export class SpikeEncoder {
  constructor(lab, inputs, scaleName) {
    this.lab = lab;
    this.inputs = inputs;
    this.scale = SCALES[scaleName];
  }

  noteOn(midiNote, velocity = 0.8) {
    const rel = ((((midiNote % 12) - this.lab.key.offset) % 12) + 12) % 12;
    let best = 0;
    let bestDist = 99;
    this.scale.forEach((s, i) => {
      const d = Math.min((rel - s + 12) % 12, (s - rel + 12) % 12);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    const input = this.inputs[best];
    if (!input) return null;
    const burst = velocity > 0.7 ? 3 : velocity > 0.35 ? 2 : 1;
    for (let b = 0; b < burst; b++) this.lab.fireInput(input.id, b * 12);
    this.lab.attention?.noteHeard(best);
    return { degree: best, inputId: input.id };
  }
}
