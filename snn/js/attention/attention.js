// Regional attention — the MA-SNN idea (arXiv:2209.13929) made gradient-free
// and developmental.
//
// MA-SNN learns attention weights over time/channel/space and uses them to
// modulate membrane potentials, getting sparser spiking AND better task
// performance. Here the "channels" are Morpho's leaf regions, and the
// attention statistic is computed, not learned: each region's gain reflects
// how well its pitch material matches what the human has recently played
// (recency-weighted degree histogram — the temporal-attention dimension).
// Gains multiply synaptic delivery via engine.modulation, so attended
// anatomy leans in and unrelated anatomy quiets down.
//
// The developmental twist none of the attention papers have: attention wins
// trickle survival energy into the winning regions, so sustained attention
// literally shapes what grows — attention as morphogen.

import { cosine } from '../duet/dialogue.js';

export const DEFAULT_ATTN = {
  strength: 0.6, // 0 = off (all gains 1) … 1 = full effect
  // 'suppress' won the ablation decisively (rel 0.61 vs 0.54, 85% fewer
  // spikes): damping non-matching regions without boosting anything keeps
  // the recurrent net stable. 'balanced' (gains 1±s/2) kept for comparison.
  bias: 'suppress',
  periodMs: 250, // recompute cadence (medium timescale: faster than dev, slower than spikes)
  inputDecay: 0.82, // per-update decay of the heard-note histogram (recency weighting)
  energyTrickle: 0.02, // per-update survival energy for strongly attended regions
};

export class RegionalAttention {
  constructor(graph, scaleLen, params = {}) {
    this.graph = graph;
    this.scaleLen = scaleLen;
    this.params = { ...DEFAULT_ATTN, ...params };
    this.periodMs = this.params.periodMs;
    this.inputHist = new Array(scaleLen).fill(0);
    this.gains = new Map(); // region path -> gain
    this.topRegion = null;
  }

  noteHeard(degree) {
    this.inputHist[((degree % this.scaleLen) + this.scaleLen) % this.scaleLen] += 1;
  }

  gainOf(neuron) {
    return this.gains.get(neuron.region) ?? 1;
  }

  update() {
    const p = this.params;
    for (let i = 0; i < this.inputHist.length; i++) this.inputHist[i] *= p.inputDecay;
    const heard = this.inputHist.reduce((a, b) => a + b, 0);
    if (heard < 0.05 || p.strength <= 0) {
      // nothing recent to attend to — neutral gains
      this.gains.clear();
      this.topRegion = null;
      return;
    }

    const sims = [];
    for (const region of this.graph.leafRegions()) {
      const profile = new Array(this.scaleLen).fill(0);
      for (const id of region.members) {
        const n = this.graph.neurons.get(id);
        if (n && n.role === 'excitatory') profile[n.structDegree % this.scaleLen]++;
      }
      sims.push({ region, sim: cosine(profile, this.inputHist) });
    }
    if (!sims.length) return;
    const maxSim = Math.max(...sims.map((s) => s.sim), 1e-9);

    this.gains.clear();
    let top = null;
    for (const { region, sim } of sims) {
      const rel = sim / maxSim; // 0..1 across regions
      const gain =
        p.bias === 'suppress'
          ? 1 - p.strength * (1 - rel) // best region untouched, rest damped
          : 1 + p.strength * (rel - 0.5); // suppressed … boosted
      this.gains.set(region.path, gain);
      if (!top || gain > top.gain) top = { path: region.path, gain };
      // attention as morphogen: strongly attended regions receive survival
      // energy, so what the player attends to is what develops
      if (gain > 1 + p.strength * 0.3) {
        for (const id of region.members) {
          const n = this.graph.neurons.get(id);
          if (n && n.role !== 'input') {
            n.energy = Math.min(1.5, n.energy + p.energyTrickle * p.strength);
          }
        }
      }
    }
    this.topRegion = top;
  }
}
