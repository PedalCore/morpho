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
  // STSA-inspired temporal mixing (SpikeVoice, ACL 2024): instead of one
  // recency-decayed context, keep contexts at several timescales and attend
  // over DEPTH — the sharpest (most informative) timescale dominates. Just
  // played something distinct → fast context leads; gone quiet → the slow
  // session-memory context holds. Sequential mixing = the regional gains.
  temporalMix: false,
  timescaleDecays: [0.55, 0.85, 0.97], // ≈0.4 s, ≈1.6 s, ≈8 s half-lives at 250 ms updates
};

export class RegionalAttention {
  constructor(graph, scaleLen, params = {}) {
    this.graph = graph;
    this.scaleLen = scaleLen;
    this.params = { ...DEFAULT_ATTN, ...params };
    this.periodMs = this.params.periodMs;
    this.inputHist = new Array(scaleLen).fill(0);
    // multi-timescale contexts for temporal mixing
    this.hists = this.params.timescaleDecays.map(() => new Array(scaleLen).fill(0));
    this.temporalWeights = this.hists.map(() => 0);
    this.gains = new Map(); // region path -> gain
    this.topRegion = null;
  }

  noteHeard(degree) {
    const d = ((degree % this.scaleLen) + this.scaleLen) % this.scaleLen;
    this.inputHist[d] += 1;
    for (const h of this.hists) h[d] += 1;
  }

  // context to attend against: single recency histogram, or the STSA-style
  // sharpness-weighted mixture over timescales
  _context() {
    if (!this.params.temporalMix) return this.inputHist;
    const mixed = new Array(this.scaleLen).fill(0);
    const weights = [];
    for (const h of this.hists) {
      const sum = h.reduce((a, b) => a + b, 0);
      if (sum < 0.05) {
        weights.push(0);
        continue;
      }
      // sharpness = how peaked this timescale's distribution is (max/mean);
      // a distinct recent phrase out-competes a diffuse long-term memory
      const sharp = Math.max(...h) / (sum / this.scaleLen);
      weights.push(sharp * sum ** 0.25); // slight mass term breaks ties
    }
    const total = weights.reduce((a, b) => a + b, 0);
    if (total <= 0) return this.inputHist;
    this.temporalWeights = weights.map((w) => w / total);
    for (let k = 0; k < this.hists.length; k++) {
      const h = this.hists[k];
      const sum = h.reduce((a, b) => a + b, 0) || 1;
      for (let i = 0; i < this.scaleLen; i++) mixed[i] += (this.temporalWeights[k] * h[i]) / sum;
    }
    return mixed;
  }

  gainOf(neuron) {
    return this.gains.get(neuron.region) ?? 1;
  }

  update() {
    const p = this.params;
    for (let i = 0; i < this.inputHist.length; i++) this.inputHist[i] *= p.inputDecay;
    this.hists.forEach((h, k) => {
      for (let i = 0; i < h.length; i++) h[i] *= p.timescaleDecays[k];
    });
    const context = this._context();
    const heard = context.reduce((a, b) => a + b, 0);
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
      sims.push({ region, sim: cosine(profile, context) });
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
      // energy, so what the player attends to is what develops.
      // (threshold on relative match — gain caps at 1 in suppress mode)
      if (p.energyTrickle > 0 && rel > 0.75) {
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
