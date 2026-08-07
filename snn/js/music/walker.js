// Stochastic graph walkers — the plugin briefs' LIVE mode, prototyped here.
//
// Walkers traverse synapses on a musical grid (a subdivision of the input
// pulse), emitting a note at each excitatory neuron they land on. Traversal
// is weighted, not uniform:
//
//   weight = |synapticWeight|
//          × momentum        (prefer staying in / leaving the current region)
//          × repetitionPenalty (recently visited nodes are discouraged)
//   then sharpened by a temperature ("variation") exponent.
//
// Walkers are part of the deterministic simulation (own RNG stream) because
// they feed back into survival: each visit deposits energy on the neuron
// (plugin brief §36 — music affects survival), and occupied neurons are
// protected from pruning (§37).

export const DEFAULT_WALK = {
  count: 2,
  stepDivisor: 2, // walker step = pulsePeriod / divisor (eighth notes)
  variation: 0.45, // 0 → follow strongest path, 1 → near-uniform
  momentum: 0.65, // preference for staying within the current leaf region
  repetitionPenalty: 0.25,
  historyLen: 6,
  teleportProb: 0.03, // occasional phrase jump
};

export class WalkerSystem {
  constructor(graph, rng, params = {}) {
    this.graph = graph;
    this.rng = rng;
    this.params = { ...DEFAULT_WALK, ...params };
    this.walkers = [];
    this.onNote = null; // (neuron, walkerIndex) => void
    // performance steering (plugin brief §23): an attractor in layout space
    // pulls traversal toward it. UI-driven, so runs steered by hand are not
    // seed-reproducible — it is a live instrument control, not an experiment
    // parameter.
    this.attractor = null; // {x, y, strength} in unit layout space
    this.posOf = null; // (neuronId) => {x, y} | null — provided by the UI
    this.setCount(this.params.count);
  }

  setCount(count) {
    this.params.count = count;
    while (this.walkers.length > count) this.walkers.pop();
    while (this.walkers.length < count) {
      this.walkers.push({ at: this.randomStart(), history: [] });
    }
  }

  randomStart() {
    const candidates = [...this.graph.neurons.values()].filter((n) => n.role === 'excitatory');
    if (!candidates.length) return null;
    return candidates[Math.floor(this.rng() * candidates.length)].id;
  }

  occupiedIds() {
    return new Set(this.walkers.map((w) => w.at).filter((id) => id != null));
  }

  // Called every sim step; moves walkers on their musical grid.
  tick(stepCount, pulsePeriodMs) {
    if (!this.params.count) return;
    const period = Math.max(20, Math.round(pulsePeriodMs / this.params.stepDivisor));
    if (stepCount % period !== 0) return;
    this.walkers.forEach((w, i) => this.move(w, i));
  }

  move(w, index) {
    const p = this.params;
    let current = w.at != null ? this.graph.neurons.get(w.at) : null;
    if (!current || this.rng() < p.teleportProb) {
      w.at = this.randomStart();
      current = w.at != null ? this.graph.neurons.get(w.at) : null;
      if (!current) return;
    }

    const edges = this.graph.outgoing
      .get(current.id)
      .filter((s) => s.enabled && this.graph.neurons.has(s.target));
    let next = null;
    if (!edges.length) {
      next = this.graph.neurons.get(this.randomStart());
    } else {
      const temperature = Math.max(0.05, p.variation);
      const scores = edges.map((s) => {
        const target = this.graph.neurons.get(s.target);
        let score = Math.abs(s.weight);
        score *= target.region === current.region ? p.momentum : 1 - p.momentum + 0.05;
        if (w.history.includes(s.target)) score *= p.repetitionPenalty;
        if (target.role === 'input') score *= 0.05;
        if (this.attractor && this.posOf) {
          const pos = this.posOf(s.target);
          if (pos) {
            const d = Math.hypot(pos.x - this.attractor.x, pos.y - this.attractor.y);
            score *= Math.exp(-d * this.attractor.strength);
          }
        }
        return Math.pow(score, 1 / temperature);
      });
      let total = 0;
      for (const s of scores) total += s;
      let roll = this.rng() * total;
      let chosen = edges.length - 1;
      for (let i = 0; i < edges.length; i++) {
        roll -= scores[i];
        if (roll <= 0) {
          chosen = i;
          break;
        }
      }
      next = this.graph.neurons.get(edges[chosen].target);
    }
    if (!next) return;

    w.at = next.id;
    w.history.push(next.id);
    if (w.history.length > p.historyLen) w.history.shift();

    next.walkerVisits++; // survival energy — music keeps structure alive
    if (next.role === 'excitatory' && this.onNote) this.onNote(next, index);
  }
}
