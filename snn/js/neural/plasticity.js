// Trace-based STDP (medium timescale). Each neuron carries an exponentially
// decaying spike trace; when a neuron fires:
//   - incoming excitatory synapses whose source fired recently are
//     potentiated (pre-before-post → strengthen)
//   - outgoing excitatory synapses whose target fired recently are
//     depressed (post-before-pre → weaken)
// Slight depression bias keeps runaway potentiation in check; weights are
// clamped. Inhibitory synapses are left untouched.
//
// Deterministic — no RNG. This is the "short-term evidence → weight change"
// half of brief §13; persistent strong weights becoming structure is the
// next experiment on top of this.

export const DEFAULT_STDP = {
  tauMs: 25,
  aPlus: 0.012,
  aMinus: 0.014, // slightly stronger depression for stability
  wMin: 0.05,
  wMax: 1.1,
};

export class STDP {
  constructor(params = {}) {
    this.p = { ...DEFAULT_STDP, ...params };
  }

  readTrace(n, t) {
    if (!n.stdpTraceT && n.stdpTraceT !== 0) return 0;
    return (n.stdpTrace ?? 0) * Math.exp(-(t - n.stdpTraceT) / this.p.tauMs);
  }

  onSpike(graph, n, t) {
    for (const s of graph.incoming.get(n.id) ?? []) {
      if (s.weight <= 0) continue;
      const pre = graph.neurons.get(s.source);
      if (!pre) continue;
      s.weight = Math.min(this.p.wMax, s.weight + this.p.aPlus * this.readTrace(pre, t));
    }
    for (const s of graph.outgoing.get(n.id) ?? []) {
      if (s.weight <= 0) continue;
      const post = graph.neurons.get(s.target);
      if (!post) continue;
      s.weight = Math.max(this.p.wMin, s.weight - this.p.aMinus * this.readTrace(post, t));
    }
    n.stdpTrace = this.readTrace(n, t) + 1;
    n.stdpTraceT = t;
  }
}
