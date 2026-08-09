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
  // reward mode (R-STDP, Izhikevich 2007 "distal reward"): pairings write to
  // per-synapse eligibility traces instead of weights; traces decay over
  // ~seconds and convert to weight change only when reward arrives. Bridges
  // the gap between note-to-note timing and delayed musical judgment.
  mode: 'immediate', // 'immediate' | 'reward'
  eligTauMs: 1500,
  rewardRate: 1.0,
};

export class STDP {
  constructor(params = {}) {
    this.p = { ...DEFAULT_STDP, ...params };
    this.eligibility = new Map(); // synapseId -> {e, t}
  }

  readTrace(n, t) {
    if (!n.stdpTraceT && n.stdpTraceT !== 0) return 0;
    return (n.stdpTrace ?? 0) * Math.exp(-(t - n.stdpTraceT) / this.p.tauMs);
  }

  _apply(s, dw, t) {
    if (this.p.mode === 'reward') {
      const rec = this.eligibility.get(s.id);
      const e = rec ? rec.e * Math.exp(-(t - rec.t) / this.p.eligTauMs) : 0;
      this.eligibility.set(s.id, { e: e + dw, t });
    } else if (dw > 0) {
      s.weight = Math.min(this.p.wMax, s.weight + dw);
    } else {
      s.weight = Math.max(this.p.wMin, s.weight + dw);
    }
  }

  onSpike(graph, n, t) {
    for (const s of graph.incoming.get(n.id) ?? []) {
      if (s.weight <= 0) continue;
      const pre = graph.neurons.get(s.source);
      if (!pre) continue;
      this._apply(s, this.p.aPlus * this.readTrace(pre, t), t);
    }
    for (const s of graph.outgoing.get(n.id) ?? []) {
      if (s.weight <= 0) continue;
      const post = graph.neurons.get(s.target);
      if (!post) continue;
      this._apply(s, -this.p.aMinus * this.readTrace(post, t), t);
    }
    n.stdpTrace = this.readTrace(n, t) + 1;
    n.stdpTraceT = t;
  }

  // reward mode: convert decayed eligibility into weight change. amount may
  // be negative (punishment). Stale traces are dropped.
  applyReward(graph, t, amount = 1) {
    if (this.p.mode !== 'reward') return 0;
    let touched = 0;
    for (const [id, rec] of this.eligibility) {
      const e = rec.e * Math.exp(-(t - rec.t) / this.p.eligTauMs);
      if (Math.abs(e) < 1e-5) {
        this.eligibility.delete(id);
        continue;
      }
      const s = graph.synapses.get(id);
      if (!s) {
        this.eligibility.delete(id);
        continue;
      }
      const dw = this.p.rewardRate * amount * e;
      s.weight = Math.max(this.p.wMin, Math.min(this.p.wMax, s.weight + dw));
      rec.e = 0; // consumed
      rec.t = t;
      touched++;
    }
    return touched;
  }
}
