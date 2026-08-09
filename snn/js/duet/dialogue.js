// Call-and-response tracking. Segments the session into exchanges by
// silence gaps (you play a phrase → a gap → its answer window), and scores
// each response's *relatedness* to the call: cosine similarity between the
// degree histograms of call and response. Not a memorization metric —
// a relevant answer shares material without copying order.
//
// Sim-time driven (engine step count in ms), so it works identically in the
// browser and in headless experiments.

export function degreeHist(degrees, len = 12) {
  const h = new Array(len).fill(0);
  for (const d of degrees) h[((d % len) + len) % len]++;
  return h;
}

// Inter-onset-interval histogram over log-spaced buckets — rhythm as a
// distribution, comparable with cosine just like pitch material.
export const IOI_BUCKETS = [90, 160, 280, 500, 900, 1600];

export function ioiHist(times) {
  const h = new Array(IOI_BUCKETS.length + 1).fill(0);
  for (let i = 1; i < times.length; i++) {
    const ioi = times[i] - times[i - 1];
    let b = IOI_BUCKETS.findIndex((edge) => ioi < edge);
    if (b === -1) b = IOI_BUCKETS.length;
    h[b]++;
  }
  return h;
}

export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / Math.sqrt(na * nb);
}

export class DialogueTracker {
  constructor({ gapMs = 600, windowMs = 2500, histLen = 12 } = {}) {
    this.gapMs = gapMs;
    this.windowMs = windowMs;
    this.histLen = histLen;
    this.state = 'idle'; // idle | call | response
    this.call = [];
    this.callTimes = [];
    this.resp = [];
    this.respTimes = [];
    this.lastHumanAt = -1e9;
    this.respStart = 0;
    this.exchanges = []; // {t, callNotes, respNotes, score}
    this.maxExchanges = 200;
    this.onExchange = null;
    this.onCallStart = null; // new call begins (reset activity tracking)
    this.onResponseStart = null; // call ended — seed the answer
  }

  humanNote(degree, t) {
    if (this.state === 'response') this.closeExchange(t);
    if (this.state !== 'call') {
      this.callTimes = [];
      if (this.onCallStart) this.onCallStart(t);
    }
    this.state = 'call';
    this.call.push(degree);
    this.callTimes.push(t);
    this.lastHumanAt = t;
  }

  // the call's pace — mean inter-onset interval, for tempo-matched answers
  meanCallIOI() {
    const ts = this.callTimes ?? [];
    if (ts.length < 2) return null;
    let sum = 0;
    for (let i = 1; i < ts.length; i++) sum += ts[i] - ts[i - 1];
    return sum / (ts.length - 1);
  }

  modelNote(degree, t) {
    if (this.state === 'response') {
      this.resp.push(degree);
      this.respTimes.push(t);
    }
  }

  // any network spike during the response window — the energy cost of the
  // answer (SGNNBench-style accounting: spikes per answer)
  spike() {
    if (this.state === 'response') this.respSpikes = (this.respSpikes ?? 0) + 1;
  }

  // is the human mid-phrase right now? (used to gate Q&A audio)
  humanActive(t, holdMs = 350) {
    return t - this.lastHumanAt < holdMs;
  }

  tick(t) {
    if (this.state === 'call' && t - this.lastHumanAt > this.gapMs) {
      this.state = 'response';
      this.respStart = t;
      this.resp = [];
      this.respTimes = [];
      this.respSpikes = 0;
      if (this.onResponseStart) this.onResponseStart(t);
    } else if (this.state === 'response' && t - this.respStart > this.windowMs) {
      this.closeExchange(t);
    }
  }

  closeExchange(t) {
    if (this.call.length) {
      const score = cosine(
        degreeHist(this.call, this.histLen),
        degreeHist(this.resp, this.histLen)
      );
      const rhythm =
        this.callTimes.length > 2 && this.respTimes.length > 2
          ? cosine(ioiHist(this.callTimes), ioiHist(this.respTimes))
          : 0;
      this.exchanges.push({
        t,
        callNotes: this.call.length,
        respNotes: this.resp.length,
        respSpikes: this.respSpikes ?? 0,
        score,
        rhythm,
      });
      if (this.exchanges.length > this.maxExchanges) this.exchanges.shift();
      if (this.onExchange) this.onExchange(this.exchanges[this.exchanges.length - 1]);
    }
    this.call = [];
    this.callTimes = [];
    this.resp = [];
    this.respTimes = [];
    this.respSpikes = 0;
    this.state = 'idle';
  }

  recentMean(n = 10) {
    const scored = this.exchanges.filter((e) => e.respNotes > 0).slice(-n);
    if (!scored.length) return 0;
    return scored.reduce((a, e) => a + e.score, 0) / scored.length;
  }
}
