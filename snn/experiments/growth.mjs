// Error-driven growth on language: does the organism grow ITSELF toward the
// capacity the task needs?
//
// Mechanism ("error sets the growth budget, activity decides where"):
//   - an ONLINE linear softmax readout (delta rule — no backprop into the
//     network) predicts the next character as text streams through
//   - the rolling error rate modulates development's growth probability:
//     wrong answers → capacity pressure, low error → growth stops
//   - correct predictions deposit survival energy on the neurons that were
//     just active (what helped, lives; what didn't, starves and is pruned)
//
// Arms (same genotype, 100k chars of online exposure):
//   frozen        — no development at all
//   error-driven  — growth gated by rolling error, correctness feeds energy
//   always-grow   — maximum growth pressure regardless of error (control:
//                   is it error GATING that matters, or just more neurons?)
//
// After exposure each organism is frozen and evaluated identically: fresh
// ridge readout fit on 25k held-out-from-training chars, tested on 5k.
// Also: a Forward-Forward readout head (Hinton 2022; arXiv:2502.20411 for
// SNNs) trained on the same final features, layer-local, no backprop chain.
//
// Run: npm run experiment:growth   (~15 min)

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { Lab } from '../js/sim/lab.js';
import { mulberry32 } from '../js/core/rng.js';

const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;

const STEPS_PER_CHAR = 12;
const TRACE_TAU = 40;
const ONLINE_CHARS = 100000; // scaled-up exposure
const FIT_CHARS = 25000;
const TEST_CHARS = 5000;
const MAX_NEURONS = Number(process.argv[3] ?? 600);
const MAX_SLOTS = Math.max(900, MAX_NEURONS + 400); // stable feature slots (pruned slots retire, never reused)
const SEED = Number(process.argv[2] ?? 42);

async function getText() {
  if (!existsSync(DATA_PATH)) {
    mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
    const res = await fetch(DATA_URL);
    writeFileSync(DATA_PATH, await res.text());
  }
  return readFileSync(DATA_PATH, 'utf8');
}

function makeOrganism(seed, devMode) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0 },
    sim: {
      pulseFireProb: 0,
      backgroundHz: 0,
      stdpEnabled: true,
      developmentEnabled: devMode !== 'frozen',
    },
    dev: { maxNeurons: MAX_NEURONS, maxSynapses: 9000, maxGrowPerEpoch: 5 },
    walk: { count: 0 },
  });
  lab.stdp.p.tauMs = 60;
  return lab;
}

function wireCharInputs(lab, V) {
  const rng = lab.streams.build;
  lab.graph.addRegion('IN', 0, 'input', null);
  const excitatory = [...lab.graph.neurons.values()].filter((n) => n.role === 'excitatory');
  const inputs = [];
  for (let c = 0; c < V; c++) {
    const n = lab.graph.addNeuron({ role: 'input', region: 'IN' });
    lab.graph.regions.get('IN').members.add(n.id);
    for (let k = 0; k < 8; k++) {
      const t = excitatory[Math.floor(rng() * excitatory.length)];
      lab.graph.addSynapse({
        source: n.id,
        target: t.id,
        weight: 0.6 + rng() * 0.35,
        delaySteps: 1 + Math.floor(rng() * 3),
      });
    }
    inputs.push(n.id);
  }
  lab.inputIds = inputs;
  return inputs;
}

// stable feature slots that survive growth/pruning
function makeSlots() {
  return { map: new Map(), next: 0 };
}
function slotOf(slots, id) {
  let s = slots.map.get(id);
  if (s === undefined && slots.next < MAX_SLOTS) {
    s = slots.next++;
    slots.map.set(id, s);
  }
  return s;
}

function onlineExposure(lab, charInputs, ids, span, devMode, V) {
  const slots = makeSlots();
  const trace = new Float64Array(MAX_SLOTS);
  const decay = Math.exp(-1 / TRACE_TAU);
  const recentlyActive = [];
  lab.engine.onSpike = (n) => {
    if (n.role === 'input') return;
    const s = slotOf(slots, n.id);
    if (s !== undefined) {
      trace[s] += 1;
      recentlyActive.push(n.id);
      if (recentlyActive.length > 400) recentlyActive.shift();
    }
  };

  // online softmax readout: W[v] rows over [slots..., char one-hot..., bias]
  const d = MAX_SLOTS + V + 1;
  const W = Array.from({ length: V }, () => new Float64Array(d));
  const x = new Float64Array(d);
  const p = new Float64Array(V);
  let lr = 0.05;
  let errEMA = 1.0;
  const baseGrow = lab.dev.params.growProb;
  const curve = [];

  for (let pos = 0; pos < span.length - 1; pos++) {
    const c = ids[span.start + pos];
    lab.fireInput(charInputs[c]);
    lab.fireInput(charInputs[c], 5);
    for (let s = 0; s < STEPS_PER_CHAR; s++) {
      lab.step();
      for (let i = 0; i < slots.next; i++) trace[i] *= decay;
    }
    // features
    x.fill(0);
    for (let i = 0; i < slots.next; i++) x[i] = trace[i];
    x[MAX_SLOTS + c] = 1;
    x[d - 1] = 1;
    // predict
    let maxScore = -Infinity;
    let pred = 0;
    for (let v = 0; v < V; v++) {
      let s = 0;
      const Wv = W[v];
      for (let i = 0; i < slots.next; i++) if (x[i]) s += Wv[i] * x[i];
      s += Wv[MAX_SLOTS + c] + Wv[d - 1];
      p[v] = s;
      if (s > maxScore) {
        maxScore = s;
        pred = v;
      }
    }
    let z = 0;
    for (let v = 0; v < V; v++) {
      p[v] = Math.exp(p[v] - maxScore);
      z += p[v];
    }
    const target = ids[span.start + pos + 1];
    const correct = pred === target;
    errEMA = 0.999 * errEMA + 0.001 * (correct ? 0 : 1);
    // delta rule on the readout only
    for (let v = 0; v < V; v++) {
      const g = p[v] / z - (v === target ? 1 : 0);
      if (!g) continue;
      const Wv = W[v];
      const step = lr * g;
      for (let i = 0; i < slots.next; i++) if (x[i]) Wv[i] -= step * x[i];
      Wv[MAX_SLOTS + c] -= step;
      Wv[d - 1] -= step;
    }
    lr = 0.05 / (1 + pos / 30000);

    if (devMode === 'error') {
      // error sets the growth budget…
      lab.dev.params.growProb = Math.min(0.95, baseGrow * (0.1 + 2.2 * errEMA));
      // …and correctness feeds the circuits that were just active
      if (correct && pos % 3 === 0) {
        const now = lab.engine.stepCount;
        for (const id of recentlyActive) {
          const n = lab.graph.neurons.get(id);
          if (n && now - n.lastSpikeStep < 150) {
            n.energy = Math.min(1.5, n.energy + 0.04);
          }
        }
      }
    } else if (devMode === 'always') {
      lab.dev.params.growProb = 0.95;
    }

    if ((pos + 1) % 10000 === 0) {
      curve.push({
        chars: pos + 1,
        err: errEMA,
        neurons: lab.graph.neurons.size,
      });
    }
  }
  lab.engine.onSpike = null;
  return { curve, errEMA };
}

// ---- frozen evaluation: fresh ridge on final architecture ----

function collect(lab, charInputs, ids, span, V) {
  const neurons = [...lab.graph.neurons.values()].filter((n) => n.role !== 'input');
  const index = new Map(neurons.map((n, i) => [n.id, i]));
  const R = neurons.length;
  const trace = new Float64Array(R);
  const decay = Math.exp(-1 / TRACE_TAU);
  lab.engine.onSpike = (n) => {
    const i = index.get(n.id);
    if (i !== undefined) trace[i] += 1;
  };
  const X = [];
  const y = [];
  for (let p = 0; p < span.length - 1; p++) {
    const c = ids[span.start + p];
    lab.fireInput(charInputs[c]);
    lab.fireInput(charInputs[c], 5);
    for (let s = 0; s < STEPS_PER_CHAR; s++) {
      lab.step();
      for (let i = 0; i < R; i++) trace[i] *= decay;
    }
    const row = new Float64Array(R + V + 1);
    row.set(trace, 0);
    row[R + c] = 1;
    row[R + V] = 1;
    X.push(row);
    y.push(ids[span.start + p + 1]);
  }
  lab.engine.onSpike = null;
  return { X, y };
}

function ridgeFit(X, y, V, lambda = 1.0) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(V));
  for (let s = 0; s < X.length; s++) {
    const x = X[s];
    for (let i = 0; i < d; i++) {
      if (x[i] === 0) continue;
      const xi = x[i];
      const Ai = A[i];
      for (let j = i; j < d; j++) Ai[j] += xi * x[j];
      B[i][y[s]] += xi;
    }
  }
  for (let i = 0; i < d; i++) {
    A[i][i] += lambda;
    for (let j = 0; j < i; j++) A[i][j] = A[j][i];
  }
  for (let col = 0; col < d; col++) {
    let piv = col;
    for (let r = col + 1; r < d; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [B[col], B[piv]] = [B[piv], B[col]];
    const diag = A[col][col] || 1e-12;
    for (let r = 0; r < d; r++) {
      if (r === col) continue;
      const f = A[r][col] / diag;
      if (!f) continue;
      for (let j = col; j < d; j++) A[r][j] -= f * A[col][j];
      for (let v = 0; v < V; v++) B[r][v] -= f * B[col][v];
    }
  }
  return Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(V);
    for (let v = 0; v < V; v++) row[v] = B[i][v] / (A[i][i] || 1e-12);
    return row;
  });
}

function ridgeAcc(W, X, y, V) {
  let correct = 0;
  const scores = new Float64Array(V);
  for (let s = 0; s < X.length; s++) {
    scores.fill(0);
    const x = X[s];
    for (let i = 0; i < x.length; i++) {
      if (x[i] === 0) continue;
      const Wi = W[i];
      for (let v = 0; v < V; v++) scores[v] += x[i] * Wi[v];
    }
    let best = 0;
    for (let v = 1; v < V; v++) if (scores[v] > scores[best]) best = v;
    if (best === y[s]) correct++;
  }
  return correct / X.length;
}

// ---- Forward-Forward readout head (layer-local, no backprop chain) ----
// Label is written into the label slots of the input; positive = true label,
// negative = random wrong label. Goodness = Σh²; classify = argmax goodness.

function ffTrain(X, y, V, R, { hidden = 64, epochs = 3, lr = 0.003, seed = 1 } = {}) {
  const rng = mulberry32(seed);
  const d = X[0].length;
  const W = Array.from({ length: hidden }, () => {
    const row = new Float64Array(d);
    for (let i = 0; i < d; i++) row[i] = (rng() - 0.5) * 0.1;
    return row;
  });
  const theta = hidden;
  const h = new Float64Array(hidden);
  const forward = (x, label) => {
    let goodness = 0;
    for (let u = 0; u < hidden; u++) {
      let a = 0;
      const Wu = W[u];
      for (let i = 0; i < R; i++) if (x[i]) a += Wu[i] * x[i];
      a += Wu[R + label] + Wu[d - 1]; // label slot + bias
      h[u] = a > 0 ? a : 0;
      goodness += h[u] * h[u];
    }
    return goodness;
  };
  const update = (x, label, sign) => {
    const g = forward(x, label);
    const pLike = 1 / (1 + Math.exp(-(g - theta)));
    const dLdg = sign > 0 ? pLike - 1 : pLike; // push pos up, neg down
    for (let u = 0; u < hidden; u++) {
      if (h[u] <= 0) continue;
      const f = lr * dLdg * 2 * h[u];
      const Wu = W[u];
      for (let i = 0; i < R; i++) if (x[i]) Wu[i] -= f * x[i];
      Wu[R + label] -= f;
      Wu[d - 1] -= f;
    }
  };
  for (let e = 0; e < epochs; e++) {
    for (let s = 0; s < X.length; s++) {
      update(X[s], y[s], +1);
      let wrong = Math.floor(rng() * V);
      if (wrong === y[s]) wrong = (wrong + 1) % V;
      update(X[s], wrong, -1);
    }
  }
  return { classify: (x) => {
    let best = 0;
    let bg = -Infinity;
    for (let v = 0; v < V; v++) {
      const g = forward(x, v);
      if (g > bg) {
        bg = g;
        best = v;
      }
    }
    return best;
  } };
}

// ---- run ----

const text = (await getText()).slice(0, 250000);
const chars = [...new Set(text)].sort();
const V = chars.length;
const ids = Array.from(text, (c) => chars.indexOf(c));
const spans = {
  online: { start: 0, length: ONLINE_CHARS },
  fit: { start: ONLINE_CHARS, length: FIT_CHARS },
  test: { start: ONLINE_CHARS + FIT_CHARS, length: TEST_CHARS },
};
console.log(`vocab ${V}, online ${ONLINE_CHARS} chars, seed ${SEED}\n`);

for (const devMode of ['frozen', 'error', 'always']) {
  const lab = makeOrganism(SEED, devMode);
  const charInputs = wireCharInputs(lab, V);
  const t0 = Date.now();
  const { curve } = onlineExposure(lab, charInputs, ids, spans.online, devMode, V);
  // freeze and evaluate
  lab.simParams.developmentEnabled = false;
  lab.simParams.stdpEnabled = false;
  const fit = collect(lab, charInputs, ids, spans.fit, V);
  const test = collect(lab, charInputs, ids, spans.test, V);
  const W = ridgeFit(fit.X, fit.y, V);
  const acc = ridgeAcc(W, test.X, test.y, V);
  const R = fit.X[0].length - V - 1;
  const ff = ffTrain(fit.X, fit.y, V, R, { seed: SEED });
  let ffCorrect = 0;
  for (let s = 0; s < test.X.length; s++) {
    if (ff.classify(test.X[s]) === test.y[s]) ffCorrect++;
  }
  console.log(`== ${devMode} ==  (${((Date.now() - t0) / 60000).toFixed(1)} min)`);
  console.log(
    '  growth curve: ' +
      curve.map((c) => `${c.chars / 1000}k:${c.neurons}n/${(c.err * 100).toFixed(0)}%err`).join('  ')
  );
  console.log(
    `  final: ${lab.graph.neurons.size} neurons, ${lab.graph.synapses.size} synapses  |  ridge acc ${(acc * 100).toFixed(1)}%  |  FF acc ${((ffCorrect / test.X.length) * 100).toFixed(1)}%\n`
  );
}
console.log('error-driven success = beats frozen, and matches/beats always-grow with fewer neurons');
