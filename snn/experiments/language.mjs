// Language sideline: the organism as a liquid state machine on tiny
// shakespeare. Characters arrive as spikes; at each character boundary we
// read the reservoir's spike traces and train ONLY a closed-form linear
// readout (ridge regression — backprop-free) to predict the next character.
//
// Honest expectations: a char transformer sits near 1.4–1.6 bits/char. We
// are asking a different question:
//   Q1  does the spiking reservoir add predictive signal beyond a readout
//       on the current character alone (≈ bigram)?
//   Q2  does DEVELOPMENTAL EXPOSURE to the text (dev+STDP while listening,
//       then frozen) make the reservoir a better feature map than the same
//       genotype fresh?
//
// Arms: bigram (exact, add-1), char-only readout, fresh reservoir,
//       exposed reservoir. Metrics: top-1 accuracy, bits/char (softmax over
//       ridge scores, temperature fit on train — approximate calibration).
//
// Run: npm run experiment:language  (fetches ~1MB dataset on first run)

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { Lab } from '../js/sim/lab.js';

const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;

const STEPS_PER_CHAR = 12; // ms of sim per character
const TRACE_TAU = 40; // ms decay of the readout trace
const EXPOSE_CHARS = 15000; // developmental listening (exposed arm)
const TRAIN_CHARS = 30000;
const TEST_CHARS = 5000;
const RIDGE_LAMBDA = 1.0;

async function getText() {
  if (!existsSync(DATA_PATH)) {
    mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
    console.log('fetching tiny shakespeare…');
    const res = await fetch(DATA_URL);
    writeFileSync(DATA_PATH, await res.text());
  }
  return readFileSync(DATA_PATH, 'utf8');
}

function makeOrganism(seed) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: false, developmentEnabled: false },
    walk: { count: 0 },
  });
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

// stream text through the organism; at each boundary optionally emit
// [trace features..., one-hot current char..., 1] with next char as target
function stream(lab, charInputs, ids, span, collect, useReservoir) {
  const neurons = [...lab.graph.neurons.values()].filter((n) => n.role !== 'input');
  const index = new Map(neurons.map((n, i) => [n.id, i]));
  const R = useReservoir ? neurons.length : 0;
  const trace = new Float64Array(neurons.length);
  const decay = Math.exp(-1 / TRACE_TAU);
  const prevOnSpike = lab.engine.onSpike;
  lab.engine.onSpike = (n) => {
    const i = index.get(n.id);
    if (i !== undefined) trace[i] += 1;
  };
  const V = charInputs.length;
  const X = [];
  const y = [];
  for (let p = 0; p < span.length - 1; p++) {
    lab.fireInput(charInputs[ids[span.start + p]]);
    lab.fireInput(charInputs[ids[span.start + p]], 5);
    for (let s = 0; s < STEPS_PER_CHAR; s++) {
      lab.step();
      for (let i = 0; i < trace.length; i++) trace[i] *= decay;
    }
    if (collect) {
      const row = new Float64Array(R + V + 1);
      if (useReservoir) row.set(trace.subarray(0, R), 0);
      row[R + ids[span.start + p]] = 1;
      row[R + V] = 1; // bias
      X.push(row);
      y.push(ids[span.start + p + 1]);
    }
  }
  lab.engine.onSpike = prevOnSpike;
  return { X, y };
}

// ridge: solve (XtX + λI) W = XtY, Gaussian elimination with V right sides
function ridgeFit(X, y, V, lambda) {
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
  // in-place elimination
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
  const W = Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(V);
    for (let v = 0; v < V; v++) row[v] = B[i][v] / (A[i][i] || 1e-12);
    return row;
  });
  return W;
}

function evalReadout(W, X, y, V, temperature) {
  let correct = 0;
  let nll = 0;
  const scores = new Float64Array(V);
  for (let s = 0; s < X.length; s++) {
    scores.fill(0);
    const x = X[s];
    for (let i = 0; i < x.length; i++) {
      if (x[i] === 0) continue;
      const xi = x[i];
      const Wi = W[i];
      for (let v = 0; v < V; v++) scores[v] += xi * Wi[v];
    }
    let best = 0;
    let maxS = -Infinity;
    for (let v = 0; v < V; v++) {
      if (scores[v] > maxS) {
        maxS = scores[v];
        best = v;
      }
    }
    if (best === y[s]) correct++;
    let z = 0;
    for (let v = 0; v < V; v++) z += Math.exp((scores[v] - maxS) * temperature);
    nll += -Math.log(Math.exp((scores[y[s]] - maxS) * temperature) / z);
  }
  return { acc: correct / X.length, bpc: nll / X.length / Math.LN2 };
}

function fitTemperature(W, X, y, V) {
  let best = { bpc: Infinity, t: 1 };
  for (const t of [0.5, 1, 2, 4, 8, 16]) {
    const r = evalReadout(W, X.slice(0, 2000), y.slice(0, 2000), V, t);
    if (r.bpc < best.bpc) best = { bpc: r.bpc, t };
  }
  return best.t;
}

function bigramBaseline(ids, V, trainSpan, testSpan) {
  const counts = Array.from({ length: V }, () => new Float64Array(V).fill(1)); // add-1
  for (let p = trainSpan.start; p < trainSpan.start + trainSpan.length - 1; p++) {
    counts[ids[p]][ids[p + 1]]++;
  }
  const rows = counts.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map((c) => c / sum);
  });
  let nll = 0;
  let correct = 0;
  let n = 0;
  for (let p = testSpan.start; p < testSpan.start + testSpan.length - 1; p++) {
    const row = rows[ids[p]];
    nll += -Math.log(row[ids[p + 1]]);
    let best = 0;
    for (let v = 1; v < V; v++) if (row[v] > row[best]) best = v;
    if (best === ids[p + 1]) correct++;
    n++;
  }
  return { acc: correct / n, bpc: nll / n / Math.LN2 };
}

function arm(name, seed, { expose, useReservoir }, chars, ids, spans) {
  const lab = makeOrganism(seed);
  const charInputs = wireCharInputs(lab, chars.length);
  if (expose) {
    lab.simParams.developmentEnabled = true;
    lab.simParams.stdpEnabled = true;
    lab.stdp.p.tauMs = 60;
    stream(lab, charInputs, ids, spans.expose, false, false);
    lab.simParams.developmentEnabled = false;
    lab.simParams.stdpEnabled = false;
  }
  const train = stream(lab, charInputs, ids, spans.train, true, useReservoir);
  const test = stream(lab, charInputs, ids, spans.test, true, useReservoir);
  const W = ridgeFit(train.X, train.y, chars.length, RIDGE_LAMBDA);
  const t = fitTemperature(W, train.X, train.y, chars.length);
  const r = evalReadout(W, test.X, test.y, chars.length, t);
  console.log(
    `${name.padEnd(18)} acc ${(r.acc * 100).toFixed(1)}%   ${r.bpc.toFixed(3)} bits/char   (neurons ${lab.graph.neurons.size})`
  );
  return r;
}

const text = (await getText()).slice(0, 200000);
const chars = [...new Set(text)].sort();
const ids = Array.from(text, (c) => chars.indexOf(c));
console.log(`tiny shakespeare: vocab ${chars.length}, using ${text.length} chars\n`);

const spans = {
  expose: { start: 0, length: EXPOSE_CHARS },
  train: { start: EXPOSE_CHARS, length: TRAIN_CHARS },
  test: { start: EXPOSE_CHARS + TRAIN_CHARS, length: TEST_CHARS },
};

console.log(`uniform            ${Math.log2(chars.length).toFixed(3)} bits/char`);
const bg = bigramBaseline(ids, chars.length, spans.train, spans.test);
console.log(`bigram (exact)     acc ${(bg.acc * 100).toFixed(1)}%   ${bg.bpc.toFixed(3)} bits/char`);
const SEED = Number(process.argv[2] ?? 42);
arm('char-only readout', SEED, { expose: false, useReservoir: false }, chars, ids, spans);
arm('fresh reservoir', SEED, { expose: false, useReservoir: true }, chars, ids, spans);
arm('exposed reservoir', SEED, { expose: true, useReservoir: true }, chars, ids, spans);
console.log(
  '\nreference: char transformers ≈ 1.4–1.6 bits/char on this dataset.' +
    '\nQ1 = reservoir arms below bigram bpc?  Q2 = exposed below fresh?'
);
