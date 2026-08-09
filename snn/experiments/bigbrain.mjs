// BIG BRAIN: start from a KNOWN deep structure, massively overprovisioned,
// and let error-credit pruning shrink it — development run in reverse.
//
// This is the opposite regime from the musical organism (which grows from
// ~150 neurons). Here we build a deep layered spiking network at 10^5 scale
// with a high-performance SoA engine (typed arrays, CSR synapses, lazy
// exponential decay, event-driven — neurons are only touched when spikes
// reach them), stream tiny shakespeare through it, and prune:
//
//   structure ("known network"): L layers, feedforward + intra-layer
//     recurrence + skip connections + feedback + inhibition, homeostatic
//     per-layer thresholds (without which deep spiking layers die or seize)
//   credit: neurons that were active when the online readout predicted the
//     next character CORRECTLY accumulate credit; active-during-error earns
//     nothing. Every round, the lowest-credit fraction is pruned.
//   evaluation per round: fresh ridge readout on sampled feature neurons,
//     held-out accuracy — an accuracy-vs-size curve of the pruning descent.
//
// Run: npm run experiment:bigbrain [seed] [neurons]   (default 120000)

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { mulberry32 } from '../js/core/rng.js';

const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;

const SEED = Number(process.argv[2] ?? 42);
const N_TARGET = Number(process.argv[3] ?? 120000);
const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const TRACE_TAU = 40;
const N_FEATURES = 1024;
const CALIB_CHARS = 5000;
const CREDIT_CHARS = 12000; // per pruning round
const FIT_CHARS = 12000;
const TEST_CHARS = 2000;
const PRUNE_ROUNDS = 5;
const PRUNE_FRAC = 0.25; // per role, per round
const TARGET_RATE = 0.002; // ~2 Hz — at 10^5 scale activity must be very sparse

async function getText() {
  if (!existsSync(DATA_PATH)) {
    mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
    const res = await fetch(DATA_URL);
    writeFileSync(DATA_PATH, await res.text());
  }
  return readFileSync(DATA_PATH, 'utf8');
}

// ---------- build the known structure ----------

function buildBrain(seed, N) {
  const rng = mulberry32(seed);
  const L = 4;
  const layerSize = Math.floor(N / L);
  const layer = new Uint8Array(N);
  for (let i = 0; i < N; i++) layer[i] = Math.min(L - 1, Math.floor(i / layerSize));
  const inhibitory = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (rng() < 0.15) inhibitory[i] = 1;

  // synapse counts per neuron by rule
  const FF = 14, REC = 6, SKIP = 3, FB = 2, INH = 22;
  const layerStart = (l) => l * layerSize;
  const pickIn = (l) => layerStart(l) + Math.floor(rng() * layerSize);

  const srcs = [];
  const tgts = [];
  const ws = [];
  const dls = [];
  const wE = () => 0.28 + rng() * 0.22;
  const wI = () => -(0.5 + rng() * 0.3);
  for (let i = 0; i < N; i++) {
    const l = layer[i];
    const inh = inhibitory[i];
    const add = (t, w, d) => {
      srcs.push(i);
      tgts.push(t);
      ws.push(w);
      dls.push(d);
    };
    if (inh) {
      for (let k = 0; k < INH; k++) add(pickIn(l), wI(), 1 + Math.floor(rng() * 3));
    } else {
      if (l + 1 < L) for (let k = 0; k < FF; k++) add(pickIn(l + 1), wE(), 1 + Math.floor(rng() * 4));
      for (let k = 0; k < REC; k++) add(pickIn(l), wE() * 0.45, 1 + Math.floor(rng() * 6));
      if (l + 2 < L) for (let k = 0; k < SKIP; k++) add(pickIn(l + 2), wE(), 2 + Math.floor(rng() * 6));
      if (l > 0) for (let k = 0; k < FB; k++) add(pickIn(l - 1), wE() * 0.3, 3 + Math.floor(rng() * 8));
    }
  }
  // CSR
  const M = srcs.length;
  const synStart = new Int32Array(N + 1);
  for (let s = 0; s < M; s++) synStart[srcs[s] + 1]++;
  for (let i = 0; i < N; i++) synStart[i + 1] += synStart[i];
  const synTgt = new Int32Array(M);
  const synW = new Float32Array(M);
  const synDelay = new Uint8Array(M);
  const cursor = synStart.slice(0, N);
  for (let s = 0; s < M; s++) {
    const i = srcs[s];
    const p = cursor[i]++;
    synTgt[p] = tgts[s];
    synW[p] = ws[s];
    synDelay[p] = dls[s];
  }
  return { N, L, layerSize, layer, inhibitory, synStart, synTgt, synW, synDelay, M };
}

// ---------- engine (SoA, event-driven, lazy decay) ----------

function makeEngine(brain) {
  const { N } = brain;
  const v = new Float32Array(N);
  const lastT = new Int32Array(N);
  const refracUntil = new Int32Array(N);
  const alive = new Uint8Array(N).fill(1);
  const thrLayer = new Float32Array(brain.L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let d = 0; d < 256; d++) decayPow[d] = Math.exp(-d / TAU);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const spikesNow = new Int32Array(N); // ids of neurons that spiked this step
  const layerSpikes = new Float64Array(brain.L);

  let t = 0;
  function deliver(target, w) {
    ring[(t + 1) % MAX_DELAY].push(target, w);
  }
  function deliverAt(target, w, delay) {
    ring[(t + Math.min(delay, MAX_DELAY - 1)) % MAX_DELAY].push(target, w);
  }
  function step(onSpike) {
    const bucket = ring[t % MAX_DELAY];
    let nSpikes = 0;
    for (let b = 0; b < bucket.length; b += 2) {
      const i = bucket[b];
      if (!alive[i]) continue;
      if (t < refracUntil[i]) continue;
      const dt = t - lastT[i];
      if (dt > 0) {
        v[i] *= decayPow[dt > 255 ? 255 : dt];
        lastT[i] = t;
      }
      v[i] += bucket[b + 1];
      const l = brain.layer[i];
      const cap = thrLayer[l] * 3;
      if (v[i] > cap) v[i] = cap;
      if (v[i] >= thrLayer[l]) {
        v[i] = 0;
        refracUntil[i] = t + REFRAC;
        spikesNow[nSpikes++] = i;
        layerSpikes[l]++;
        const s0 = brain.synStart[i];
        const s1 = brain.synStart[i + 1];
        for (let s = s0; s < s1; s++) {
          deliverAt(brain.synTgt[s], brain.synW[s], brain.synDelay[s]);
        }
        if (onSpike) onSpike(i);
      }
    }
    bucket.length = 0;
    t++;
    return nSpikes;
  }
  return {
    v, alive, thrLayer, layerSpikes, spikesNow,
    deliver, deliverAt, step,
    now: () => t,
  };
}

// ---------- run ----------

const text = await getText(); // the WHOLE corpus this time
const chars = [...new Set(text)].sort();
const V = chars.length;
const ids = new Int32Array(text.length);
for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);

console.log(
  `corpus: full tiny shakespeare, ${text.length} chars, vocab ${V} · brain: ${N_TARGET} neurons target, seed ${SEED}`
);
const t0 = Date.now();
const brain = buildBrain(SEED, N_TARGET);
const eng = makeEngine(brain);
console.log(`built: ${brain.N} neurons, ${(brain.M / 1e6).toFixed(2)}M synapses, ${brain.L} layers (${(Date.now() - t0) / 1000 | 0}s)\n`);

// input wiring: char c → 250 random layer-0 targets
const inRng = mulberry32(SEED ^ 0xabc);
const inputFan = 250;
const inputTgt = new Int32Array(V * inputFan);
for (let c = 0; c < V; c++) {
  for (let k = 0; k < inputFan; k++) {
    inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize); // layer 0
  }
}
function injectChar(c) {
  const base = c * inputFan;
  for (let k = 0; k < inputFan; k++) eng.deliver(inputTgt[base + k], 1.2);
}

// feature sampling: N_FEATURES neurons, weighted toward deeper layers
const featRng = mulberry32(SEED ^ 0xf00d);
const featIds = new Int32Array(N_FEATURES);
const featSlot = new Int32Array(brain.N).fill(-1);
for (let f = 0; f < N_FEATURES; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  const id = l * brain.layerSize + Math.floor(featRng() * brain.layerSize);
  featIds[f] = id;
  featSlot[id] = f;
}
const trace = new Float64Array(N_FEATURES);
const traceDecay = Math.exp(-1 / TRACE_TAU);

const credit = new Float64Array(brain.N);
const activeThisChar = [];
let recordActive = false;

const onSpike = (i) => {
  const f = featSlot[i];
  if (f >= 0) trace[f] += 1;
  if (recordActive) activeThisChar.push(i);
};

function runChar(c) {
  injectChar(c);
  for (let s = 0; s < STEPS_PER_CHAR; s++) {
    eng.step(onSpike);
    for (let i = 0; i < N_FEATURES; i++) trace[i] *= traceDecay;
  }
}

// homeostasis: per-layer thresholds toward target rate — runs continuously
const aliveCountL = new Float64Array(brain.L);
function homeostat(windowChars, exponent) {
  aliveCountL.fill(0);
  for (let i = 0; i < brain.N; i++) if (eng.alive[i]) aliveCountL[brain.layer[i]]++;
  for (let l = 0; l < brain.L; l++) {
    if (!aliveCountL[l]) continue;
    const rate = eng.layerSpikes[l] / (aliveCountL[l] * windowChars * STEPS_PER_CHAR);
    const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
    eng.thrLayer[l] = Math.max(0.5, Math.min(25, eng.thrLayer[l] * Math.pow(factor, exponent)));
    eng.layerSpikes[l] = 0;
  }
}
console.log('calibrating homeostatic thresholds…');
let pos = 0;
for (let cchar = 0; cchar < CALIB_CHARS; cchar++) {
  runChar(ids[pos++]);
  if ((cchar + 1) % 100 === 0) homeostat(100, 0.6);
}
{
  eng.layerSpikes.fill(0);
  for (let k = 0; k < 250; k++) runChar(ids[pos++]);
  const aliveCountL = new Float64Array(brain.L);
  for (let i = 0; i < brain.N; i++) if (eng.alive[i]) aliveCountL[brain.layer[i]]++;
  const rates = [...eng.layerSpikes].map((sp, l) => ((sp / (aliveCountL[l] * 250 * STEPS_PER_CHAR)) * 1000).toFixed(1));
  console.log(
    `  thresholds: [${[...eng.thrLayer].map((x) => x.toFixed(2)).join(', ')}]  layer rates [${rates.join(', ')}] Hz  (${((Date.now() - t0) / 1000) | 0}s, ${(CALIB_CHARS / ((Date.now() - t0) / 1000)).toFixed(0)} chars/s)\n`
  );
}

// online readout for credit assignment
const dOnline = N_FEATURES + V + 1;
const Won = Array.from({ length: V }, () => new Float64Array(dOnline));
const px = new Float64Array(V);
function onlinePredictLearn(c, target, lr) {
  let maxs = -Infinity;
  let pred = 0;
  for (let vv = 0; vv < V; vv++) {
    let s = 0;
    const Wv = Won[vv];
    for (let i = 0; i < N_FEATURES; i++) if (trace[i]) s += Wv[i] * trace[i];
    s += Wv[N_FEATURES + c] + Wv[dOnline - 1];
    px[vv] = s;
    if (s > maxs) {
      maxs = s;
      pred = vv;
    }
  }
  let z = 0;
  for (let vv = 0; vv < V; vv++) {
    px[vv] = Math.exp(px[vv] - maxs);
    z += px[vv];
  }
  for (let vv = 0; vv < V; vv++) {
    const g = px[vv] / z - (vv === target ? 1 : 0);
    if (!g) continue;
    const Wv = Won[vv];
    const stepw = lr * g;
    for (let i = 0; i < N_FEATURES; i++) if (trace[i]) Wv[i] -= stepw * trace[i];
    Wv[N_FEATURES + c] -= stepw;
    Wv[dOnline - 1] -= stepw;
  }
  return pred === target;
}

// ridge eval on sampled features
function collect(startPos, nChars) {
  const X = [];
  const y = [];
  let p = startPos;
  for (let k = 0; k < nChars; k++) {
    const c = ids[p];
    runChar(c);
    const row = new Float64Array(N_FEATURES + V + 1);
    row.set(trace, 0);
    row[N_FEATURES + c] = 1;
    row[N_FEATURES + V] = 1;
    X.push(row);
    y.push(ids[p + 1]);
    p++;
  }
  return { X, y, end: p };
}

function ridge(X, y, lambda = 1.0) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(V));
  for (let s = 0; s < X.length; s++) {
    const x = X[s];
    for (let i = 0; i < d; i++) {
      if (!x[i]) continue;
      const Ai = A[i];
      for (let j = i; j < d; j++) Ai[j] += x[i] * x[j];
      B[i][y[s]] += x[i];
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
      for (let vv = 0; vv < V; vv++) B[r][vv] -= f * B[col][vv];
    }
  }
  return Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(V);
    for (let vv = 0; vv < V; vv++) row[vv] = B[i][vv] / (A[i][i] || 1e-12);
    return row;
  });
}

function accOf(W, X, y) {
  let correct = 0;
  const sc = new Float64Array(V);
  for (let s = 0; s < X.length; s++) {
    sc.fill(0);
    const x = X[s];
    for (let i = 0; i < x.length; i++) {
      if (!x[i]) continue;
      const Wi = W[i];
      for (let vv = 0; vv < V; vv++) sc[vv] += x[i] * Wi[vv];
    }
    let best = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[best]) best = vv;
    if (best === y[s]) correct++;
  }
  return correct / X.length;
}

function aliveCount() {
  let n = 0;
  for (let i = 0; i < brain.N; i++) if (eng.alive[i]) n++;
  return n;
}

// ---------- pruning descent ----------

for (let round = 0; round <= PRUNE_ROUNDS; round++) {
  // credit phase
  credit.fill(0);
  let rollingErr = 0;
  let seen = 0;
  let spikeSum = 0;
  recordActive = true;
  for (let k = 0; k < CREDIT_CHARS; k++) {
    activeThisChar.length = 0;
    const c = ids[pos];
    runChar(c);
    const correct = onlinePredictLearn(c, ids[pos + 1], 0.03 / (1 + k / 20000));
    if (correct) {
      for (const i of activeThisChar) credit[i] += 1;
    } else {
      for (const i of activeThisChar) if (brain.inhibitory[i]) credit[i] += 0.5;
    }
    spikeSum += activeThisChar.length;
    if ((k + 1) % 500 === 0) homeostat(500, 0.15);
    rollingErr += correct ? 0 : 1;
    seen++;
    pos++;
  }
  recordActive = false;

  // evaluate frozen
  const fit = collect(pos, FIT_CHARS);
  const test = collect(fit.end, TEST_CHARS);
  pos = test.end;
  const W = ridge(fit.X, fit.y);
  const acc = accOf(W, test.X, test.y);
  const nAlive = aliveCount();
  console.log(
    `round ${round}: ${nAlive} neurons alive · ${(spikeSum / seen).toFixed(0)} spikes/char · online err ${((rollingErr / seen) * 100).toFixed(1)}% · held-out ridge acc ${(acc * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)} min)`
  );

  if (round === PRUNE_ROUNDS) break;
  // ROLE-AWARE pruning: excitatory and inhibitory pruned separately so the
  // stabilizing scaffolding survives in proportion (naive credit pruning
  // strips inhibition first → seizure, then collapse — measured, round 1-2)
  for (const role of [0, 1]) {
    const living = [];
    for (let i = 0; i < brain.N; i++) {
      if (eng.alive[i] && brain.inhibitory[i] === role) living.push(i);
    }
    living.sort((a, b) => credit[a] - credit[b]);
    const toPrune = Math.floor(living.length * PRUNE_FRAC);
    for (let k = 0; k < toPrune; k++) eng.alive[living[k]] = 0;
  }
}

console.log('\naccuracy-vs-size curve above = the pruning descent from a known structure');
console.log(`compare: grown-organism best 33.3% (834n) · bigram 28.8% · transformer ≈58%`);
