// v16c FORWARD-FORWARD, PROPERLY: the three prior FF attempts (12.1%,
// 10.1%) were shallow, unnormalized, casually-trained heads. This is the
// serious attempt, with the ingredients the FF literature says are
// load-bearing:
//
//   STANDARDIZATION  per-dim z-score of all input features (fit-set stats)
//   DEPTH + LAYERNORM  3 hidden layers; each layer's output vector is
//     L2-normalized before feeding the next (Hinton's recipe: magnitude
//     carries goodness, direction carries information — normalization
//     forces each layer to find NEW evidence)
//   HARD NEGATIVES  negative samples embed a WRONG label; half are chosen
//     as the ridge control's most-confusable wrong class, half uniform
//
// Training is layer-local (goodness = mean squared activation vs threshold;
// gradient never crosses a layer boundary — no backprop). Label is embedded
// by appending its one-hot to the input; inference scores all 65 labels by
// total goodness of layers 2+. Arms: ff-depth3, ff-depth1 (depth ablation),
// ridge on identical standardized features (control).
//
// Substrate: ladder-best genome @120k, A0 features (1024 taps + 3 char
// one-hots + bias), 40k fit / 3k test — directly comparable to the 43.2%
// ridge figure. Per-epoch checkpointing (weights .bin + metrics JSON).
//
// Run: npm run experiment:ff2 [seed]

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';
import { developBrain } from './evolve.mjs';

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const TAPS = 1024;
const CALIB_CHARS = 5000;
const FIT_CHARS = 40000;
const TEST_CHARS = 3000;
const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const TARGET_RATE = 0.002;
const HIDDEN = 512;
const EPOCHS = 8;
const THETA = 2.0;
const BATCH = 32;

async function main() {
const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;
if (!existsSync(DATA_PATH)) {
  mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
  const res = await fetch(DATA_URL);
  writeFileSync(DATA_PATH, await res.text());
}
const text = readFileSync(DATA_PATH, 'utf8');
const chars = [...new Set(text)].sort();
const V = chars.length;
const ids = new Int32Array(text.length);
for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);

const ck = JSON.parse(readFileSync(
  new URL('./results/evolve-seed42.json', import.meta.url), 'utf8'));
const genome = ck.history[0].bestGenome;
const brain = developBrain(genome, N, SEED ^ (N * 2654435761));
console.log(`ff2 · ladder-best genome @${N}n · taps ${TAPS} · hidden ${HIDDEN} · θ=${THETA} · seed ${SEED}`);
const t0 = Date.now();

// ---------- collect A0 features (identical protocol to readout3) ----------
const featRng = mulberry32(SEED ^ 0xf00d);
const tapSlot = new Int32Array(N).fill(-1);
let taps = 0;
for (let f = 0; f < TAPS; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  const id = l * brain.layerSize + Math.floor(featRng() * brain.layerSize);
  if (tapSlot[id] < 0) tapSlot[id] = taps++;
}
const dFeat = taps + 3 * V + 1;

const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
const thrLayer = new Float32Array(brain.L).fill(1.0);
const decayPow = new Float64Array(256);
for (let dd = 0; dd < 256; dd++) decayPow[dd] = Math.exp(-dd / TAU);
const ring = Array.from({ length: MAX_DELAY }, () => []);
const layerSpikes = new Float64Array(brain.L);
let t = 0;
const fast = new Float64Array(taps);
const decFast = Math.exp(-1 / TAU);
const inRng = mulberry32(SEED ^ 0xabc);
const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
const inputTgt = new Int32Array(V * inputFan);
for (let c = 0; c < V; c++)
  for (let k = 0; k < inputFan; k++)
    inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);
function step() {
  const bucket = ring[t % MAX_DELAY];
  for (let b = 0; b < bucket.length; b += 2) {
    const i = bucket[b];
    if (t < refracUntil[i]) continue;
    const dt = t - lastT[i];
    if (dt > 0) { v[i] *= decayPow[dt > 255 ? 255 : dt]; lastT[i] = t; }
    v[i] += bucket[b + 1];
    const l = brain.layer[i];
    const cap = thrLayer[l] * 3;
    if (v[i] > cap) v[i] = cap;
    if (v[i] >= thrLayer[l]) {
      v[i] = 0; refracUntil[i] = t + REFRAC; layerSpikes[l]++;
      for (let s = brain.synStart[i]; s < brain.synStart[i + 1]; s++) {
        ring[(t + Math.min(brain.synDelay[s], MAX_DELAY - 1)) % MAX_DELAY]
          .push(brain.synTgt[s], brain.synW[s]);
      }
      const f = tapSlot[i];
      if (f >= 0) fast[f] += 1;
    }
  }
  bucket.length = 0;
  t++;
}
function runChar(c) {
  const base = c * inputFan;
  for (let k = 0; k < inputFan; k++) ring[(t + 1) % MAX_DELAY].push(inputTgt[base + k], 1.2);
  for (let s = 0; s < STEPS_PER_CHAR; s++) {
    step();
    for (let i = 0; i < taps; i++) fast[i] *= decFast;
  }
}
const aliveL = new Float64Array(brain.L);
for (let i = 0; i < N; i++) aliveL[brain.layer[i]]++;
let pos = 0;
for (let c = 0; c < CALIB_CHARS; c++) {
  runChar(ids[pos++]);
  if ((c + 1) % 100 === 0) {
    for (let l = 0; l < brain.L; l++) {
      const rate = layerSpikes[l] / (aliveL[l] * 100 * STEPS_PER_CHAR);
      const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
      thrLayer[l] = Math.max(0.5, Math.min(25, thrLayer[l] * Math.pow(factor, 0.6)));
      layerSpikes[l] = 0;
    }
  }
}
const X = [], y = [];
for (let k = 0; k < FIT_CHARS + TEST_CHARS; k++) {
  const cur = ids[pos];
  runChar(cur);
  const r = new Float32Array(dFeat);
  r.set(fast.subarray(0, taps), 0);
  r[taps + cur] = 1;
  r[taps + V + (pos > 0 ? ids[pos - 1] : 0)] = 1;
  r[taps + 2 * V + (pos > 1 ? ids[pos - 2] : 0)] = 1;
  r[dFeat - 1] = 1;
  X.push(r); y.push(ids[pos + 1]);
  pos++;
}
console.log(`features collected (${((Date.now() - t0) / 60000).toFixed(1)}m)`);

// ---------- standardize (fit-set stats) ----------
const mu = new Float64Array(dFeat), sd = new Float64Array(dFeat);
for (let s = 0; s < FIT_CHARS; s++) for (let i = 0; i < dFeat; i++) mu[i] += X[s][i];
for (let i = 0; i < dFeat; i++) mu[i] /= FIT_CHARS;
for (let s = 0; s < FIT_CHARS; s++) {
  const x = X[s];
  for (let i = 0; i < dFeat; i++) { const dd = x[i] - mu[i]; sd[i] += dd * dd; }
}
for (let i = 0; i < dFeat; i++) sd[i] = Math.max(Math.sqrt(sd[i] / FIT_CHARS), 1e-3);
for (let s = 0; s < X.length; s++) {
  const x = X[s];
  for (let i = 0; i < dFeat; i++) x[i] = (x[i] - mu[i]) / sd[i];
}

// ---------- ridge control on standardized features ----------
function ridge(Xs, ys, d, lambda = 1.0) {
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(V));
  for (let s = 0; s < Xs.length; s++) {
    const x = Xs[s];
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Ai = A[i];
      for (let j = i; j < d; j++) Ai[j] += xi * x[j];
      B[i][ys[s]] += xi;
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
const Wr = ridge(X.slice(0, FIT_CHARS), y.slice(0, FIT_CHARS), dFeat);
const ridgeScores = (x) => {
  const sc = new Float64Array(V);
  for (let i = 0; i < dFeat; i++) {
    const xi = x[i];
    if (!xi) continue;
    const Wi = Wr[i];
    for (let vv = 0; vv < V; vv++) sc[vv] += xi * Wi[vv];
  }
  return sc;
};
let rOk = 0;
const hardNeg = new Int32Array(FIT_CHARS);
for (let s = 0; s < X.length; s++) {
  const sc = ridgeScores(X[s]);
  let b = 0;
  for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
  if (s >= FIT_CHARS) { if (b === y[s]) rOk++; }
  else {
    // most-confusable WRONG class for this sample
    let hb = -1;
    for (let vv = 0; vv < V; vv++) if (vv !== y[s] && (hb < 0 || sc[vv] > sc[hb])) hb = vv;
    hardNeg[s] = hb;
  }
}
const ridgeAcc = rOk / TEST_CHARS;
console.log(`ridge control (standardized): ${(ridgeAcc * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)}m)`);

// ---------- FF net ----------
const dIn = dFeat + V; // features ⊕ label one-hot
function makeNet(depth, rng) {
  const layers = [];
  let inDim = dIn;
  for (let l = 0; l < depth; l++) {
    const W = new Float32Array(inDim * HIDDEN);
    const scale = Math.sqrt(2 / inDim);
    for (let i = 0; i < W.length; i++) {
      const u = Math.max(rng(), 1e-9), w = rng();
      W[i] = scale * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    }
    layers.push({ W, b: new Float32Array(HIDDEN), mW: new Float32Array(inDim * HIDDEN), mB: new Float32Array(HIDDEN), inDim });
    inDim = HIDDEN;
  }
  return layers;
}
// forward one layer: h = relu(W·x + b); returns {h, g} (g = mean h²)
function fwdLayer(L, x, h) {
  let g = 0;
  for (let j = 0; j < HIDDEN; j++) {
    let s = L.b[j];
    const off = j * L.inDim;
    for (let i = 0; i < L.inDim; i++) s += L.W[off + i] * x[i];
    h[j] = s > 0 ? s : 0;
    g += h[j] * h[j];
  }
  return g / HIDDEN;
}
function l2norm(h, out) {
  let n = 0;
  for (let j = 0; j < HIDDEN; j++) n += h[j] * h[j];
  n = Math.sqrt(n) + 1e-6;
  for (let j = 0; j < HIDDEN; j++) out[j] = h[j] / n;
}
const sig = (z) => 1 / (1 + Math.exp(-z));

function trainNet(net, epochs, tag) {
  const rng = mulberry32(SEED ^ 0xff2);
  const xbuf = new Float32Array(dIn);
  const hs = net.map(() => new Float32Array(HIDDEN));
  const xin = net.map((L) => new Float32Array(L.inDim));
  const order = new Int32Array(FIT_CHARS);
  for (let i = 0; i < FIT_CHARS; i++) order[i] = i;
  const metrics = [];
  for (let ep = 0; ep < epochs; ep++) {
    const lr = 0.03 / (1 + ep / 3);
    // shuffle
    for (let i = FIT_CHARS - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    for (let bi = 0; bi < FIT_CHARS; bi++) {
      const s = order[bi];
      for (const posPass of [1, 0]) {
        const label = posPass ? y[s]
          : (rng() < 0.5 ? hardNeg[s] : (() => { let w; do { w = Math.floor(rng() * V); } while (w === y[s]); return w; })());
        xbuf.set(X[s], 0);
        xbuf.fill(0, dFeat);
        xbuf[dFeat + label] = 4; // label embedding amplitude (standardized-scale)
        let x = xbuf;
        for (let l = 0; l < net.length; l++) {
          const L = net[l];
          xin[l].set(x.subarray(0, L.inDim));
          const g = fwdLayer(L, xin[l], hs[l]);
          // local objective: pos → g > θ, neg → g < θ
          const p = sig(g - THETA);
          const dg = (posPass ? p - 1 : p) * (2 / HIDDEN); // dLoss/dg × dg/dh common factor
          for (let j = 0; j < HIDDEN; j++) {
            const hj = hs[l][j];
            if (!hj) continue;
            const delta = dg * hj * lr;
            const off = j * L.inDim;
            const xi = xin[l];
            for (let i = 0; i < L.inDim; i++) L.W[off + i] -= delta * xi[i];
            L.b[j] -= delta;
          }
          const nx = new Float32Array(HIDDEN);
          l2norm(hs[l], nx);
          x = nx;
        }
      }
    }
    // eval on a 1000-row test subset (full test at the end)
    const acc = evalNet(net, 1000);
    metrics.push({ ep, acc });
    console.log(`  [${tag}] epoch ${ep}: acc ${(acc * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
  }
  return metrics;
}
function evalNet(net, nTest) {
  const xbuf = new Float32Array(dIn);
  const h = new Float32Array(HIDDEN);
  const nx = new Float32Array(HIDDEN);
  let ok = 0;
  const stride = Math.max(1, Math.floor(TEST_CHARS / nTest));
  let count = 0;
  for (let s = FIT_CHARS; s < FIT_CHARS + TEST_CHARS; s += stride) {
    let best = -Infinity, bl = 0;
    for (let lab = 0; lab < V; lab++) {
      xbuf.set(X[s], 0);
      xbuf.fill(0, dFeat);
      xbuf[dFeat + lab] = 4;
      let x = xbuf;
      let good = 0;
      for (let l = 0; l < net.length; l++) {
        const L = net[l];
        const g = fwdLayer(L, x.subarray(0, L.inDim), h);
        if (l > 0 || net.length === 1) good += g; // skip layer-1 goodness at depth>1
        l2norm(h, nx);
        x = nx;
      }
      if (good > best) { best = good; bl = lab; }
    }
    if (bl === y[s]) ok++;
    count++;
  }
  return ok / count;
}

// ---------- arms ----------
const outPath = new URL(`./results/ff2-seed${SEED}.json`, import.meta.url);
const results = existsSync(outPath) ? JSON.parse(readFileSync(outPath, 'utf8'))
  : { SEED, N, TAPS, HIDDEN, THETA, EPOCHS, ridgeAcc, arms: {} };
results.ridgeAcc = ridgeAcc;
for (const depth of [3, 1]) {
  const tag = `ff-depth${depth}`;
  if (results.arms[tag]) { console.log(`${tag} (checkpointed): ${(results.arms[tag].finalAcc * 100).toFixed(1)}%`); continue; }
  const net = makeNet(depth, mulberry32(SEED ^ (0xd0 + depth)));
  const metrics = trainNet(net, EPOCHS, tag);
  const finalAcc = evalNet(net, TEST_CHARS);
  results.arms[tag] = { metrics, finalAcc };
  writeFileSync(outPath, JSON.stringify(results, null, 1));
  console.log(`${tag}: final ${(finalAcc * 100).toFixed(1)}%  (ridge control ${(ridgeAcc * 100).toFixed(1)}%)`);
}
console.log(`\nsaved to ${outPath.pathname}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
