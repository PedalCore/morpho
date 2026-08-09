// Readout v2 on the 120k big brain: the ceiling was the readout, so upgrade
// exactly that — and test the Mamba idea while we're at it.
//
//   MORE TAPS + TEMPORAL DEPTH: feature taps carry traces at THREE
//     timescales (τ = 20/80/320 ms ≈ up to ~32 characters of context).
//     A multi-τ exponential trace bank is precisely a diagonal linear
//     state-space memory (S4-family kernels) — computed by the spikes.
//   SELECTIVE STATE (Mamba-Spike arXiv:2408.11823 / SpikingMamba
//     arXiv:2510.04595, gradient-free analog): Mamba's core mechanism is
//     input-dependent state write/retention. Here the slow trace's write
//     gain is the character's bigram SURPRISE — rare inputs write strongly
//     into memory, predictable ones barely perturb it.
//   LONGER SYMBOLIC CONTEXT: previous-char one-hot joins current-char.
//   DEEPER FF STACK: 2-layer Forward-Forward head (arXiv:2502.20411),
//     standardized features, layer-normed hidden pass, goodness summed
//     across layers, used to RERANK the ridge readout's top candidates.
//
// Arms (one brain, one collection pass, same fit/test):
//   A  1024 taps, single fast trace            (v10 baseline features)
//   B  384 taps × 3 timescales                 (state-space memory)
//   C  B with selective slow trace             (Mamba-style gating)
//   D  FF-v2 rerank of the best ridge arm
//
// Run: npm run experiment:readout2 [seed]

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { mulberry32 } from '../js/core/rng.js';

const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const CALIB_CHARS = 5000;
const FIT_CHARS = 16000;
const TEST_CHARS = 2500;
const TAPS_A = 1024;
const TAPS_MT = Number(process.argv[3] ?? 384);
const TAUS = [20, 80, 320];
const TARGET_RATE = 0.002;

async function getText() {
  if (!existsSync(DATA_PATH)) {
    mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
    const res = await fetch(DATA_URL);
    writeFileSync(DATA_PATH, await res.text());
  }
  return readFileSync(DATA_PATH, 'utf8');
}

// ---- brain + engine (as v10 bigbrain, stability recipe included) ----
function buildBrain(seed) {
  const rng = mulberry32(seed);
  const L = 4;
  const layerSize = Math.floor(N / L);
  const layer = new Uint8Array(N);
  for (let i = 0; i < N; i++) layer[i] = Math.min(L - 1, Math.floor(i / layerSize));
  const inhibitory = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (rng() < 0.15) inhibitory[i] = 1;
  const FF = 14, REC = 6, SKIP = 3, FB = 2, INH = 22;
  const pickIn = (l) => l * layerSize + Math.floor(rng() * layerSize);
  const srcs = [], tgts = [], ws = [], dls = [];
  const wE = () => 0.28 + rng() * 0.22;
  const wI = () => -(0.5 + rng() * 0.3);
  for (let i = 0; i < N; i++) {
    const l = layer[i];
    const add = (t, w, d) => { srcs.push(i); tgts.push(t); ws.push(w); dls.push(d); };
    if (inhibitory[i]) {
      for (let k = 0; k < INH; k++) add(pickIn(l), wI(), 1 + Math.floor(rng() * 3));
    } else {
      if (l + 1 < L) for (let k = 0; k < FF; k++) add(pickIn(l + 1), wE(), 1 + Math.floor(rng() * 4));
      for (let k = 0; k < REC; k++) add(pickIn(l), wE() * 0.45, 1 + Math.floor(rng() * 6));
      if (l + 2 < L) for (let k = 0; k < SKIP; k++) add(pickIn(l + 2), wE(), 2 + Math.floor(rng() * 6));
      if (l > 0) for (let k = 0; k < FB; k++) add(pickIn(l - 1), wE() * 0.3, 3 + Math.floor(rng() * 8));
    }
  }
  const M = srcs.length;
  const synStart = new Int32Array(N + 1);
  for (let s = 0; s < M; s++) synStart[srcs[s] + 1]++;
  for (let i = 0; i < N; i++) synStart[i + 1] += synStart[i];
  const synTgt = new Int32Array(M), synW = new Float32Array(M), synDelay = new Uint8Array(M);
  const cursor = synStart.slice(0, N);
  for (let s = 0; s < M; s++) {
    const p = cursor[srcs[s]]++;
    synTgt[p] = tgts[s]; synW[p] = ws[s]; synDelay[p] = dls[s];
  }
  return { N, L, layerSize, layer, inhibitory, synStart, synTgt, synW, synDelay, M };
}

const brain = buildBrain(SEED);
const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
const thrLayer = new Float32Array(brain.L).fill(1.0);
const decayPow = new Float64Array(256);
for (let d = 0; d < 256; d++) decayPow[d] = Math.exp(-d / TAU);
const ring = Array.from({ length: MAX_DELAY }, () => []);
const layerSpikes = new Float64Array(brain.L);
let t = 0;
let onSpike = null;
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
        ring[(t + Math.min(brain.synDelay[s], MAX_DELAY - 1)) % MAX_DELAY].push(brain.synTgt[s], brain.synW[s]);
      }
      if (onSpike) onSpike(i);
    }
  }
  bucket.length = 0;
  t++;
}

const text = await getText();
const chars = [...new Set(text)].sort();
const V = chars.length;
const ids = new Int32Array(text.length);
for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);
console.log(`readout v2 · brain ${N}n/${(brain.M / 1e6).toFixed(1)}M syn · vocab ${V} · seed ${SEED}`);
const t0 = Date.now();

const inRng = mulberry32(SEED ^ 0xabc);
const inputFan = 250;
const inputTgt = new Int32Array(V * inputFan);
for (let c = 0; c < V; c++) for (let k = 0; k < inputFan; k++) inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

// bigram surprise table (corpus statistics — the selective gate's input signal)
const big = Array.from({ length: V }, () => new Float64Array(V).fill(1));
for (let p = 0; p < 60000; p++) big[ids[p]][ids[p + 1]]++;
const surprise = Array.from({ length: V }, (_, a) => {
  const sum = big[a].reduce((x, y) => x + y, 0);
  return new Float64Array([...big[a]].map((c) => {
    const s = -Math.log2(c / sum) / Math.log2(V);
    return Math.max(0.15, Math.min(1, s));
  }));
});

// taps: TAPS_A neurons (deep-weighted); the first TAPS_MT of them carry the multi-τ bank
const featRng = mulberry32(SEED ^ 0xf00d);
const tapSlot = new Int32Array(N).fill(-1);
for (let f = 0; f < TAPS_A; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  tapSlot[l * brain.layerSize + Math.floor(featRng() * brain.layerSize)] = f;
}
const fast = new Float64Array(TAPS_A);
const mid = new Float64Array(TAPS_MT);
const slow = new Float64Array(TAPS_MT);
const sel = new Float64Array(TAPS_MT);
const dec = TAUS.map((tau) => Math.exp(-1 / tau));
let selGain = 1;
onSpike = (i) => {
  const f = tapSlot[i];
  if (f < 0) return;
  fast[f] += 1;
  if (f < TAPS_MT) {
    mid[f] += 1;
    slow[f] += 1;
    sel[f] += selGain; // selective write: surprise-gated state update
  }
};
let prevChar = 0;
function runChar(c) {
  selGain = surprise[prevChar][c];
  const base = c * inputFan;
  for (let k = 0; k < inputFan; k++) ring[(t + 1) % MAX_DELAY].push(inputTgt[base + k], 1.2);
  for (let s = 0; s < STEPS_PER_CHAR; s++) {
    step();
    for (let i = 0; i < TAPS_A; i++) fast[i] *= dec[0];
    for (let i = 0; i < TAPS_MT; i++) { mid[i] *= dec[1]; slow[i] *= dec[2]; sel[i] *= dec[2]; }
  }
  prevChar = c;
}

// homeostat + calibration
const aliveL = new Float64Array(brain.L);
aliveL.fill(0);
for (let i = 0; i < N; i++) aliveL[brain.layer[i]]++;
function homeostat(windowChars, exp_) {
  for (let l = 0; l < brain.L; l++) {
    const rate = layerSpikes[l] / (aliveL[l] * windowChars * STEPS_PER_CHAR);
    const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
    thrLayer[l] = Math.max(0.5, Math.min(25, thrLayer[l] * Math.pow(factor, exp_)));
    layerSpikes[l] = 0;
  }
}
let pos = 0;
for (let c = 0; c < CALIB_CHARS; c++) {
  runChar(ids[pos++]);
  if ((c + 1) % 100 === 0) homeostat(100, 0.6);
}
console.log(`calibrated (${((Date.now() - t0) / 1000) | 0}s) thr [${[...thrLayer].map((x) => x.toFixed(2)).join(', ')}]`);

// one collection pass, all arms' rows built together
const dA = TAPS_A + 2 * V + 1;
const dMT = TAPS_MT * 3 + 2 * V + 1;
function makeRow(kind, cur, prv) {
  if (kind === 'A') {
    const r = new Float32Array(dA);
    r.set(fast, 0);
    r[TAPS_A + cur] = 1; r[TAPS_A + V + prv] = 1; r[dA - 1] = 1;
    return r;
  }
  const r = new Float32Array(dMT);
  for (let i = 0; i < TAPS_MT; i++) {
    r[i] = fast[i];
    r[TAPS_MT + i] = mid[i];
    r[2 * TAPS_MT + i] = kind === 'C' ? sel[i] : slow[i];
  }
  r[3 * TAPS_MT + cur] = 1; r[3 * TAPS_MT + V + prv] = 1; r[dMT - 1] = 1;
  return r;
}
function collect(n) {
  const out = { A: [], B: [], C: [], y: [] };
  for (let k = 0; k < n; k++) {
    const cur = ids[pos];
    const prv = pos > 0 ? ids[pos - 1] : 0;
    runChar(cur);
    out.A.push(makeRow('A', cur, prv));
    out.B.push(makeRow('B', cur, prv));
    out.C.push(makeRow('C', cur, prv));
    out.y.push(ids[pos + 1]);
    pos++;
  }
  return out;
}
const fit = collect(FIT_CHARS);
const test = collect(TEST_CHARS);
console.log(`collected (${((Date.now() - t0) / 60000).toFixed(1)} min)`);

function ridge(X, y, lambda = 1.0) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(V));
  for (let s = 0; s < X.length; s++) {
    const x = X[s];
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      if (!xi) continue;
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
      for (let vv = 0; vv < V; vv++) B[r][vv] -= f * B[col][vv];
    }
  }
  return Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(V);
    for (let vv = 0; vv < V; vv++) row[vv] = B[i][vv] / (A[i][i] || 1e-12);
    return row;
  });
}
function scoresOf(W, x, out) {
  out.fill(0);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (!xi) continue;
    const Wi = W[i];
    for (let vv = 0; vv < V; vv++) out[vv] += xi * Wi[vv];
  }
}
function accOf(W, X, y) {
  const sc = new Float64Array(V);
  let ok = 0;
  for (let s = 0; s < X.length; s++) {
    scoresOf(W, X[s], sc);
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
  }
  return ok / X.length;
}

const results = {};
for (const arm of ['A', 'B', 'C']) {
  const W = ridge(fit[arm], fit.y);
  results[arm] = { W, acc: accOf(W, test[arm], test.y) };
  const label = { A: 'A  1024 taps, fast trace', B: 'B  384×3τ state-space bank', C: 'C  B + selective (surprise-gated) slow state' }[arm];
  console.log(`${label}: ${(results[arm].acc * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)} min)`);
}

// ---- FF v2: standardized features, 2 layers, goodness-summed rerank ----
const bestArm = results.C.acc >= results.B.acc ? 'C' : 'B';
const Xf = fit[bestArm];
const Xt = test[bestArm];
const dF = Xf[0].length;
const mean = new Float64Array(dF);
const std = new Float64Array(dF).fill(1e-6);
for (const x of Xf) for (let i = 0; i < dF; i++) mean[i] += x[i];
for (let i = 0; i < dF; i++) mean[i] /= Xf.length;
for (const x of Xf) for (let i = 0; i < dF; i++) std[i] += (x[i] - mean[i]) ** 2;
for (let i = 0; i < dF; i++) std[i] = Math.sqrt(std[i] / Xf.length) + 1e-6;

const H1 = 192, H2 = 96;
const ffRng = mulberry32(SEED ^ 0xff);
const W1 = Array.from({ length: H1 }, () => Float64Array.from({ length: dF + V }, () => (ffRng() - 0.5) * 0.05));
const W2 = Array.from({ length: H2 }, () => Float64Array.from({ length: H1 }, () => (ffRng() - 0.5) * 0.1));
const xn = new Float64Array(dF);
const h1 = new Float64Array(H1);
const h1n = new Float64Array(H1);
const h2 = new Float64Array(H2);
function ffForward(x, label) {
  for (let i = 0; i < dF; i++) xn[i] = (x[i] - mean[i]) / std[i];
  let g1 = 0;
  for (let u = 0; u < H1; u++) {
    let a = 0;
    const Wu = W1[u];
    for (let i = 0; i < dF; i++) a += Wu[i] * xn[i];
    a += Wu[dF + label] * 3;
    h1[u] = a > 0 ? a : 0;
    g1 += h1[u] * h1[u];
  }
  const norm = Math.sqrt(g1) + 1e-6;
  for (let u = 0; u < H1; u++) h1n[u] = h1[u] / norm;
  let g2 = 0;
  for (let u = 0; u < H2; u++) {
    let a = 0;
    const Wu = W2[u];
    for (let i = 0; i < H1; i++) a += Wu[i] * h1n[i];
    h2[u] = a > 0 ? a : 0;
    g2 += h2[u] * h2[u];
  }
  return { g1, g2 };
}
function ffUpdate(x, label, sign, lr) {
  const { g1, g2 } = ffForward(x, label);
  const p1 = 1 / (1 + Math.exp(-(g1 - H1 * 0.5)));
  const p2 = 1 / (1 + Math.exp(-(g2 - H2 * 0.5)));
  const d1 = (sign > 0 ? p1 - 1 : p1) * lr;
  const d2 = (sign > 0 ? p2 - 1 : p2) * lr;
  for (let u = 0; u < H1; u++) {
    if (h1[u] <= 0) continue;
    const f = d1 * 2 * h1[u];
    const Wu = W1[u];
    for (let i = 0; i < dF; i++) if (xn[i]) Wu[i] -= f * xn[i];
    Wu[dF + label] -= f * 3;
  }
  for (let u = 0; u < H2; u++) {
    if (h2[u] <= 0) continue;
    const f = d2 * 2 * h2[u];
    const Wu = W2[u];
    for (let i = 0; i < H1; i++) Wu[i] -= f * h1n[i];
  }
}
const SKIP_FF = process.argv[4] === 'noff';
if (!SKIP_FF) {
console.log('training FF stack…');
const FF_SAMPLES = 10000;
for (let e = 0; e < 2; e++) {
  for (let s = 0; s < FF_SAMPLES; s++) {
    ffUpdate(Xf[s], fit.y[s], +1, 0.02);
    let wrong = Math.floor(ffRng() * V);
    if (wrong === fit.y[s]) wrong = (wrong + 1) % V;
    ffUpdate(Xf[s], wrong, -1, 0.02);
  }
}
// rerank ridge top-8 by summed goodness
const sc = new Float64Array(V);
let ffOk = 0;
const TOPK = 8;
for (let s = 0; s < Xt.length; s++) {
  scoresOf(results[bestArm].W, Xt[s], sc);
  const idx = [...sc.keys()].sort((a, b) => sc[b] - sc[a]).slice(0, TOPK);
  let best = idx[0];
  let bg = -Infinity;
  for (const cand of idx) {
    const { g1, g2 } = ffForward(Xt[s], cand);
    const g = g1 / H1 + g2 / H2;
    if (g > bg) { bg = g; best = cand; }
  }
  if (best === test.y[s]) ffOk++;
}
console.log(`D  FF-v2 rerank (2-layer, standardized, top-${TOPK} of arm ${bestArm}): ${((ffOk / Xt.length) * 100).toFixed(1)}%`);
}
console.log(`\ntotal ${((Date.now() - t0) / 60000).toFixed(1)} min · baselines: v10 single-τ 33.0% · bigram 28.8% · transformer ≈58%`);
