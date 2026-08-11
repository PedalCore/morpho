// v18a MATCHED FEATURE-SOURCE ABLATION: what does the spiking reservoir
// actually contribute? Under the frozen A0 budget (ladder-best genome
// @120k, 1024 taps, 40k fit, ctx one-hots present in every arm), across
// 5 build seeds:
//
//   ctx-only    context one-hots + bias, no reservoir        (floor)
//   real        ctx + correctly aligned spike traces
//   shuffled    ctx + trace vectors permuted across samples  (alignment killed)
//   timeshift   ctx + per-tap circular shifts                (cross-tap timing killed)
//   delayline   ctx + neuron-free random temporal bank over raw chars
//               (feature_j = leaky trace of [char(t−lag_j)=c_j], random lag/decay)
//   rate-esn    ctx + NON-spiking tanh reservoir with the IDENTICAL
//               topology and weights (same CSR), same tap neurons
//
// The decisive comparisons are real vs delayline and real vs rate-esn —
// "real ≫ shuffled" alone only proves traces carry within-sample char
// information. Per-(seed,arm) checkpointing; report mean ± sd.
//
// Run: npm run experiment:ablate

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';
import { developBrain } from './evolve.mjs';

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
const BUILD_SEEDS = [42, 7, 99, 1234, 2024];

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

const NROWS = FIT_CHARS + TEST_CHARS;
const dCtx = 3 * V + 1;
const d = TAPS + dCtx;

// ---------- shared: ridge + acc ----------
function ridgeAcc(X, y) {
  const dd = X[0].length;
  const A = Array.from({ length: dd }, () => new Float64Array(dd));
  const B = Array.from({ length: dd }, () => new Float64Array(V));
  for (let s = 0; s < FIT_CHARS; s++) {
    const x = X[s];
    for (let i = 0; i < dd; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Ai = A[i];
      for (let j = i; j < dd; j++) Ai[j] += xi * x[j];
      B[i][y[s]] += xi;
    }
  }
  for (let i = 0; i < dd; i++) {
    A[i][i] += 1.0;
    for (let j = 0; j < i; j++) A[i][j] = A[j][i];
  }
  for (let col = 0; col < dd; col++) {
    let piv = col;
    for (let r = col + 1; r < dd; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [B[col], B[piv]] = [B[piv], B[col]];
    const diag = A[col][col] || 1e-12;
    for (let r = 0; r < dd; r++) {
      if (r === col) continue;
      const f = A[r][col] / diag;
      if (!f) continue;
      for (let j = col; j < dd; j++) A[r][j] -= f * A[col][j];
      for (let vv = 0; vv < V; vv++) B[r][vv] -= f * B[col][vv];
    }
  }
  const W = Array.from({ length: dd }, (_, i) => {
    const row = new Float64Array(V);
    for (let vv = 0; vv < V; vv++) row[vv] = B[i][vv] / (A[i][i] || 1e-12);
    return row;
  });
  const sc = new Float64Array(V);
  let ok = 0;
  for (let s = FIT_CHARS; s < NROWS; s++) {
    sc.fill(0);
    const x = X[s];
    for (let i = 0; i < dd; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Wi = W[i];
      for (let vv = 0; vv < V; vv++) sc[vv] += xi * Wi[vv];
    }
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
  }
  return ok / TEST_CHARS;
}
function rowsFromTraces(traces, startChar) {
  // traces: Float32Array(NROWS × TAPS) or null (ctx-only)
  const X = [], y = [];
  for (let k = 0; k < NROWS; k++) {
    const pos = startChar + k;
    const nt = traces ? TAPS : 0;
    const r = new Float32Array(nt + dCtx);
    if (traces) r.set(traces.subarray(k * TAPS, (k + 1) * TAPS), 0);
    r[nt + ids[pos]] = 1;
    r[nt + V + ids[pos - 1]] = 1;
    r[nt + 2 * V + ids[pos - 2]] = 1;
    r[nt + dCtx - 1] = 1;
    X.push(r); y.push(ids[pos + 1]);
  }
  return { X, y };
}

// ---------- spiking pass → raw trace matrix ----------
function spikingTraces(seed) {
  const brain = developBrain(genome, N, seed ^ (N * 2654435761));
  const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
  const thrLayer = new Float32Array(brain.L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let dd = 0; dd < 256; dd++) decayPow[dd] = Math.exp(-dd / TAU);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const layerSpikes = new Float64Array(brain.L);
  let t = 0;
  const featRng = mulberry32(seed ^ 0xf00d);
  const tapSlot = new Int32Array(N).fill(-1);
  let taps = 0;
  for (let f = 0; f < TAPS; f++) {
    const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
    const id = l * brain.layerSize + Math.floor(featRng() * brain.layerSize);
    if (tapSlot[id] < 0) tapSlot[id] = taps++;
  }
  const fast = new Float64Array(TAPS);
  const decFast = Math.exp(-1 / TAU);
  const inRng = mulberry32(seed ^ 0xabc);
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
      for (let i = 0; i < TAPS; i++) fast[i] *= decFast;
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
  const traces = new Float32Array(NROWS * TAPS);
  for (let k = 0; k < NROWS; k++) {
    runChar(ids[pos]);
    for (let i = 0; i < TAPS; i++) traces[k * TAPS + i] = fast[i];
    pos++;
  }
  return { traces, brain, tapSlot };
}

// ---------- rate-ESN pass: identical CSR, tanh leaky dynamics ----------
function esnTraces(seed, brain, tapSlot) {
  const x = new Float32Array(N), acc = new Float32Array(N);
  const LEAK = 0.5, ESN_STEPS = 2, GAIN = 0.4;
  const inRng = mulberry32(seed ^ 0xabc);
  const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
  const inputTgt = new Int32Array(V * inputFan);
  for (let c = 0; c < V; c++)
    for (let k = 0; k < inputFan; k++)
      inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);
  const traces = new Float32Array(NROWS * TAPS);
  let pos = CALIB_CHARS - 500; // brief washout before collection
  for (let k = -500; k < NROWS; k++) {
    const c = ids[pos];
    for (let s = 0; s < ESN_STEPS; s++) {
      acc.fill(0);
      for (let i = 0; i < N; i++) {
        const xi = x[i];
        if (!xi) continue;
        for (let sy = brain.synStart[i]; sy < brain.synStart[i + 1]; sy++) {
          acc[brain.synTgt[sy]] += brain.synW[sy] * xi;
        }
      }
      const base = c * inputFan;
      for (let f = 0; f < inputFan; f++) acc[inputTgt[base + f]] += 1.2;
      for (let i = 0; i < N; i++) x[i] = (1 - LEAK) * x[i] + LEAK * Math.tanh(GAIN * acc[i]);
    }
    if (k >= 0) {
      for (let i = 0; i < N; i++) {
        const f = tapSlot[i];
        if (f >= 0) traces[k * TAPS + f] = x[i] * 5; // scale to trace-like range
      }
    }
    pos++;
  }
  return traces;
}

// ---------- neuron-free delay-line bank over raw chars ----------
function delaylineTraces(seed) {
  const rng = mulberry32(seed ^ 0xd1a);
  const fc = new Int32Array(TAPS), flag = new Int32Array(TAPS), fdec = new Float32Array(TAPS);
  for (let j = 0; j < TAPS; j++) {
    fc[j] = Math.floor(rng() * V);
    flag[j] = 1 + Math.floor(rng() * 8);
    fdec[j] = Math.exp(-1 / (5 + rng() * 75));
  }
  const traces = new Float32Array(NROWS * TAPS);
  const state = new Float32Array(TAPS);
  let pos = CALIB_CHARS - 200;
  for (let k = -200; k < NROWS; k++) {
    for (let j = 0; j < TAPS; j++) {
      const src = pos - flag[j];
      state[j] = state[j] * fdec[j] + (src >= 0 && ids[src] === fc[j] ? 1 : 0);
    }
    if (k >= 0) traces.set(state, k * TAPS);
    pos++;
  }
  return traces;
}

// ---------- destroyers ----------
function shuffleTraces(traces, seed) {
  const out = new Float32Array(traces.length);
  const perm = (n, off, rng) => {
    const p = new Int32Array(n);
    for (let i = 0; i < n; i++) p[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = p[i]; p[i] = p[j]; p[j] = tmp;
    }
    for (let i = 0; i < n; i++)
      out.set(traces.subarray((off + p[i]) * TAPS, (off + p[i] + 1) * TAPS), (off + i) * TAPS);
  };
  const rng = mulberry32(seed ^ 0x5f5f);
  perm(FIT_CHARS, 0, rng);       // permute within fit
  perm(TEST_CHARS, FIT_CHARS, rng); // and within test — no cross-boundary leak
  return out;
}
function timeshiftTraces(traces, seed) {
  const out = new Float32Array(traces.length);
  const rng = mulberry32(seed ^ 0x7571);
  for (let i = 0; i < TAPS; i++) {
    const sf = 1 + Math.floor(rng() * (FIT_CHARS - 2));
    const st = 1 + Math.floor(rng() * (TEST_CHARS - 2));
    for (let k = 0; k < FIT_CHARS; k++)
      out[((k + sf) % FIT_CHARS) * TAPS + i] = traces[k * TAPS + i];
    for (let k = 0; k < TEST_CHARS; k++)
      out[(FIT_CHARS + ((k + st) % TEST_CHARS)) * TAPS + i] = traces[(FIT_CHARS + k) * TAPS + i];
  }
  return out;
}

// ---------- run ----------
const outPath = new URL('./results/ablate.json', import.meta.url);
const results = existsSync(outPath) ? JSON.parse(readFileSync(outPath, 'utf8'))
  : { N, TAPS, FIT_CHARS, BUILD_SEEDS, arms: {} };
const t0 = Date.now();
const startChar = CALIB_CHARS;
const save = () => writeFileSync(outPath, JSON.stringify(results, null, 1));
const record = (arm, seed, acc) => {
  (results.arms[arm] ??= {})[seed] = acc;
  save();
  console.log(`  ${arm.padEnd(10)} seed ${String(seed).padEnd(5)} acc=${(acc * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
};

if (results.arms['ctx-only']?.all === undefined) {
  const { X, y } = rowsFromTraces(null, startChar);
  results.arms['ctx-only'] = { all: ridgeAcc(X, y) };
  save();
  console.log(`  ctx-only   (seed-free)  acc=${(results.arms['ctx-only'].all * 100).toFixed(1)}%`);
}

for (const seed of BUILD_SEEDS) {
  const need = (arm) => !(results.arms[arm]?.[seed] !== undefined);
  if (!['real', 'shuffled', 'timeshift', 'delayline', 'rate-esn'].some(need)) continue;
  console.log(`— build seed ${seed} —`);
  let spk = null;
  if (need('real') || need('shuffled') || need('timeshift') || need('rate-esn')) {
    spk = spikingTraces(seed);
  }
  if (need('real')) {
    const { X, y } = rowsFromTraces(spk.traces, startChar);
    record('real', seed, ridgeAcc(X, y));
  }
  if (need('shuffled')) {
    const { X, y } = rowsFromTraces(shuffleTraces(spk.traces, seed), startChar);
    record('shuffled', seed, ridgeAcc(X, y));
  }
  if (need('timeshift')) {
    const { X, y } = rowsFromTraces(timeshiftTraces(spk.traces, seed), startChar);
    record('timeshift', seed, ridgeAcc(X, y));
  }
  if (need('delayline')) {
    const { X, y } = rowsFromTraces(delaylineTraces(seed), startChar);
    record('delayline', seed, ridgeAcc(X, y));
  }
  if (need('rate-esn')) {
    const { X, y } = rowsFromTraces(esnTraces(seed, spk.brain, spk.tapSlot), startChar);
    record('rate-esn', seed, ridgeAcc(X, y));
  }
}

// summary: mean ± sd
console.log('\n— summary (mean ± sd over build seeds) —');
for (const [arm, bySeed] of Object.entries(results.arms)) {
  if (bySeed.all !== undefined) {
    console.log(`${arm.padEnd(10)} ${(bySeed.all * 100).toFixed(1)}%`);
    continue;
  }
  const vals = BUILD_SEEDS.map((s) => bySeed[s]).filter((x) => x !== undefined);
  const m = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length);
  console.log(`${arm.padEnd(10)} ${(m * 100).toFixed(1)} ± ${(sd * 100).toFixed(1)}%  (n=${vals.length})`);
}
console.log(`\nsaved to ${outPath.pathname}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
