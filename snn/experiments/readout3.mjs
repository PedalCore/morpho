// v16a RICHER FEATURES, LINEAR SOLVE: is the information already in the
// reservoir but invisible to rate-traces? Five arms, all closed-form ridge,
// all on the 42.5% ladder-best genome @120k under the full budget — only
// the FEATURE MAP changes:
//
//   A0  control        1024 rate traces (τ20)           d ≈ 1.2k  (= 42.5% arm)
//   A1  multi-τ        τ ∈ {20, 80, 320} banks          d ≈ 3.3k
//       (v11's mechanism at the sample budget v11 lacked: 40k fit vs 16k)
//   A2  pairwise       + trace_i·trace_j on 2048 random co-tap pairs
//       (second-order correlations — what rates alone cannot see)
//   A3  ELM            + 2048 fixed random ReLU projections of the traces
//       (generic nonlinearity, extreme-learning-machine style)
//   A4  char-gated     + trace_i·[cur=c] on 2048 random (i,c) pairs
//       (bilinear context gating — traces READ differently per current char)
//
// Each arm re-simulates the identical deterministic pass (sim is ~2 min for
// this sparse phenotype; memory stays flat) and solves ridge once. Per-arm
// checkpointing. PREDICTION (pre-registered): A1/A4 are the likely winners
// if the linear ceiling is a feature-map problem; if all five tie at ~42%,
// the reservoir's readable information is exhausted and the lever is FF
// depth (v16c) or substrate plasticity (v16d).
//
// Run: npm run experiment:readout3 [seed]

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';
import { developBrain } from './evolve.mjs';

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const TAPS = Number(process.env.TAPS ?? 1024);
const CALIB_CHARS = 5000;
const FIT_CHARS = Number(process.env.FIT_CHARS ?? 40000);
const TEST_CHARS = 3000;
const STEPS_PER_CHAR = 10;
const REFRAC = 4;
const MAX_DELAY = 16;
const TARGET_RATE = 0.002;
const TAUS = [20, 80, 320];
const NPAIR = 2048;
const NELM = 2048;
const NGATE = 2048;

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
console.log(
  `readout3 · ladder-best genome @${N}n (${(brain.M / 1e6).toFixed(2)}M syn) · ` +
  `taps ${TAPS} × τ[${TAUS}] · fit ${FIT_CHARS} · seed ${SEED}`
);
const t0 = Date.now();

// fixed random structures shared by all arms (seeded, arm-independent)
const featRng = mulberry32(SEED ^ 0xf00d);
const tapSlot = new Int32Array(N).fill(-1);
let taps = 0;
for (let f = 0; f < TAPS; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  const id = l * brain.layerSize + Math.floor(featRng() * brain.layerSize);
  if (tapSlot[id] < 0) tapSlot[id] = taps++;
}
const pairRng = mulberry32(SEED ^ 0x9a17);
const pairA = new Int32Array(NPAIR), pairB = new Int32Array(NPAIR);
for (let k = 0; k < NPAIR; k++) {
  pairA[k] = Math.floor(pairRng() * taps);
  pairB[k] = Math.floor(pairRng() * taps);
}
const elmRng = mulberry32(SEED ^ 0xe1a);
// sparse random projection: each ELM unit sums 16 signed taps, ReLU
const ELM_K = 16;
const elmIdx = new Int32Array(NELM * ELM_K), elmSgn = new Float32Array(NELM * ELM_K);
for (let k = 0; k < NELM * ELM_K; k++) {
  elmIdx[k] = Math.floor(elmRng() * taps);
  elmSgn[k] = elmRng() < 0.5 ? -1 : 1;
}
const gateRng = mulberry32(SEED ^ 0x6a7e);
const gateTap = new Int32Array(NGATE), gateChar = new Int32Array(NGATE);
for (let k = 0; k < NGATE; k++) {
  gateTap[k] = Math.floor(gateRng() * taps);
  gateChar[k] = Math.floor(gateRng() * V);
}

// ---------- one deterministic simulation pass, arm-specific rows ----------
function simulate(buildRow, d) {
  const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
  const thrLayer = new Float32Array(brain.L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let dd = 0; dd < 256; dd++) decayPow[dd] = Math.exp(-dd / TAUS[0]);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const layerSpikes = new Float64Array(brain.L);
  let t = 0;
  const banks = TAUS.map(() => new Float64Array(taps));
  const decays = TAUS.map((tau) => Math.exp(-1 / tau));
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
        if (f >= 0) for (const bank of banks) bank[f] += 1;
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
      for (let bnk = 0; bnk < banks.length; bnk++) {
        const bank = banks[bnk], dec = decays[bnk];
        for (let i = 0; i < taps; i++) bank[i] *= dec;
      }
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
    X.push(buildRow(banks, cur, ids[pos - 1] ?? 0, ids[pos - 2] ?? 0, d));
    y.push(ids[pos + 1]);
    pos++;
  }
  return { X, y };
}

// row builders: context block (3 one-hots + bias) appended uniformly
function ctx(r, off, cur, p1, p2) {
  r[off + cur] = 1;
  r[off + V + p1] = 1;
  r[off + 2 * V + p2] = 1;
  r[r.length - 1] = 1;
}
const ARMS = {
  'A0-rates': {
    d: taps + 3 * V + 1,
    row(banks, cur, p1, p2, d) {
      const r = new Float32Array(d);
      for (let i = 0; i < taps; i++) r[i] = banks[0][i];
      ctx(r, taps, cur, p1, p2);
      return r;
    },
  },
  'A1-multitau': {
    d: 3 * taps + 3 * V + 1,
    row(banks, cur, p1, p2, d) {
      const r = new Float32Array(d);
      for (let b = 0; b < 3; b++)
        for (let i = 0; i < taps; i++) r[b * taps + i] = banks[b][i];
      ctx(r, 3 * taps, cur, p1, p2);
      return r;
    },
  },
  'A2-pairs': {
    d: taps + NPAIR + 3 * V + 1,
    row(banks, cur, p1, p2, d) {
      const r = new Float32Array(d);
      const f = banks[0];
      for (let i = 0; i < taps; i++) r[i] = f[i];
      for (let k = 0; k < NPAIR; k++) r[taps + k] = f[pairA[k]] * f[pairB[k]];
      ctx(r, taps + NPAIR, cur, p1, p2);
      return r;
    },
  },
  'A3-elm': {
    d: taps + NELM + 3 * V + 1,
    row(banks, cur, p1, p2, d) {
      const r = new Float32Array(d);
      const f = banks[0];
      for (let i = 0; i < taps; i++) r[i] = f[i];
      for (let k = 0; k < NELM; k++) {
        let s = 0;
        for (let j = 0; j < ELM_K; j++) {
          const idx = k * ELM_K + j;
          s += elmSgn[idx] * f[elmIdx[idx]];
        }
        r[taps + k] = s > 0 ? s : 0; // ReLU
      }
      ctx(r, taps + NELM, cur, p1, p2);
      return r;
    },
  },
  'A4-chargate': {
    d: taps + NGATE + 3 * V + 1,
    row(banks, cur, p1, p2, d) {
      const r = new Float32Array(d);
      const f = banks[0];
      for (let i = 0; i < taps; i++) r[i] = f[i];
      for (let k = 0; k < NGATE; k++) {
        if (gateChar[k] === cur) r[taps + k] = f[gateTap[k]];
      }
      ctx(r, taps + NGATE, cur, p1, p2);
      return r;
    },
  },
};

// ---------- ridge / acc (dimension-generic) ----------
function ridge(X, y, d, lambda = 1.0) {
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
function accOf(W, X, y) {
  const sc = new Float64Array(V);
  let ok = 0;
  for (let s = 0; s < X.length; s++) {
    sc.fill(0);
    const x = X[s];
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Wi = W[i];
      for (let vv = 0; vv < V; vv++) sc[vv] += xi * Wi[vv];
    }
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
  }
  return ok / X.length;
}

// ---------- run arms (per-arm checkpoint) ----------
const cfgTag = (TAPS === 1024 && FIT_CHARS === 40000) ? '' : `-taps${TAPS}-fit${FIT_CHARS}`;
const outPath = new URL(`./results/readout3-seed${SEED}${cfgTag}.json`, import.meta.url);
const results = existsSync(outPath)
  ? JSON.parse(readFileSync(outPath, 'utf8'))
  : { SEED, N, TAPS, TAUS, FIT_CHARS, arms: {} };
const armFilter = process.env.ARMS ? process.env.ARMS.split(',') : null;
for (const [name, arm] of Object.entries(ARMS)) {
  if (armFilter && !armFilter.includes(name)) continue;
  if (results.arms[name]) {
    console.log(`${name.padEnd(12)} acc=${(results.arms[name].acc * 100).toFixed(1)}%  (checkpointed)`);
    continue;
  }
  const { X, y } = simulate(arm.row, arm.d);
  const W = ridge(X.slice(0, FIT_CHARS), y.slice(0, FIT_CHARS), arm.d);
  const acc = accOf(W, X.slice(FIT_CHARS), y.slice(FIT_CHARS));
  results.arms[name] = { acc, d: arm.d, trainedParams: arm.d * V };
  writeFileSync(outPath, JSON.stringify(results, null, 1));
  console.log(
    `${name.padEnd(12)} acc=${(acc * 100).toFixed(1)}%  d=${arm.d} ` +
    `trained=${((arm.d * V) / 1000).toFixed(0)}k  (${((Date.now() - t0) / 60000).toFixed(1)}m)`
  );
}
console.log(`\nsaved to experiments/results/readout3-seed${SEED}.json`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
