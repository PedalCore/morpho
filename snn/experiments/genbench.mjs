// v15 GENERATION BENCHMARK: make generation a measurable scaling signal
// instead of a vibe, and test the memory-gap hypothesis directly.
//
//   METRIC — score generated text in bits/char under an interpolated 1-5
//     gram model trained on the real corpus. Real held-out text sets the
//     floor; uniform noise sets the ceiling; generated text lands between.
//   SCHEDULED SAMPLING, CLOSED FORM — exposure bias exists because the
//     readout only ever trains on teacher-forced features. Fix without
//     backprop: drive the brain with a mixture of true and self-sampled
//     characters (probability p), collect ridge rows under that mixed
//     distribution (targets stay the TRUE next chars), refit in closed
//     form. Arms p ∈ {0, 0.25, 0.5}.
//   PREDICTION (memory-gap hypothesis): scheduled sampling improves
//     generation-bpc somewhat but teacher-forced accuracy and gen quality
//     stay far apart — the gap is long-context memory, not exposure alone.
//
// Substrate: the 42.5% ladder-best genome (v13 seed-42 gen-0) at 120k
// under the full v12 readout budget. Everything deterministic; per-arm
// checkpointing (background tasks get reaped on this machine).
//
// Run: npm run experiment:genbench [seed]

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
const GEN_CHARS = 1500;
const PRIME_CHARS = 100;
const TEMP = 0.8; // power-law temperature for feeding AND generation
const GREEDY = 0.0; // extra generation arm: argmax decoding
const PS = [0, 0.25, 0.5];
const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const TARGET_RATE = 0.002;

// ---------- interpolated n-gram scorer (orders 1..5) ----------
// counts from the first 900k chars — disjoint from everything we generate
// against (readout windows all sit below 300k)
export function buildNgram(ids, V, upto, maxOrder = 5) {
  const maps = Array.from({ length: maxOrder }, () => new Map());
  for (let i = 0; i < upto; i++) {
    let key = 0;
    for (let o = 0; o < maxOrder && i - o >= 0; o++) {
      // key encodes the (o)-char history ending at i-1, target ids[i]
      if (o > 0) key = key * V + ids[i - o];
      const m = maps[o];
      const hk = key * V + ids[i];
      m.set(hk, (m.get(hk) ?? 0) + 1);
      if (o > 0) m.set(-key - 1, (m.get(-key - 1) ?? 0) + 1); // history count
    }
  }
  const uni = new Float64Array(V);
  let tot = 0;
  for (let i = 0; i < upto; i++) { uni[ids[i]]++; tot++; }
  const LAMBDA = [0.15, 0.2, 0.25, 0.25, 0.15]; // fixed interpolation weights
  return function prob(hist, c) {
    // hist: array of previous chars, most recent last
    let p = 0;
    let key = 0;
    for (let o = 0; o < maxOrder; o++) {
      let po;
      if (o === 0) po = (uni[c] + 1) / (tot + V);
      else {
        if (hist.length < o) { po = 1 / V; }
        else {
          key = o === 1 ? hist[hist.length - 1] : key * V + hist[hist.length - o];
          const hc = maps[o].get(-key - 1) ?? 0;
          const cc = maps[o].get(key * V + c) ?? 0;
          po = hc > 0 ? cc / hc : (uni[c] + 1) / (tot + V);
        }
      }
      p += LAMBDA[o] * po;
    }
    return p;
  };
}
export function bpcOf(seq, prob) {
  let bits = 0;
  const hist = [];
  for (const c of seq) {
    bits += -Math.log2(Math.max(prob(hist, c), 1e-12));
    hist.push(c);
    if (hist.length > 8) hist.shift();
  }
  return bits / seq.length;
}

// ---------- experiment body (only when invoked directly) ----------
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

// ---------- brain (ladder-best genome) ----------
const ck = JSON.parse(readFileSync(
  new URL('./results/evolve-seed42.json', import.meta.url), 'utf8'));
const genome = ck.history[0].bestGenome; // gen0-best: the 42.5% genome
const t0 = Date.now();
const brain = developBrain(genome, N, SEED ^ (N * 2654435761));
console.log(
  `genbench · ladder-best genome @ ${N}n (${(brain.M / 1e6).toFixed(2)}M syn) · ` +
  `taps ${TAPS} · fit ${FIT_CHARS} · temp ${TEMP} · arms p=[${PS}]`
);

const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
const thrLayer = new Float32Array(brain.L).fill(1.0);
const decayPow = new Float64Array(256);
for (let d = 0; d < 256; d++) decayPow[d] = Math.exp(-d / TAU);
const ring = Array.from({ length: MAX_DELAY }, () => []);
const layerSpikes = new Float64Array(brain.L);
let t = 0;

const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
const inRng = mulberry32(SEED ^ 0xabc);
const inputTgt = new Int32Array(V * inputFan);
for (let c = 0; c < V; c++)
  for (let k = 0; k < inputFan; k++)
    inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

const featRng = mulberry32(SEED ^ 0xf00d);
const tapSlot = new Int32Array(N).fill(-1);
let taps = 0;
for (let f = 0; f < TAPS; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  const id = l * brain.layerSize + Math.floor(featRng() * brain.layerSize);
  if (tapSlot[id] < 0) tapSlot[id] = taps++;
}
const fast = new Float64Array(taps);
const decFast = Math.exp(-1 / TAU);

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

// calibration (frozen protocol)
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
console.log(`calibrated (${((Date.now() - t0) / 60000).toFixed(1)}m)`);

// ---------- rows / ridge ----------
const d = taps + 3 * V + 1;
function rowOf(cur, p1, p2) {
  const r = new Float32Array(d);
  r.set(fast.subarray(0, taps), 0);
  r[taps + cur] = 1;
  r[taps + V + p1] = 1;
  r[taps + 2 * V + p2] = 1;
  r[d - 1] = 1;
  return r;
}
function ridge(X, y, lambda = 1.0) {
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
const scScratch = new Float64Array(V);
function scoresOf(W, x) {
  scScratch.fill(0);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (!xi) continue;
    const Wi = W[i];
    for (let vv = 0; vv < V; vv++) scScratch[vv] += xi * Wi[vv];
  }
  return scScratch;
}
// PROTOCOL AMENDMENT (documented in EXPERIMENT.md): ridge scores are
// approximate class probabilities in [0,1] — softmaxing raw scores at
// T≈1 is near-uniform (max ratio e^{1/T}) and made generation read as
// noise regardless of model belief (v12's generation demo shared this
// flaw). Amended: clamp scores to a probability floor and apply
// power-law temperature p_i ∝ max(s_i, ε)^{1/T}; temp → 0 is greedy.
function sampleFrom(sc, temp, rng) {
  if (temp <= 0.01) { // greedy
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    return b;
  }
  let z = 0;
  const p = new Float64Array(V);
  for (let vv = 0; vv < V; vv++) { p[vv] = Math.pow(Math.max(sc[vv], 1e-4), 1 / temp); z += p[vv]; }
  let roll = rng() * z;
  for (let vv = 0; vv < V; vv++) { roll -= p[vv]; if (roll <= 0) return vv; }
  return V - 1;
}
function accOf(W, X, y) {
  let ok = 0;
  for (let s = 0; s < X.length; s++) {
    const sc = scoresOf(W, X[s]);
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
  }
  return ok / X.length;
}

// ---------- n-gram floor/ceiling ----------
const NGRAM_UPTO = 900000;
const prob = buildNgram(ids, V, NGRAM_UPTO);
const realHeldOut = [...ids.slice(NGRAM_UPTO + 1000, NGRAM_UPTO + 1000 + GEN_CHARS)];
const floorBpc = bpcOf(realHeldOut, prob);
const noiseRng = mulberry32(1);
const noiseBpc = bpcOf(
  Array.from({ length: GEN_CHARS }, () => Math.floor(noiseRng() * V)), prob);
console.log(
  `n-gram scorer (1-5 gram, ${(NGRAM_UPTO / 1e3) | 0}k chars): ` +
  `real held-out ${floorBpc.toFixed(2)} bpc · uniform noise ${noiseBpc.toFixed(2)} bpc\n`
);

// ---------- phase A: teacher-forced fit + test ----------
function collectTF(n, drive) {
  const X = [], y = [];
  for (let k = 0; k < n; k++) {
    const cur = ids[pos];
    runChar(cur);
    X.push(rowOf(cur, ids[pos - 1] ?? 0, ids[pos - 2] ?? 0));
    y.push(ids[pos + 1]);
    pos++;
  }
  return { X, y };
}
const fit = collectTF(FIT_CHARS);
const test = collectTF(TEST_CHARS);
const W0 = ridge(fit.X, fit.y);
const acc0 = accOf(W0, test.X, test.y);
console.log(`teacher-forced ridge: acc ${(acc0 * 100).toFixed(1)}% (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
fit.X.length = 0; // free

// ---------- generation (shared) ----------
const outPath = new URL(`./results/genbench-seed${SEED}.json`, import.meta.url);
const results = existsSync(outPath) ? JSON.parse(readFileSync(outPath, 'utf8')) : {
  SEED, N, TAPS, FIT_CHARS, TEMP, floorBpc, noiseBpc, arms: {},
};
function generate(W, tag, temp = TEMP) {
  // prime with real text, then free-run; brain hears its own output
  for (let k = 0; k < PRIME_CHARS; k++) runChar(ids[pos + k]);
  let c1 = ids[pos + PRIME_CHARS - 1], c2 = ids[pos + PRIME_CHARS - 2],
      c3 = ids[pos + PRIME_CHARS - 3];
  const rng = mulberry32(SEED ^ 0x9e9);
  const seq = [];
  for (let g = 0; g < GEN_CHARS; g++) {
    const sc = scoresOf(W, rowOf(c1, c2, c3));
    const next = sampleFrom(sc, temp, rng);
    seq.push(next);
    runChar(next);
    c3 = c2; c2 = c1; c1 = next;
  }
  const bpc = bpcOf(seq, prob);
  const sample = seq.slice(0, 200).map((c) => chars[c]).join('').replace(/\n/g, '\\n');
  console.log(`  [${tag}] gen ${bpc.toFixed(2)} bpc · "${sample.slice(0, 90)}…"`);
  return { bpc, sample };
}

// ---------- arms: scheduled sampling at p, then generate ----------
for (const p of PS) {
  const tag = `p=${p}`;
  if (results.arms[tag]) { console.log(`  [${tag}] checkpointed, skipping`); continue; }
  let W = W0, accTF = acc0;
  if (p > 0) {
    // collect mixed-driving rows: with prob p the char DRIVEN into the brain
    // (and placed in the context one-hots) is sampled from the current model;
    // regression targets remain the true next characters
    const rng = mulberry32(SEED ^ (p * 1e6));
    const X = [], y = [];
    let fed1 = ids[pos - 1] ?? 0, fed2 = ids[pos - 2] ?? 0, fed3 = ids[pos - 3] ?? 0;
    for (let k = 0; k < FIT_CHARS; k++) {
      const truth = ids[pos];
      let drive = truth;
      if (rng() < p) {
        const sc = scoresOf(W0, rowOf(fed1, fed2, fed3));
        drive = sampleFrom(sc, TEMP, rng);
      }
      runChar(drive);
      X.push(rowOf(drive, fed1, fed2));
      y.push(ids[pos + 1]);
      fed3 = fed2; fed2 = fed1; fed1 = drive;
      pos++;
    }
    W = ridge(X, y);
    accTF = accOf(W, test.X, test.y);
  }
  const gen = generate(W, tag);
  results.arms[tag] = { accTF, ...gen };
  if (p === 0) {
    const greedy = generate(W, 'p=0 greedy', GREEDY);
    results.arms['p=0-greedy'] = { accTF, ...greedy };
  }
  writeFileSync(outPath, JSON.stringify(results, null, 1));
  console.log(
    `  [${tag}] teacher-forced acc ${(accTF * 100).toFixed(1)}% · gen ${gen.bpc.toFixed(2)} bpc ` +
    `(floor ${floorBpc.toFixed(2)}, noise ${noiseBpc.toFixed(2)}) (${((Date.now() - t0) / 60000).toFixed(1)}m)`
  );
}
console.log(`\nsaved to experiments/results/genbench-seed${SEED}.json`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
