// Autoregression: the readout already trains on the autoregressive
// factorization P(next | history) with teacher forcing. This experiment
// tests the two senses in which it is NOT yet autoregressive:
//
//   BELIEF FEEDBACK — feed the model's own posterior from the previous
//     position back as features (two-stage stacking: stage-1 ridge produces
//     teacher-forced posteriors, stage-2 ridge consumes them — no leakage).
//   GENERATION — actually sample: prime the spiking brain with real text,
//     then feed its own sampled characters back in and let it write.
//
// Also applies the v11 lesson: 40k fit samples (the previous 16k starved
// larger feature sets), and a second previous-char context.
//
// Arms: E0  1024 fast taps + 1 prev char   (v11 best, now at 40k fit)
//       E   + 2nd previous char
//       F   E + previous-position posterior (belief feedback)
// Then: 400 chars of free-running generation from the best arm.
//
// Run: npm run experiment:autoreg [seed]

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
const FIT_CHARS = 40000;
const TEST_CHARS = 3000;
const TAPS = 1024;
const TARGET_RATE = 0.002;
const GEN_CHARS = 400;
const GEN_TEMP = 0.7;

async function getText() {
  if (!existsSync(DATA_PATH)) {
    mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
    const res = await fetch(DATA_URL);
    writeFileSync(DATA_PATH, await res.text());
  }
  return readFileSync(DATA_PATH, 'utf8');
}

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
  return { N, L, layerSize, layer, synStart, synTgt, synW, synDelay, M };
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
console.log(`autoregression · brain ${N}n · vocab ${V} · fit ${FIT_CHARS} · seed ${SEED}`);
const t0 = Date.now();

const inRng = mulberry32(SEED ^ 0xabc);
const inputFan = 250;
const inputTgt = new Int32Array(V * inputFan);
for (let c = 0; c < V; c++) for (let k = 0; k < inputFan; k++) inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

const featRng = mulberry32(SEED ^ 0xf00d);
const tapSlot = new Int32Array(N).fill(-1);
for (let f = 0; f < TAPS; f++) {
  const l = Math.min(brain.L - 1, 1 + Math.floor(featRng() * (brain.L - 1)));
  tapSlot[l * brain.layerSize + Math.floor(featRng() * brain.layerSize)] = f;
}
const fast = new Float64Array(TAPS);
const decFast = Math.exp(-1 / TAU);
onSpike = (i) => {
  const f = tapSlot[i];
  if (f >= 0) fast[f] += 1;
};
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
console.log(`calibrated (${((Date.now() - t0) / 1000) | 0}s)`);

// base rows: taps + THREE char one-hots (cur, prev, prev2) + bias
const dBase = TAPS + 3 * V + 1;
function baseRow(cur, p1, p2) {
  const r = new Float32Array(dBase);
  r.set(fast, 0);
  r[TAPS + cur] = 1;
  r[TAPS + V + p1] = 1;
  r[TAPS + 2 * V + p2] = 1;
  r[dBase - 1] = 1;
  return r;
}
function collect(n) {
  const X = [], y = [];
  for (let k = 0; k < n; k++) {
    const cur = ids[pos];
    runChar(cur);
    X.push(baseRow(cur, pos > 0 ? ids[pos - 1] : 0, pos > 1 ? ids[pos - 2] : 0));
    y.push(ids[pos + 1]);
    pos++;
  }
  return { X, y };
}
const fit = collect(FIT_CHARS);
const test = collect(TEST_CHARS);
console.log(`collected (${((Date.now() - t0) / 60000).toFixed(1)} min)`);

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
function scoresOf(W, x, out) {
  out.fill(0);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (!xi) continue;
    const Wi = W[i];
    for (let vv = 0; vv < V; vv++) out[vv] += xi * Wi[vv];
  }
}
function softmaxInto(sc, temp) {
  let mx = -Infinity;
  for (let vv = 0; vv < V; vv++) if (sc[vv] > mx) mx = sc[vv];
  let z = 0;
  for (let vv = 0; vv < V; vv++) {
    sc[vv] = Math.exp((sc[vv] - mx) / temp);
    z += sc[vv];
  }
  for (let vv = 0; vv < V; vv++) sc[vv] /= z;
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

// arm E0: mask third one-hot (single prev char) — reuse rows with zeroed prev2
const mask2 = (X) => X.map((x) => {
  const r = Float32Array.from(x);
  r.fill(0, TAPS + 2 * V, TAPS + 3 * V);
  return r;
});
const W_E0 = ridge(mask2(fit.X), fit.y, dBase);
console.log(`E0  taps + 1 prev char @40k fit:  ${(accOf(W_E0, mask2(test.X), test.y) * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)} min)`);

const W_E = ridge(fit.X, fit.y, dBase);
const accE = accOf(W_E, test.X, test.y);
console.log(`E   taps + 2 prev chars:          ${(accE * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)} min)`);

// arm F: belief feedback — append previous position's stage-1 posterior
const dF = dBase + V;
function withBelief(X, y) {
  const rows = [];
  const sc = new Float64Array(V);
  let prevPost = new Float64Array(V).fill(1 / V);
  for (let s = 0; s < X.length; s++) {
    const r = new Float32Array(dF);
    r.set(X[s], 0);
    for (let vv = 0; vv < V; vv++) r[dBase + vv] = prevPost[vv];
    rows.push(r);
    scoresOf(W_E, X[s], sc);
    softmaxInto(sc, 2.0);
    prevPost = Float64Array.from(sc);
  }
  return rows;
}
const fitF = withBelief(fit.X, fit.y);
const testF = withBelief(test.X, test.y);
const W_F = ridge(fitF, fit.y, dF);
const accF = accOf(W_F, testF, test.y);
console.log(`F   E + belief feedback:          ${(accF * 100).toFixed(1)}%  (${((Date.now() - t0) / 60000).toFixed(1)} min)`);

// ---- free-running generation from arm E ----
console.log('\n— generation (primed with real text, then fed its own output) —');
const primeStart = pos + 100;
for (let k = 0; k < 100; k++) runChar(ids[primeStart + k]);
let c1 = ids[primeStart + 99];
let c2 = ids[primeStart + 98];
const genRng = mulberry32(SEED ^ 0x9e9);
const sc = new Float64Array(V);
let outText = '';
for (let g = 0; g < GEN_CHARS; g++) {
  const row = baseRow(c1, c2, c2); // note: row built from current trace state
  // fix context ordering: cur=c1, prev=c2 — rebuild properly below
  row.fill(0, TAPS, TAPS + 3 * V);
  row[TAPS + c1] = 1;
  row[TAPS + V + c2] = 1;
  scoresOf(W_E, row, sc);
  softmaxInto(sc, GEN_TEMP);
  let roll = genRng();
  let next = V - 1;
  for (let vv = 0; vv < V; vv++) {
    roll -= sc[vv];
    if (roll <= 0) { next = vv; break; }
  }
  outText += chars[next];
  runChar(next); // the brain hears its own output — true autoregression
  c2 = c1;
  c1 = next;
}
console.log(outText.replace(/\n/g, '\\n'));
console.log(`\ntotal ${((Date.now() - t0) / 60000).toFixed(1)} min · prior best 34.2% (16k fit) · bigram 28.8%`);
