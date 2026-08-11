// v16d THREE-FACTOR PLASTICITY: can the substrate learn to make itself
// easier to read? Reward-modulated STDP shapes the 1.34M synapses of the
// ladder-best phenotype during a streaming phase — local eligibility
// (pre-trace × post-spike coincidence) × a global scalar (was the online
// next-char prediction correct?) — no gradients anywhere. Then everything
// freezes and the standard ridge protocol measures whether the shaped
// substrate reads out better than the identical unshaped control.
//
//   eligibility  on post-spike of j: for every incoming synapse (CSC),
//                e += preTrace[src] (potentiation-only, τ_pre = 20ms)
//   reward       per char: online delta-rule softmax readout predicts the
//                next char from tap traces; r = +1 correct / −0.2 wrong
//   update       at char end: dw = η · r · e for synapses touched this
//                char; |w| clamped to [0.25×, 2.5×] of birth magnitude,
//                sign preserved; homeostat continues throughout
//   arms         η > 0 (plastic) vs η = 0 (control) — identical stream,
//                identical online readout, identical ridge eval after
//
// Stability guard: spikes/char logged; a seizure (>20k spikes/char
// sustained) is reported as a measured failure, not hidden.
//
// Run: npm run experiment:rstdp2 [seed]

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';
import { developBrain } from './evolve.mjs';

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const TAPS = 1024;
const CALIB_CHARS = 5000;
const TRAIN_CHARS = 60000; // plasticity phase
const FIT_CHARS = 40000;
const TEST_CHARS = 3000;
const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const TARGET_RATE = 0.002;
const ETA = 0.0015;
const R_WRONG = -0.2;
const W_LO = 0.25, W_HI = 2.5; // clamp factors vs birth |w|

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

const outPath = new URL(`./results/rstdp2-seed${SEED}.json`, import.meta.url);
const results = existsSync(outPath) ? JSON.parse(readFileSync(outPath, 'utf8'))
  : { SEED, N, TAPS, TRAIN_CHARS, ETA, R_WRONG, arms: {} };

for (const [tag, eta] of [['plastic', ETA], ['control', 0]]) {
  if (results.arms[tag]) {
    console.log(`${tag} (checkpointed): ridge ${(results.arms[tag].ridgeAcc * 100).toFixed(1)}%`);
    continue;
  }
  const t0 = Date.now();
  const brain = developBrain(genome, N, SEED ^ (N * 2654435761));
  const birthMag = new Float32Array(brain.M);
  for (let s = 0; s < brain.M; s++) birthMag[s] = Math.abs(brain.synW[s]);
  // CSC (incoming index) for eligibility on post-spike
  const inStart = new Int32Array(N + 1);
  for (let s = 0; s < brain.M; s++) inStart[brain.synTgt[s] + 1]++;
  for (let i = 0; i < N; i++) inStart[i + 1] += inStart[i];
  const inSyn = new Int32Array(brain.M); // synapse index
  const inSrc = new Int32Array(brain.M);
  {
    const cursor = inStart.slice(0, N);
    for (let i = 0; i < N; i++) {
      for (let s = brain.synStart[i]; s < brain.synStart[i + 1]; s++) {
        const p = cursor[brain.synTgt[s]]++;
        inSyn[p] = s; inSrc[p] = i;
      }
    }
  }
  console.log(`${tag} · η=${eta} · brain rebuilt, CSC indexed (${((Date.now() - t0) / 60000).toFixed(1)}m)`);

  const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
  const thrLayer = new Float32Array(brain.L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let dd = 0; dd < 256; dd++) decayPow[dd] = Math.exp(-dd / TAU);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const layerSpikes = new Float64Array(brain.L);
  let t = 0;
  const preTrace = new Float32Array(N);
  const preLast = new Int32Array(N);
  const elig = new Float32Array(brain.M);
  let touched = new Int32Array(1 << 20);
  let nTouched = 0;
  const inTouched = new Uint8Array(brain.M);
  let spikesThisChar = 0;

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
  const inRng = mulberry32(SEED ^ 0xabc);
  const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
  const inputTgt = new Int32Array(V * inputFan);
  for (let c = 0; c < V; c++)
    for (let k = 0; k < inputFan; k++)
      inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

  const plasticOn = eta > 0;
  function step(learn) {
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
        spikesThisChar++;
        for (let s = brain.synStart[i]; s < brain.synStart[i + 1]; s++) {
          ring[(t + Math.min(brain.synDelay[s], MAX_DELAY - 1)) % MAX_DELAY]
            .push(brain.synTgt[s], brain.synW[s]);
        }
        const f = tapSlot[i];
        if (f >= 0) fast[f] += 1;
        // pre-trace bookkeeping (decayed to now)
        const pdt = t - preLast[i];
        preTrace[i] = preTrace[i] * decayPow[pdt > 255 ? 255 : pdt] + 1;
        preLast[i] = t;
        if (learn && plasticOn) {
          // post-spike: eligibility on incoming synapses ∝ presynaptic trace
          for (let p = inStart[i]; p < inStart[i + 1]; p++) {
            const src = inSrc[p];
            let pt = preTrace[src];
            if (!pt) continue;
            const sdt = t - preLast[src];
            if (sdt > 0) pt *= decayPow[sdt > 255 ? 255 : sdt];
            if (pt < 0.05) continue;
            const syn = inSyn[p];
            elig[syn] += pt;
            if (!inTouched[syn]) {
              inTouched[syn] = 1;
              if (nTouched === touched.length) {
                const bigger = new Int32Array(touched.length * 2);
                bigger.set(touched); touched = bigger;
              }
              touched[nTouched++] = syn;
            }
          }
        }
      }
    }
    bucket.length = 0;
    t++;
  }
  function runChar(c, learn) {
    const base = c * inputFan;
    for (let k = 0; k < inputFan; k++) ring[(t + 1) % MAX_DELAY].push(inputTgt[base + k], 1.2);
    for (let s = 0; s < STEPS_PER_CHAR; s++) {
      step(learn);
      for (let i = 0; i < taps; i++) fast[i] *= decFast;
    }
  }
  const aliveL = new Float64Array(brain.L);
  for (let i = 0; i < N; i++) aliveL[brain.layer[i]]++;
  function homeostat(win) {
    for (let l = 0; l < brain.L; l++) {
      const rate = layerSpikes[l] / (aliveL[l] * win * STEPS_PER_CHAR);
      const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
      thrLayer[l] = Math.max(0.5, Math.min(25, thrLayer[l] * Math.pow(factor, 0.6)));
      layerSpikes[l] = 0;
    }
  }
  let pos = 0;
  for (let c = 0; c < CALIB_CHARS; c++) {
    runChar(ids[pos++], false);
    if ((c + 1) % 100 === 0) homeostat(100);
  }

  // online delta-rule readout supplies the reward
  const dOn = taps + V + 1;
  const Won = Array.from({ length: V }, () => new Float64Array(dOn));
  const px = new Float64Array(V);
  function onlinePredictLearn(c, target, lr) {
    let maxs = -Infinity, pred = 0;
    for (let vv = 0; vv < V; vv++) {
      let s = 0;
      const Wv = Won[vv];
      for (let i = 0; i < taps; i++) if (fast[i]) s += Wv[i] * fast[i];
      s += Wv[taps + c] + Wv[dOn - 1];
      px[vv] = s;
      if (s > maxs) { maxs = s; pred = vv; }
    }
    let z = 0;
    for (let vv = 0; vv < V; vv++) { px[vv] = Math.exp(px[vv] - maxs); z += px[vv]; }
    for (let vv = 0; vv < V; vv++) {
      const g = px[vv] / z - (vv === target ? 1 : 0);
      if (!g) continue;
      const Wv = Won[vv];
      const stepw = lr * g;
      for (let i = 0; i < taps; i++) if (fast[i]) Wv[i] -= stepw * fast[i];
      Wv[taps + c] -= stepw;
      Wv[dOn - 1] -= stepw;
    }
    return pred === target;
  }

  // ---------- plasticity phase (both arms stream identically) ----------
  let onlineOk = 0, seen = 0, spikeSum = 0;
  const onlineCurve = [];
  for (let k = 0; k < TRAIN_CHARS; k++) {
    spikesThisChar = 0;
    nTouched = 0;
    const c = ids[pos];
    runChar(c, true);
    const correct = onlinePredictLearn(c, ids[pos + 1], 0.03 / (1 + k / 20000));
    if (plasticOn) {
      const r = correct ? 1 : R_WRONG;
      const f = eta * r;
      for (let ti = 0; ti < nTouched; ti++) {
        const syn = touched[ti];
        const w = brain.synW[syn];
        const mag = Math.abs(w) + f * elig[syn] * Math.abs(w); // multiplicative, sign-preserving
        const lo = birthMag[syn] * W_LO, hi = birthMag[syn] * W_HI;
        const clamped = Math.max(lo, Math.min(hi, mag));
        brain.synW[syn] = w < 0 ? -clamped : clamped;
        elig[syn] = 0;
        inTouched[syn] = 0;
      }
    }
    onlineOk += correct ? 1 : 0;
    spikeSum += spikesThisChar;
    seen++;
    pos++;
    if ((k + 1) % 500 === 0) homeostat(500);
    if ((k + 1) % 10000 === 0) {
      onlineCurve.push({ k: k + 1, acc: onlineOk / seen, spikesPerChar: spikeSum / seen });
      console.log(
        `  [${tag}] ${k + 1} chars: online ${(100 * onlineOk / seen).toFixed(1)}% · ` +
        `${(spikeSum / seen).toFixed(0)} spikes/char (${((Date.now() - t0) / 60000).toFixed(1)}m)`
      );
      onlineOk = 0; seen = 0; spikeSum = 0;
    }
  }

  // ---------- frozen ridge eval (standard protocol) ----------
  const d = taps + 3 * V + 1;
  const X = [], y = [];
  for (let k = 0; k < FIT_CHARS + TEST_CHARS; k++) {
    const cur = ids[pos];
    runChar(cur, false);
    const r = new Float32Array(d);
    r.set(fast.subarray(0, taps), 0);
    r[taps + cur] = 1;
    r[taps + V + ids[pos - 1]] = 1;
    r[taps + 2 * V + ids[pos - 2]] = 1;
    r[d - 1] = 1;
    X.push(r); y.push(ids[pos + 1]);
    pos++;
  }
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(V));
  for (let s = 0; s < FIT_CHARS; s++) {
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
    A[i][i] += 1.0;
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
  const W = Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(V);
    for (let vv = 0; vv < V; vv++) row[vv] = B[i][vv] / (A[i][i] || 1e-12);
    return row;
  });
  const sc = new Float64Array(V);
  let ok = 0;
  for (let s = FIT_CHARS; s < FIT_CHARS + TEST_CHARS; s++) {
    sc.fill(0);
    const x = X[s];
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Wi = W[i];
      for (let vv = 0; vv < V; vv++) sc[vv] += xi * Wi[vv];
    }
    let b = 0;
    for (let vv = 1; vv < V; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
  }
  const ridgeAcc = ok / TEST_CHARS;
  // weight-change census
  let changed = 0, magSum = 0;
  for (let s = 0; s < brain.M; s++) {
    const dm = Math.abs(Math.abs(brain.synW[s]) - birthMag[s]);
    if (dm > 1e-6) { changed++; magSum += dm / birthMag[s]; }
  }
  results.arms[tag] = {
    eta, ridgeAcc, onlineCurve,
    changedSynapses: changed,
    meanRelChange: changed ? +(magSum / changed).toFixed(4) : 0,
  };
  writeFileSync(outPath, JSON.stringify(results, null, 1));
  console.log(
    `${tag}: ridge ${(ridgeAcc * 100).toFixed(1)}% · ${changed} synapses changed ` +
    `(mean rel Δ ${results.arms[tag].meanRelChange}) (${((Date.now() - t0) / 60000).toFixed(1)}m)`
  );
}
console.log(`\nsaved to ${outPath.pathname}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
