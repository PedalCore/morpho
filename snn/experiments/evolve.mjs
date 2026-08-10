// v13 EVOLUTION OF DEVELOPMENT: evolve the parameters of the wiring LAW,
// never the network. An 11-gene genome parameterizes the same per-neuron
// statistical developmental rule that built the v10-v12 structured brains
// (fan-outs, E/I ratio, weight scales, recurrent/feedback gains, delay
// spreads) — genome length is constant in N, so the same genome can be
// instantiated at any size. The readout pipeline is v12's, frozen: fast
// spike traces on sampled taps + current/prev/prev2 char one-hots + bias,
// closed-form ridge, held-out accuracy. If fitness moves, development moved
// it — not the learner.
//
// Design (pre-registered in EXPERIMENT.md §v13):
//   fitness   F(g) = mean_N acc − α·std_N acc − β·synPerNeuron/100
//             over evolution scales N ∈ {2k, 4k, 8k} — selecting for scale
//             ROBUSTNESS, not accuracy at one size
//   optimizer boring (μ+λ) ES: top 25% survive, uniform crossover + gaussian
//             mutation in normalized gene space; common random numbers per
//             generation (all genomes share build seed + corpus window,
//             rotated each generation); elites re-evaluated every generation
//   baselines v12 hand genome · random genomes (gen-0 population) · best of
//             gen 0 — all through the identical pipeline
//   transfer  frozen winners instantiated at held-out 16k/32k/60k — and 120k
//             stays genuinely held out behind an explicit --120k flag
//
// Run:  npm run experiment:evolve [seed]                 — evolve + checkpoint
//       npm run experiment:evolve -- transfer <ckpt.json> [--120k]
// Env:  POP GENS FIT_CHARS TAPS SCALES  (see CFG below; SMOKE=1 → tiny run)

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';

// ---------- genome ----------
// Each gene is a developmental law parameter: lo/hi is the legal range,
// v12 is the hand-designed value from bigbrain/readout2/autoreg. Genomes
// are stored NORMALIZED in [0,1]; decode() maps to physical values.

export const GENES = [
  { key: 'inhib_frac',  lo: 0.05, hi: 0.35, v12: 0.15 },          // excitatory_ratio
  { key: 'ff_fan',      lo: 2,    hi: 28,   v12: 14, int: true }, // branching (forward)
  { key: 'rec_fan',     lo: 0,    hi: 16,   v12: 6,  int: true }, // recurrent_connection_bias
  { key: 'skip_fan',    lo: 0,    hi: 8,    v12: 3,  int: true }, // long-range forward
  { key: 'fb_fan',      lo: 0,    hi: 6,    v12: 2,  int: true }, // long-range backward
  { key: 'inh_fan',     lo: 4,    hi: 40,   v12: 22, int: true }, // inhibitory branching
  { key: 'w_exc',       lo: 0.15, hi: 0.55, v12: 0.28 },          // excitatory weight base
  { key: 'w_inh',       lo: 0.2,  hi: 1.2,  v12: 0.5 },           // inhibitory weight base
  { key: 'rec_gain',    lo: 0.1,  hi: 1.0,  v12: 0.45 },          // recurrent damping
  { key: 'fb_gain',     lo: 0.05, hi: 0.8,  v12: 0.3 },           // feedback damping
  { key: 'delay_scale', lo: 0.3,  hi: 2.0,  v12: 1.0 },           // temporal wiring spread
];

export function decode(norm) {
  const g = {};
  for (let i = 0; i < GENES.length; i++) {
    const { key, lo, hi, int } = GENES[i];
    const x = lo + Math.min(1, Math.max(0, norm[i])) * (hi - lo);
    g[key] = int ? Math.round(x) : x;
  }
  return g;
}

export function encodeV12() {
  return GENES.map(({ lo, hi, v12 }) => (v12 - lo) / (hi - lo));
}

// ---------- development: genome + size → brain (CSR) ----------
// Generalizes the v10 builder. L=4 layers is FIXED in v13 (structural genes
// are v14's question); everything statistical about the wiring is genomic.

const L = 4;
const MAX_DELAY = 16;

export function developBrain(genome, N, seed) {
  const g = decode(genome);
  const rng = mulberry32(seed);
  const layerSize = Math.floor(N / L);
  const layer = new Uint8Array(N);
  for (let i = 0; i < N; i++) layer[i] = Math.min(L - 1, Math.floor(i / layerSize));
  const inhibitory = new Uint8Array(N);
  for (let i = 0; i < N; i++) if (rng() < g.inhib_frac) inhibitory[i] = 1;

  const pickIn = (l) => l * layerSize + Math.floor(rng() * layerSize);
  const srcs = [], tgts = [], ws = [], dls = [];
  // weight jitter kept proportional to base (v12's spread/base ratios)
  const wE = () => g.w_exc * (1 + rng() * 0.786);
  const wI = () => -g.w_inh * (1 + rng() * 0.6);
  const del = (base, spread) =>
    Math.max(1, Math.min(MAX_DELAY - 1, base + Math.floor(rng() * spread * g.delay_scale)));
  const kinds = { ff: 0, rec: 0, skip: 0, fb: 0, inh: 0 };
  for (let i = 0; i < N; i++) {
    const l = layer[i];
    const add = (t, w, d, kind) => {
      srcs.push(i); tgts.push(t); ws.push(w); dls.push(d); kinds[kind]++;
    };
    if (inhibitory[i]) {
      for (let k = 0; k < g.inh_fan; k++) add(pickIn(l), wI(), del(1, 3), 'inh');
    } else {
      if (l + 1 < L) for (let k = 0; k < g.ff_fan; k++) add(pickIn(l + 1), wE(), del(1, 4), 'ff');
      for (let k = 0; k < g.rec_fan; k++) add(pickIn(l), wE() * g.rec_gain, del(1, 6), 'rec');
      if (l + 2 < L) for (let k = 0; k < g.skip_fan; k++) add(pickIn(l + 2), wE(), del(2, 6), 'skip');
      if (l > 0) for (let k = 0; k < g.fb_fan; k++) add(pickIn(l - 1), wE() * g.fb_gain, del(3, 8), 'fb');
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
  return { N, L, layerSize, layer, inhibitory, synStart, synTgt, synW, synDelay, M, kinds };
}

// ---------- frozen v12 physiology + readout ----------

const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const TARGET_RATE = 0.002; // ~2 Hz sparse — part of the frozen protocol, not the genome

export function evaluate(genome, N, seed, ids, V, cfg) {
  const tBuild = Date.now();
  const brain = developBrain(genome, N, seed);
  const buildMs = Date.now() - tBuild;
  const tSim = Date.now();

  const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
  const thrLayer = new Float32Array(L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let d = 0; d < 256; d++) decayPow[d] = Math.exp(-d / TAU);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const layerSpikes = new Float64Array(L);
  const spikeCount = new Uint32Array(N);
  let t = 0;
  let spikesTotal = 0;

  // input wiring is a developmental constant of the protocol: same fractional
  // fan into layer 0 at every size (v12's 250/30000), floor for legibility
  const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
  const inRng = mulberry32(seed ^ 0xabc);
  const inputTgt = new Int32Array(V * inputFan);
  for (let c = 0; c < V; c++)
    for (let k = 0; k < inputFan; k++)
      inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

  // taps: fixed readout budget at every size (fairness across scales)
  const featRng = mulberry32(seed ^ 0xf00d);
  const tapSlot = new Int32Array(N).fill(-1);
  let taps = 0;
  for (let f = 0; f < cfg.taps; f++) {
    const l = Math.min(L - 1, 1 + Math.floor(featRng() * (L - 1)));
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
        spikeCount[i]++; spikesTotal++;
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

  // homeostatic calibration (identical schedule to v12)
  const aliveL = new Float64Array(L);
  for (let i = 0; i < N; i++) aliveL[brain.layer[i]]++;
  let pos = cfg.startPos;
  for (let c = 0; c < cfg.calibChars; c++) {
    runChar(ids[pos++]);
    if ((c + 1) % 100 === 0) {
      for (let l = 0; l < L; l++) {
        const rate = layerSpikes[l] / (aliveL[l] * 100 * STEPS_PER_CHAR);
        const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
        thrLayer[l] = Math.max(0.5, Math.min(25, thrLayer[l] * Math.pow(factor, 0.6)));
        layerSpikes[l] = 0;
      }
    }
  }

  // collect rows: taps + cur/prev/prev2 one-hots + bias (v12's arm E, frozen)
  const d = taps + 3 * V + 1;
  function collect(n) {
    const X = [], y = [];
    for (let k = 0; k < n; k++) {
      const cur = ids[pos];
      runChar(cur);
      const r = new Float32Array(d);
      r.set(fast.subarray(0, taps), 0);
      r[taps + cur] = 1;
      r[taps + V + (pos > 0 ? ids[pos - 1] : 0)] = 1;
      r[taps + 2 * V + (pos > 1 ? ids[pos - 2] : 0)] = 1;
      r[d - 1] = 1;
      X.push(r); y.push(ids[pos + 1]);
      pos++;
    }
    return { X, y };
  }
  spikeCount.fill(0); spikesTotal = 0;
  const measuredChars = cfg.fitChars + cfg.testChars;
  const fit = collect(cfg.fitChars);
  const test = collect(cfg.testChars);
  const acc = accOf(ridge(fit.X, fit.y, d, V), test.X, test.y, V);

  let dead = 0;
  for (let i = 0; i < N; i++) if (!spikeCount[i]) dead++;
  const k = brain.kinds;
  return {
    acc,
    metrics: {
      neurons: N,
      synapses: brain.M,
      synPerNeuron: +(brain.M / N).toFixed(2),
      inhibFrac: +([...brain.inhibitory].reduce((a, b) => a + b, 0) / N).toFixed(3),
      recFrac: +((k.rec + k.inh) / brain.M).toFixed(3),
      longFrac: +((k.skip + k.fb) / brain.M).toFixed(3),
      spikesPerChar: +(spikesTotal / measuredChars).toFixed(1),
      hz: +((spikesTotal / measuredChars / N / STEPS_PER_CHAR) * 1000).toFixed(2),
      deadFrac: +(dead / N).toFixed(3),
      taps,
      buildMs,
      simMs: Date.now() - tSim,
    },
  };
}

function ridge(X, y, d, V, lambda = 1.0) {
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

function accOf(W, X, y, V) {
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

// ---------- multi-scale fitness ----------

export function fitnessOf(perScaleAccs, synPerNeuron, alpha, beta) {
  const mean = perScaleAccs.reduce((a, b) => a + b, 0) / perScaleAccs.length;
  const varr =
    perScaleAccs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / perScaleAccs.length;
  return mean - alpha * Math.sqrt(varr) - beta * (synPerNeuron / 100);
}

function evalGenome(genome, scales, seed, startPos, ids, V, cfg, alpha, beta) {
  const accs = [], scaleMetrics = [];
  for (const N of scales) {
    const { acc, metrics } = evaluate(genome, N, seed ^ (N * 2654435761), ids, V, {
      ...cfg, startPos,
    });
    accs.push(acc);
    scaleMetrics.push(metrics);
  }
  const synPerNeuron = scaleMetrics[scaleMetrics.length - 1].synPerNeuron;
  return { accs, scaleMetrics, F: fitnessOf(accs, synPerNeuron, alpha, beta) };
}

// ---------- (μ+λ) evolution ----------

async function getIds() {
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
  const ids = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);
  return { ids, V: chars.length, chars };
}

function mutate(norm, rng, sigma, pGene) {
  const child = norm.slice();
  for (let i = 0; i < child.length; i++) {
    if (rng() < pGene) {
      // Box-Muller gaussian
      const u = Math.max(rng(), 1e-9), w = rng();
      child[i] += sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
      child[i] = Math.min(1, Math.max(0, child[i]));
    }
  }
  return child;
}

function crossover(a, b, rng) {
  return a.map((x, i) => (rng() < 0.5 ? x : b[i]));
}

async function runEvolution(seed) {
  const SMOKE = process.env.SMOKE === '1';
  const CFG = {
    pop: Number(process.env.POP ?? (SMOKE ? 4 : 16)),
    gens: Number(process.env.GENS ?? (SMOKE ? 2 : 12)),
    scales: (process.env.SCALES ?? (SMOKE ? '500,1000' : '2000,4000,8000'))
      .split(',').map(Number),
    taps: Number(process.env.TAPS ?? (SMOKE ? 64 : 256)),
    calibChars: Number(process.env.CALIB_CHARS ?? (SMOKE ? 300 : 1500)),
    fitChars: Number(process.env.FIT_CHARS ?? (SMOKE ? 800 : 8000)),
    testChars: Number(process.env.TEST_CHARS ?? (SMOKE ? 300 : 1500)),
    alpha: Number(process.env.ALPHA ?? 0.5),
    beta: Number(process.env.BETA ?? 0.02),
    sigma: 0.15,
    pGene: 0.5,
    pCross: 0.3,
    eliteFrac: 0.25,
  };
  const evalCfg = {
    taps: CFG.taps, calibChars: CFG.calibChars,
    fitChars: CFG.fitChars, testChars: CFG.testChars,
  };
  const { ids, V } = await getIds();
  const windowLen = CFG.calibChars + CFG.fitChars + CFG.testChars + 10;
  const genWindow = (gen) => (37013 * (gen + 1)) % (ids.length - windowLen - 1);

  const outDir = new URL('./results/', import.meta.url).pathname;
  mkdirSync(outDir, { recursive: true });
  const ckptPath = `${outDir}evolve-seed${seed}${SMOKE ? '-smoke' : ''}.json`;

  const esRng = mulberry32(seed ^ 0xe501);
  let population, history, gen0;
  if (existsSync(ckptPath)) {
    const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
    population = ckpt.population;
    history = ckpt.history;
    gen0 = history.length;
    // resume the ES RNG deterministically: replay the stream consumption count
    for (let k = 0; k < (ckpt.rngDraws ?? 0); k++) esRng();
    console.log(`resuming from ${ckptPath} at generation ${gen0}`);
  } else {
    population = Array.from({ length: CFG.pop }, () => GENES.map(() => esRng()));
    history = [];
    gen0 = 0;
  }
  let rngDraws = history.length ? history[history.length - 1].rngDraws : CFG.pop * GENES.length;
  const draw = () => { rngDraws++; return esRng(); };

  console.log(
    `v13 evolution · seed ${seed} · pop ${CFG.pop} · gens ${CFG.gens} · ` +
    `scales [${CFG.scales}] · taps ${CFG.taps} · fit ${CFG.fitChars} · ` +
    `F = mean − ${CFG.alpha}·std − ${CFG.beta}·syn/n/100`
  );
  const t0 = Date.now();

  for (let gen = gen0; gen < CFG.gens; gen++) {
    const evalSeed = (seed ^ (0xbeef + gen * 101)) >>> 0;
    const startPos = genWindow(gen);
    const scored = population.map((genome, i) => {
      const r = evalGenome(genome, CFG.scales, evalSeed, startPos, ids, V, evalCfg, CFG.alpha, CFG.beta);
      console.log(
        `  gen ${gen} #${String(i).padStart(2)} F=${r.F.toFixed(4)} ` +
        `acc=[${r.accs.map((a) => (a * 100).toFixed(1)).join(' ')}]% ` +
        `syn/n=${r.scaleMetrics.at(-1).synPerNeuron} (${((Date.now() - t0) / 60000).toFixed(1)}m)`
      );
      return { genome, ...r };
    });
    scored.sort((a, b) => b.F - a.F);
    const best = scored[0];
    history.push({
      gen, evalSeed, startPos, rngDraws,
      bestF: best.F, bestAccs: best.accs, bestGenome: best.genome,
      bestPhys: decode(best.genome), bestMetrics: best.scaleMetrics,
      meanF: +(scored.reduce((a, s) => a + s.F, 0) / scored.length).toFixed(4),
      all: scored.map((s) => ({ F: s.F, accs: s.accs, synPerNeuron: s.scaleMetrics.at(-1).synPerNeuron })),
    });
    console.log(
      `gen ${gen}: best F=${best.F.toFixed(4)} acc=[${best.accs.map((a) => (a * 100).toFixed(1)).join(' ')}]% ` +
      `mean F=${history.at(-1).meanF} · ${JSON.stringify(history.at(-1).bestPhys)}`
    );

    // (μ+λ): elites survive as-is (re-evaluated next gen on fresh window)
    const mu = Math.max(2, Math.round(CFG.pop * CFG.eliteFrac));
    const elites = scored.slice(0, mu).map((s) => s.genome);
    const next = elites.slice();
    while (next.length < CFG.pop) {
      const pa = elites[Math.floor(draw() * mu)];
      const base = draw() < CFG.pCross ? crossover(pa, elites[Math.floor(draw() * mu)], draw) : pa;
      next.push(mutate(base, draw, CFG.sigma, CFG.pGene));
    }
    population = next;
    history[history.length - 1].rngDraws = rngDraws;
    writeFileSync(ckptPath, JSON.stringify({ seed, CFG, population, history, rngDraws }, null, 1));
  }

  // final report: v12 baseline + gen-0 best + evolved best on ONE fresh window
  const finalSeed = (seed ^ 0xfade) >>> 0;
  const finalPos = genWindow(CFG.gens + 7);
  const report = {};
  for (const [name, genome] of [
    ['v12-hand', encodeV12()],
    ['gen0-best', history[0].bestGenome],
    ['evolved-best', history.at(-1).bestGenome],
  ]) {
    const r = evalGenome(genome, CFG.scales, finalSeed, finalPos, ids, V, evalCfg, CFG.alpha, CFG.beta);
    report[name] = { F: r.F, accs: r.accs, phys: decode(genome), metrics: r.scaleMetrics };
    console.log(
      `${name.padEnd(13)} F=${r.F.toFixed(4)} acc=[${r.accs.map((a) => (a * 100).toFixed(1)).join(' ')}]% ` +
      `syn/n=${r.scaleMetrics.at(-1).synPerNeuron}`
    );
  }
  const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
  ckpt.finalReport = { finalSeed, finalPos, report };
  writeFileSync(ckptPath, JSON.stringify(ckpt, null, 1));
  console.log(`\ncheckpoint: ${ckptPath}`);
  console.log('next: npm run experiment:evolve -- transfer ' + ckptPath);
}

// ---------- frozen-genome scale transfer ----------

async function runTransfer(ckptPath, include120k) {
  const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
  const { ids, V } = await getIds();
  const CFG = ckpt.CFG;
  const evalCfg = {
    taps: CFG.taps, calibChars: CFG.calibChars,
    fitChars: CFG.fitChars, testChars: CFG.testChars,
  };
  const scales = include120k ? [16000, 32000, 60000, 120000] : [16000, 32000, 60000];
  console.log(
    `v13 transfer · held-out scales [${scales}] · readout budget frozen ` +
    `(taps ${CFG.taps}, fit ${CFG.fitChars})${include120k ? '' : ' · 120k still held out (--120k)'}`
  );
  const t0 = Date.now();
  const seed = (ckpt.seed ^ 0x7a05) >>> 0;
  const windowLen = CFG.calibChars + CFG.fitChars + CFG.testChars + 10;
  const startPos = (37013 * 97) % (ids.length - windowLen - 1);
  const arms = [
    ['v12-hand', encodeV12()],
    ['gen0-best', ckpt.history[0].bestGenome],
    ['evolved-best', ckpt.history.at(-1).bestGenome],
  ];
  const transfer = {};
  for (const [name, genome] of arms) {
    transfer[name] = [];
    for (const N of scales) {
      const { acc, metrics } = evaluate(genome, N, seed ^ (N * 2654435761), ids, V, {
        ...evalCfg, startPos,
      });
      transfer[name].push({ N, acc, metrics });
      console.log(
        `${name.padEnd(13)} N=${String(N).padStart(6)}  acc=${(acc * 100).toFixed(1)}%  ` +
        `syn/n=${metrics.synPerNeuron} spikes/char=${metrics.spikesPerChar} ` +
        `dead=${(metrics.deadFrac * 100).toFixed(0)}% (${((Date.now() - t0) / 60000).toFixed(1)}m)`
      );
    }
  }
  ckpt.transfer = { scales, seed, startPos, transfer, include120k };
  writeFileSync(ckptPath, JSON.stringify(ckpt, null, 1));
  console.log(`\ntransfer results appended to ${ckptPath}`);
}

// ---------- main ----------

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const args = process.argv.slice(2);
  if (args[0] === 'transfer') {
    const ckpt = args[1];
    if (!ckpt) {
      console.error('usage: node experiments/evolve.mjs transfer <checkpoint.json> [--120k]');
      process.exit(1);
    }
    await runTransfer(ckpt, args.includes('--120k'));
  } else {
    await runEvolution(Number(args[0] ?? 42));
  }
}
