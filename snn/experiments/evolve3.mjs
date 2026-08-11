// v17 MEMORY-SELECTED SUBSTRATES: v13/v14 evolution kept deleting
// recurrence because next-char prediction never rewards remembering. Here
// the fitness IS memory. Three tasks, none solvable without the reservoir
// carrying information across characters — readout rows are TAPS + BIAS
// ONLY (no char one-hots: the reservoir is the only channel):
//
//   recall-2   8-symbol random stream, target = symbol from 2 chars ago
//   recall-4   same, 4 chars ago (40 sim-ms — beyond fast-trace decay)
//   parity-3   binary stream, target = XOR of last 3 bits
//
// Fitness = mean skill score (acc−chance)/(1−chance) over tasks × scales
// {2k, 8k} − α·std_scales − β·syn/n/100. Genome/ES identical to v14
// (13 genes incl n_layers, (μ+λ), common random numbers per generation).
//
// Pre-registered predictions (EXPERIMENT.md §v17):
//   P1  memory fitness RETAINS recurrence (rec_fan/delays > 0) —
//       reversing the v13/v14 deletion
//   P2  memory-selected substrates beat prediction-selected ones on
//       recall-4 by a wide margin (the latter should sit near chance)
//   P3  language transfer is the open question — a null there is itself
//       informative (next-char at 3-char context may not need memory)
//
// Transfer mode: memory tasks at held-out 16k/32k + LANGUAGE next-char
// (v14 protocol) for winner vs v14-winner vs v12-hand.
//
// Run:  npm run experiment:evolve3 [seed]
//       npm run experiment:evolve3 -- transfer <ckpt.json>

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../js/core/rng.js';
import {
  GENES, decode, encodeV12, encodePhys, developBrain,
  evaluate as evaluateLanguage,
} from './evolve2.mjs';

const STEPS_PER_CHAR = 10;
const TAU = 20;
const REFRAC = 4;
const MAX_DELAY = 16;
const TARGET_RATE = 0.002;

export const TASKS = [
  { name: 'recall2', vocab: 8, delay: 2, kind: 'recall' },
  { name: 'recall4', vocab: 8, delay: 4, kind: 'recall' },
  { name: 'parity3', vocab: 2, window: 3, kind: 'parity' },
];

export function makeTaskData(task, len, seed) {
  const rng = mulberry32(seed);
  const stream = new Int32Array(len);
  for (let i = 0; i < len; i++) stream[i] = Math.floor(rng() * task.vocab);
  const target = new Int32Array(len);
  const skip = task.kind === 'recall' ? task.delay : task.window;
  if (task.kind === 'recall') {
    for (let i = 0; i < len; i++) target[i] = i >= task.delay ? stream[i - task.delay] : 0;
    return { stream, target, nOut: task.vocab, chance: 1 / task.vocab, skip };
  }
  for (let i = 0; i < len; i++) {
    let p = 0;
    for (let w = 0; w < task.window; w++) if (i - w >= 0) p ^= stream[i - w];
    target[i] = p;
  }
  return { stream, target, nOut: 2, chance: 0.5, skip };
}

export function skillScore(acc, chance) {
  return Math.max(0, (acc - chance) / (1 - chance));
}

// ---------- memory evaluation: taps + bias ONLY ----------
export function evaluateMemory(genome, N, seed, task, cfg) {
  const brain = developBrain(genome, N, seed);
  const L = brain.L;
  const data = makeTaskData(task, cfg.calibChars + cfg.fitChars + cfg.testChars + 10, seed ^ 0x7a5c);
  const V = task.vocab;

  const v = new Float32Array(N), lastT = new Int32Array(N), refracUntil = new Int32Array(N);
  const thrLayer = new Float32Array(L).fill(1.0);
  const decayPow = new Float64Array(256);
  for (let d = 0; d < 256; d++) decayPow[d] = Math.exp(-d / TAU);
  const ring = Array.from({ length: MAX_DELAY }, () => []);
  const layerSpikes = new Float64Array(L);
  let t = 0;

  const inputFan = Math.max(48, Math.round(brain.layerSize / 120));
  const inRng = mulberry32(seed ^ 0xabc);
  const inputTgt = new Int32Array(V * inputFan);
  for (let c = 0; c < V; c++)
    for (let k = 0; k < inputFan; k++)
      inputTgt[c * inputFan + k] = Math.floor(inRng() * brain.layerSize);

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
  const aliveL = new Float64Array(L);
  for (let i = 0; i < N; i++) aliveL[brain.layer[i]]++;
  let pos = 0;
  for (let c = 0; c < cfg.calibChars; c++) {
    runChar(data.stream[pos++]);
    if ((c + 1) % 100 === 0) {
      for (let l = 0; l < L; l++) {
        const rate = layerSpikes[l] / (aliveL[l] * 100 * STEPS_PER_CHAR);
        const factor = Math.max(0.2, Math.min(8, (rate + 1e-6) / TARGET_RATE));
        thrLayer[l] = Math.max(0.5, Math.min(25, thrLayer[l] * Math.pow(factor, 0.6)));
        layerSpikes[l] = 0;
      }
    }
  }

  const d = taps + 1; // NO char context: reservoir or nothing
  const nOut = data.nOut;
  const X = [], y = [];
  for (let k = 0; k < cfg.fitChars + cfg.testChars; k++) {
    runChar(data.stream[pos]);
    const r = new Float32Array(d);
    r.set(fast.subarray(0, taps), 0);
    r[d - 1] = 1;
    X.push(r); y.push(data.target[pos]);
    pos++;
  }
  // ridge (small d, small nOut)
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const B = Array.from({ length: d }, () => new Float64Array(nOut));
  for (let s = 0; s < cfg.fitChars; s++) {
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
      for (let vv = 0; vv < nOut; vv++) B[r][vv] -= f * B[col][vv];
    }
  }
  const W = Array.from({ length: d }, (_, i) => {
    const row = new Float64Array(nOut);
    for (let vv = 0; vv < nOut; vv++) row[vv] = B[i][vv] / (A[i][i] || 1e-12);
    return row;
  });
  let ok = 0, count = 0;
  const sc = new Float64Array(nOut);
  for (let s = cfg.fitChars; s < cfg.fitChars + cfg.testChars; s++) {
    sc.fill(0);
    const x = X[s];
    for (let i = 0; i < d; i++) {
      const xi = x[i];
      if (!xi) continue;
      const Wi = W[i];
      for (let vv = 0; vv < nOut; vv++) sc[vv] += xi * Wi[vv];
    }
    let b = 0;
    for (let vv = 1; vv < nOut; vv++) if (sc[vv] > sc[b]) b = vv;
    if (b === y[s]) ok++;
    count++;
  }
  const acc = ok / count;
  return { acc, chance: data.chance, skill: skillScore(acc, data.chance), synPerNeuron: brain.M / N };
}

// ---------- fitness over tasks × scales ----------
function fitnessOf(genome, scales, seed, cfg, alpha, beta) {
  const perScale = [];
  let synPerNeuron = 0;
  const taskDetail = {};
  for (const N of scales) {
    let sum = 0;
    for (const task of TASKS) {
      const r = evaluateMemory(genome, N, (seed ^ (N * 2654435761)) >>> 0, task, cfg);
      sum += r.skill;
      synPerNeuron = r.synPerNeuron;
      (taskDetail[task.name] ??= []).push(+r.acc.toFixed(3));
    }
    perScale.push(sum / TASKS.length);
  }
  const mean = perScale.reduce((a, b) => a + b, 0) / perScale.length;
  const varr = perScale.reduce((a, b) => a + (b - mean) * (b - mean), 0) / perScale.length;
  return {
    F: mean - alpha * Math.sqrt(varr) - beta * (synPerNeuron / 100),
    meanSkill: mean, perScale, taskDetail, synPerNeuron: +synPerNeuron.toFixed(2),
  };
}

// ---------- ES (same shape as v13/v14) ----------
async function runEvolution(seed) {
  const SMOKE = process.env.SMOKE === '1';
  const CFG = {
    pop: Number(process.env.POP ?? (SMOKE ? 4 : 16)),
    gens: Number(process.env.GENS ?? (SMOKE ? 2 : 12)),
    scales: (process.env.SCALES ?? (SMOKE ? '500' : '2000,8000')).split(',').map(Number),
    taps: Number(process.env.TAPS ?? (SMOKE ? 64 : 256)),
    calibChars: SMOKE ? 200 : 800,
    fitChars: Number(process.env.FIT_CHARS ?? (SMOKE ? 500 : 4000)),
    testChars: SMOKE ? 200 : 1000,
    alpha: 0.5, beta: 0.02, sigma: 0.15, pGene: 0.5, pCross: 0.3, eliteFrac: 0.25,
  };
  const evalCfg = {
    taps: CFG.taps, calibChars: CFG.calibChars,
    fitChars: CFG.fitChars, testChars: CFG.testChars,
  };
  const outDir = new URL('./results/', import.meta.url).pathname;
  mkdirSync(outDir, { recursive: true });
  const ckptPath = `${outDir}evolve3-seed${seed}${SMOKE ? '-smoke' : ''}.json`;
  const esRng = mulberry32(seed ^ 0xe503);
  let population, history, gen0;
  if (existsSync(ckptPath)) {
    const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
    population = ckpt.population; history = ckpt.history; gen0 = history.length;
    for (let k = 0; k < (ckpt.rngDraws ?? 0); k++) esRng();
    console.log(`resuming from ${ckptPath} at generation ${gen0}`);
  } else {
    population = Array.from({ length: CFG.pop }, () => GENES.map(() => esRng()));
    history = []; gen0 = 0;
  }
  let rngDraws = history.length ? history[history.length - 1].rngDraws : CFG.pop * GENES.length;
  const draw = () => { rngDraws++; return esRng(); };

  console.log(
    `v17 memory-selected evolution · seed ${seed} · pop ${CFG.pop} · gens ${CFG.gens} · ` +
    `scales [${CFG.scales}] · tasks [${TASKS.map((t) => t.name)}] · taps+bias readout only`
  );
  const t0 = Date.now();
  const mutate = (norm) => norm.map((x) => {
    if (draw() >= CFG.pGene) return x;
    const u = Math.max(draw(), 1e-9), w = draw();
    return Math.min(1, Math.max(0, x + CFG.sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w)));
  });

  for (let gen = gen0; gen < CFG.gens; gen++) {
    const evalSeed = (seed ^ (0xbeef + gen * 101)) >>> 0;
    const scored = population.map((genome, i) => {
      const r = fitnessOf(genome, CFG.scales, evalSeed, evalCfg, CFG.alpha, CFG.beta);
      console.log(
        `  gen ${gen} #${String(i).padStart(2)} F=${r.F.toFixed(4)} skill=${r.meanSkill.toFixed(3)} ` +
        `recall4=[${r.taskDetail.recall4 ?? '-'}] L=${decode(genome).n_layers} rec=${decode(genome).rec_fan} ` +
        `(${((Date.now() - t0) / 60000).toFixed(1)}m)`
      );
      return { genome, ...r };
    });
    scored.sort((a, b) => b.F - a.F);
    const best = scored[0];
    history.push({
      gen, evalSeed, rngDraws, bestF: best.F, bestSkill: best.meanSkill,
      bestGenome: best.genome, bestPhys: decode(best.genome), bestTasks: best.taskDetail,
      meanF: +(scored.reduce((a, s) => a + s.F, 0) / scored.length).toFixed(4),
    });
    console.log(
      `gen ${gen}: best F=${best.F.toFixed(4)} skill=${best.meanSkill.toFixed(3)} ` +
      `tasks=${JSON.stringify(best.taskDetail)} · ${JSON.stringify(history.at(-1).bestPhys)}`
    );
    const mu = Math.max(2, Math.round(CFG.pop * CFG.eliteFrac));
    const elites = scored.slice(0, mu).map((s) => s.genome);
    const next = elites.slice();
    while (next.length < CFG.pop) {
      const pa = elites[Math.floor(draw() * mu)];
      const base = draw() < CFG.pCross
        ? pa.map((x, i) => (draw() < 0.5 ? x : elites[Math.floor(draw() * mu)][i]))
        : pa;
      next.push(mutate(base));
    }
    population = next;
    history[history.length - 1].rngDraws = rngDraws;
    writeFileSync(ckptPath, JSON.stringify({ seed, CFG, population, history, rngDraws }, null, 1));
  }

  // final report: memory skill of winner vs baselines on a fresh eval seed
  const finalSeed = (seed ^ 0xfade) >>> 0;
  const report = {};
  for (const [name, genome] of [
    ['v12-hand', encodeV12()],
    ['v14-winner', null], // filled below if checkpoint exists
    ['gen0-best', history[0].bestGenome],
    ['evolved-best', history.at(-1).bestGenome],
  ]) {
    let g = genome;
    if (name === 'v14-winner') {
      const p = `${outDir}evolve2-seed42.json`;
      if (!existsSync(p)) continue;
      g = JSON.parse(readFileSync(p, 'utf8')).history.at(-1).bestGenome;
    }
    const r = fitnessOf(g, CFG.scales, finalSeed, evalCfg, CFG.alpha, CFG.beta);
    report[name] = { F: r.F, meanSkill: r.meanSkill, tasks: r.taskDetail, phys: decode(g) };
    console.log(
      `${name.padEnd(13)} F=${r.F.toFixed(4)} skill=${r.meanSkill.toFixed(3)} ` +
      `tasks=${JSON.stringify(r.taskDetail)} rec_fan=${decode(g).rec_fan}`
    );
  }
  const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
  ckpt.finalReport = { finalSeed, report };
  writeFileSync(ckptPath, JSON.stringify(ckpt, null, 1));
  console.log(`\ncheckpoint: ${ckptPath}`);
}

// ---------- transfer: held-out memory scales + LANGUAGE ----------
async function runTransfer(ckptPath) {
  const ckpt = JSON.parse(readFileSync(ckptPath, 'utf8'));
  const outDir = new URL('./results/', import.meta.url).pathname;
  const evalCfg = { taps: 256, calibChars: 800, fitChars: 4000, testChars: 1000 };
  const arms = [['v12-hand', encodeV12()], ['evolved-best', ckpt.history.at(-1).bestGenome]];
  const v14p = `${outDir}evolve2-seed42.json`;
  if (existsSync(v14p)) {
    arms.splice(1, 0, ['v14-winner', JSON.parse(readFileSync(v14p, 'utf8')).history.at(-1).bestGenome]);
  }
  const seed = (ckpt.seed ^ 0x7a05) >>> 0;
  const t0 = Date.now();
  const transfer = { memory: {}, language: {} };
  for (const [name, genome] of arms) {
    transfer.memory[name] = {};
    for (const N of [16000, 32000]) {
      for (const task of TASKS) {
        const r = evaluateMemory(genome, N, (seed ^ (N * 2654435761)) >>> 0, task, evalCfg);
        (transfer.memory[name][task.name] ??= []).push({ N, acc: +r.acc.toFixed(3), chance: r.chance });
      }
      console.log(`${name.padEnd(13)} memory @${N}: ${TASKS.map((t) => `${t.name}=${(transfer.memory[name][t.name].at(-1).acc * 100).toFixed(1)}%`).join(' ')} (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
    }
  }
  // language transfer: v14 protocol (1024 taps, 40k fit, 120k neurons)
  const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;
  const text = readFileSync(DATA_PATH, 'utf8');
  const chars = [...new Set(text)].sort();
  const ids = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);
  for (const [name, genome] of arms) {
    const { acc, metrics } = evaluateLanguage(genome, 120000, (seed ^ (120000 * 2654435761)) >>> 0,
      ids, chars.length, { taps: 1024, calibChars: 5000, fitChars: 40000, testChars: 3000, startPos: 0 });
    transfer.language[name] = { acc, synPerNeuron: metrics.synPerNeuron, layers: metrics.layers };
    console.log(`${name.padEnd(13)} LANGUAGE @120k: ${(acc * 100).toFixed(1)}% (L=${metrics.layers}, syn/n=${metrics.synPerNeuron}) (${((Date.now() - t0) / 60000).toFixed(1)}m)`);
  }
  ckpt.transfer = { seed, transfer };
  writeFileSync(ckptPath, JSON.stringify(ckpt, null, 1));
  console.log(`\ntransfer appended to ${ckptPath}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const args = process.argv.slice(2);
  if (args[0] === 'transfer') {
    if (!args[1]) { console.error('usage: evolve3.mjs transfer <ckpt.json>'); process.exit(1); }
    await runTransfer(args[1]);
  } else {
    await runEvolution(Number(args[0] ?? 42));
  }
}
