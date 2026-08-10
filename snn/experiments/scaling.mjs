// v15 SCALING SWEEP: find where the linear readout actually saturates,
// instead of extrapolating. Two axes around the 42.5% operating point
// (ladder-best genome, 120k neurons, frozen protocol):
//
//   taps axis:  256 / 512 / 1024 / 2048   at fit 40k
//   data axis:  10k / 20k / 40k / 80k     at taps 1024
//
// Trained parameters per config = (taps + 3·65 + 1) × 65. The hypothesis
// from the ladder data (~+4pp per 2.5× fit data, linear model ⇒ n-gram-
// class ceiling): both axes bend toward a mid-40s plateau. Per-config
// checkpointing; one process, sequential.
//
// Run: npm run experiment:scaling [seed]

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { evaluate } from './evolve.mjs';

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const CALIB = 5000;
const TEST = 3000;

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
const genome = ck.history[0].bestGenome; // the 42.5% ladder-best genome

const CONFIGS = [
  { taps: 256,  fit: 40000 },
  { taps: 512,  fit: 40000 },
  { taps: 1024, fit: 40000 }, // operating point (≈ ladder best)
  { taps: 2048, fit: 40000 },
  { taps: 1024, fit: 10000 },
  { taps: 1024, fit: 20000 },
  { taps: 1024, fit: 80000 },
  // joint frontier: the data axis saturated at fixed d — does more d re-open it?
  { taps: 2048, fit: 80000 },
  { taps: 4096, fit: 80000 },
];

const outPath = new URL(`./results/scaling-seed${SEED}.json`, import.meta.url);
const results = existsSync(outPath)
  ? JSON.parse(readFileSync(outPath, 'utf8'))
  : { SEED, N, CALIB, TEST, configs: {} };

console.log(`scaling sweep · ladder-best genome @ ${N}n · ${CONFIGS.length} configs · seed ${SEED}`);
const t0 = Date.now();
for (const { taps, fit } of CONFIGS) {
  const tag = `taps${taps}-fit${fit}`;
  if (results.configs[tag]) {
    console.log(`${tag.padEnd(18)} acc=${(results.configs[tag].acc * 100).toFixed(1)}%  (checkpointed)`);
    continue;
  }
  const { acc, metrics } = evaluate(genome, N, SEED ^ (N * 2654435761), ids, V, {
    taps, calibChars: CALIB, fitChars: fit, testChars: TEST, startPos: 0,
  });
  const trained = (metrics.taps + 3 * V + 1) * V;
  results.configs[tag] = { taps, fit, acc, trainedParams: trained, metrics };
  writeFileSync(outPath, JSON.stringify(results, null, 1));
  console.log(
    `${tag.padEnd(18)} acc=${(acc * 100).toFixed(1)}%  trained=${(trained / 1000).toFixed(0)}k ` +
    `(${((Date.now() - t0) / 60000).toFixed(1)}m)`
  );
}
console.log(`\nsaved to experiments/results/scaling-seed${SEED}.json`);
