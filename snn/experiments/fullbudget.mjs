// v13 addendum: the ladder attempt. v13 evaluated genomes under a small
// frozen readout budget (256 taps, 8k fit) — good for fair comparisons,
// not for absolute numbers. Here the best small-scale-selected genomes are
// instantiated at 120k and read out under the FULL v12 budget (1024 taps,
// 40k fit, 3k test, 5k calibration, cur/prev/prev2 context) so the result
// is ladder-comparable to v12's 39.1%. The v12 hand genome runs through
// the identical pipeline as the control on the same corpus windows.
//
// Run: npm run experiment:fullbudget [seed]

import { readFileSync } from 'node:fs';
import { evaluate, encodeV12 } from './evolve.mjs';

const SEED = Number(process.argv[2] ?? 42);
const N = 120000;
const CFG = { taps: 1024, calibChars: 5000, fitChars: 40000, testChars: 3000 };

const { readFileSync: rf, writeFileSync, existsSync, mkdirSync } = await import('node:fs');
const DATA_URL =
  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
const DATA_PATH = new URL('./data/tinyshakespeare.txt', import.meta.url).pathname;
if (!existsSync(DATA_PATH)) {
  mkdirSync(new URL('./data/', import.meta.url).pathname, { recursive: true });
  const res = await fetch(DATA_URL);
  writeFileSync(DATA_PATH, await res.text());
}
const text = rf(DATA_PATH, 'utf8');
const chars = [...new Set(text)].sort();
const V = chars.length;
const ids = new Int32Array(text.length);
for (let i = 0; i < text.length; i++) ids[i] = chars.indexOf(text[i]);

const ck = (s) =>
  JSON.parse(readFileSync(new URL(`./results/evolve-seed${s}.json`, import.meta.url), 'utf8'));
const arms = [
  ['v12-hand (control)', encodeV12()],
  ['gen0-best seed42', ck(42).history[0].bestGenome],
  ['evolved-best seed42', ck(42).history.at(-1).bestGenome],
  ['gen0-best seed99', ck(99).history[0].bestGenome],
];

console.log(
  `full-budget ladder eval · N=${N} · taps ${CFG.taps} · fit ${CFG.fitChars} · seed ${SEED}\n` +
  `reference: v12 published best 39.1% (same budget class) · bigram 28.8% · transformer ≈58%\n`
);
const t0 = Date.now();
const startPos = 0; // same corpus region autoreg used
const out = {};
for (const [name, genome] of arms) {
  const { acc, metrics } = evaluate(genome, N, SEED ^ (N * 2654435761), ids, V, {
    ...CFG, startPos,
  });
  out[name] = { acc, metrics };
  console.log(
    `${name.padEnd(22)} acc=${(acc * 100).toFixed(1)}%  syn/n=${metrics.synPerNeuron} ` +
    `spikes/char=${metrics.spikesPerChar} dead=${(metrics.deadFrac * 100).toFixed(0)}% ` +
    `(${((Date.now() - t0) / 60000).toFixed(1)}m)`
  );
}
writeFileSync(
  new URL(`./results/fullbudget-seed${SEED}.json`, import.meta.url),
  JSON.stringify({ SEED, N, CFG, startPos, out }, null, 1)
);
console.log(`\nsaved to experiments/results/fullbudget-seed${SEED}.json`);
