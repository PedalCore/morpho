// v13 evolution experiment: the genome→brain development and the evaluation
// pipeline must be valid, deterministic, and size-agnostic (same genome,
// any N). Uses synthetic text so tests never touch the network.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  GENES, decode, encodeV12, developBrain, evaluate, fitnessOf,
} from '../experiments/evolve.mjs';
import { mulberry32 } from '../js/core/rng.js';

test('v12 genome round-trips through encode/decode', () => {
  const phys = decode(encodeV12());
  assert.equal(phys.ff_fan, 14);
  assert.equal(phys.inh_fan, 22);
  assert.ok(Math.abs(phys.inhib_frac - 0.15) < 1e-9);
  assert.ok(Math.abs(phys.rec_gain - 0.45) < 1e-9);
});

test('decode clamps out-of-range genes to legal bounds', () => {
  const phys = decode(GENES.map(() => 99));
  for (const { key, hi } of GENES) assert.ok(phys[key] <= hi);
});

test('development is deterministic and CSR-consistent at any size', () => {
  const g = encodeV12();
  for (const N of [400, 1200]) {
    const a = developBrain(g, N, 7);
    const b = developBrain(g, N, 7);
    assert.equal(a.M, b.M);
    assert.deepEqual([...a.synW.slice(0, 50)], [...b.synW.slice(0, 50)]);
    assert.equal(a.synStart[N], a.M);
    for (let s = 0; s < a.M; s++) {
      assert.ok(a.synTgt[s] >= 0 && a.synTgt[s] < N);
      assert.ok(a.synDelay[s] >= 1 && a.synDelay[s] < 16);
    }
    // genome length is constant but phenotype scales: syn/neuron ~constant
    assert.ok(a.M / N > 10 && a.M / N < 40);
  }
});

test('evaluation runs end-to-end on synthetic text and reports metrics', () => {
  const V = 8;
  const rng = mulberry32(123);
  const ids = new Int32Array(4000);
  for (let i = 0; i < ids.length; i++) ids[i] = Math.floor(rng() * V);
  const { acc, metrics } = evaluate(encodeV12(), 400, 42, ids, V, {
    taps: 32, calibChars: 200, fitChars: 600, testChars: 200, startPos: 0,
  });
  assert.ok(acc >= 0 && acc <= 1);
  assert.equal(metrics.neurons, 400);
  assert.ok(metrics.synapses > 0);
  assert.ok(metrics.taps > 0 && metrics.taps <= 32);
  assert.ok(metrics.deadFrac >= 0 && metrics.deadFrac <= 1);
});

test('fitness penalizes scale variance and connectivity', () => {
  const flat = fitnessOf([0.3, 0.3, 0.3], 25, 0.5, 0.02);
  const spiky = fitnessOf([0.4, 0.3, 0.2], 25, 0.5, 0.02);
  assert.ok(flat > spiky, 'same mean, lower variance must win');
  const lean = fitnessOf([0.3, 0.3, 0.3], 20, 0.5, 0.02);
  assert.ok(lean > flat, 'same accuracy, fewer synapses must win');
});
