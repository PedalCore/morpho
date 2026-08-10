// v15 generation benchmark: the n-gram scorer is the measuring instrument —
// it must behave sanely before any generation numbers mean anything.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildNgram, bpcOf } from '../experiments/genbench.mjs';
import { mulberry32 } from '../js/core/rng.js';

test('n-gram scorer: structured text scores far below noise', () => {
  const V = 6;
  const rng = mulberry32(5);
  // build a strongly patterned sequence: repeated motif with occasional noise
  const motif = [0, 1, 2, 3, 4, 5];
  const ids = new Int32Array(30000);
  for (let i = 0; i < ids.length; i++) {
    ids[i] = rng() < 0.05 ? Math.floor(rng() * V) : motif[i % V];
  }
  const prob = buildNgram(ids, V, 25000);
  // held-out continuation of the same pattern
  const held = [];
  for (let i = 25000; i < 26500; i++) held.push(ids[i]);
  const noise = Array.from({ length: 1500 }, () => Math.floor(rng() * V));
  const heldBpc = bpcOf(held, prob);
  const noiseBpc = bpcOf(noise, prob);
  assert.ok(heldBpc < noiseBpc - 0.5, `held ${heldBpc} should beat noise ${noiseBpc}`);
  assert.ok(heldBpc >= 0 && Number.isFinite(heldBpc));
});

test('n-gram probabilities are valid', () => {
  const V = 4;
  const rng = mulberry32(9);
  const ids = new Int32Array(5000);
  for (let i = 0; i < ids.length; i++) ids[i] = Math.floor(rng() * V);
  const prob = buildNgram(ids, V, 5000);
  const hist = [0, 1, 2];
  let z = 0;
  for (let c = 0; c < V; c++) {
    const p = prob(hist, c);
    assert.ok(p > 0 && p <= 1);
    z += p;
  }
  assert.ok(Math.abs(z - 1) < 0.05, `sums to ${z}`);
});
