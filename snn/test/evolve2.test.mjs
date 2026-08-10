// v14 structural evolution: layer count is now genomic — development and
// evaluation must be valid at every legal depth.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  GENES, decode, encodeV12, encodePhys, developBrain, evaluate,
} from '../experiments/evolve2.mjs';
import { mulberry32 } from '../js/core/rng.js';

test('v12 genome decodes with n_layers=4 and v13 phys re-encodes', () => {
  const phys = decode(encodeV12());
  assert.equal(phys.n_layers, 4);
  assert.equal(phys.ff_fan, 14);
  // a v13-style phys object (no n_layers) defaults to 4 and clamps to bounds
  const re = decode(encodePhys({ inhib_frac: 0.35, ff_fan: 2, skip_fan: 7 }));
  assert.equal(re.n_layers, 4);
  assert.ok(Math.abs(re.inhib_frac - 0.35) < 1e-9);
  assert.equal(re.skip_fan, 7);
});

test('development is valid at every legal layer count', () => {
  for (let L = 2; L <= 6; L++) {
    const norm = encodeV12();
    norm[0] = (L - GENES[0].lo) / (GENES[0].hi - GENES[0].lo);
    const b = developBrain(norm, 600, 7);
    assert.equal(b.L, L);
    assert.equal(b.synStart[600], b.M);
    for (let s = 0; s < b.M; s++) {
      assert.ok(b.synTgt[s] >= 0 && b.synTgt[s] < 600);
    }
    assert.ok(b.M / 600 > 5 && b.M / 600 < 45);
  }
});

test('evaluation runs end-to-end at shallow and deep genomes', () => {
  const V = 8;
  const rng = mulberry32(123);
  const ids = new Int32Array(3000);
  for (let i = 0; i < ids.length; i++) ids[i] = Math.floor(rng() * V);
  for (const L of [2, 6]) {
    const norm = encodeV12();
    norm[0] = (L - GENES[0].lo) / (GENES[0].hi - GENES[0].lo);
    const { acc, metrics } = evaluate(norm, 400, 42, ids, V, {
      taps: 32, calibChars: 200, fitChars: 500, testChars: 200, startPos: 0,
    });
    assert.ok(acc >= 0 && acc <= 1);
    assert.equal(metrics.layers, L);
    assert.ok(metrics.taps > 0);
  }
});
