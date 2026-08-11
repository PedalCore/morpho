// v17 memory-selected evolution: task construction and the memory
// evaluation must be correct — a shifted-target bug would fake "memory".

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  TASKS, makeTaskData, skillScore, evaluateMemory,
} from '../experiments/evolve3.mjs';
import { encodeV12 } from '../experiments/evolve2.mjs';

test('recall targets are exact shifted copies; parity is windowed XOR', () => {
  const recall = TASKS.find((t) => t.name === 'recall4');
  const rd = makeTaskData(recall, 500, 9);
  for (let i = 4; i < 500; i++) assert.equal(rd.target[i], rd.stream[i - 4]);
  assert.equal(rd.chance, 1 / 8);
  const parity = TASKS.find((t) => t.name === 'parity3');
  const pd = makeTaskData(parity, 500, 9);
  for (let i = 2; i < 500; i++) {
    assert.equal(pd.target[i], pd.stream[i] ^ pd.stream[i - 1] ^ pd.stream[i - 2]);
  }
  assert.equal(pd.chance, 0.5);
});

test('skill score normalizes chance to zero', () => {
  assert.equal(skillScore(0.125, 0.125), 0);
  assert.equal(skillScore(1, 0.125), 1);
  assert.equal(skillScore(0.1, 0.125), 0); // below chance clamps to 0
  assert.ok(Math.abs(skillScore(0.75, 0.5) - 0.5) < 1e-9);
});

test('memory evaluation runs end-to-end and reports sane accuracy', () => {
  const r = evaluateMemory(encodeV12(), 400, 42, TASKS[0], {
    taps: 32, calibChars: 150, fitChars: 400, testChars: 150,
  });
  assert.ok(r.acc >= 0 && r.acc <= 1);
  assert.ok(r.skill >= 0 && r.skill <= 1);
  assert.ok(r.synPerNeuron > 0);
});
