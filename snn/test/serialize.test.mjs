import test from 'node:test';
import assert from 'node:assert/strict';

import { Lab } from '../js/sim/lab.js';
import { serializeLab, deserializeLab } from '../js/sim/serialize.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { RegionalAttention } from '../js/attention/attention.js';

function spikeLog(lab, steps) {
  const log = [];
  for (let i = 0; i < steps; i++) {
    const spikes = lab.step();
    if (spikes.length) log.push(`${i}:${spikes.join(',')}`);
  }
  return log.join('|');
}

test('save/load round-trip: restored organism continues spike-for-spike identically', () => {
  // run a busy organism (development + walkers + key changes possible)
  const lab = new Lab({ seed: 42 });
  lab.runEpochs(7); // mid-life, with structural history and in-flight deliveries

  const snapshot = JSON.parse(JSON.stringify(serializeLab(lab)));
  const restored = deserializeLab(snapshot);

  // the original continues; the restored copy must match its every spike
  const original = spikeLog(lab, 3 * lab.simParams.epochSteps);
  const clone = spikeLog(restored, 3 * restored.simParams.epochSteps);
  assert.equal(clone, original, 'restored organism must continue identically');
  assert.deepEqual(restored.report(), lab.report());
});

test('save/load preserves duet machinery (sensory inputs, encoder, eligibility)', () => {
  const lab = new Lab({
    seed: 9,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true },
    walk: { count: 2 },
  });
  lab.stdp.p.mode = 'reward';
  const inputs = wireSensoryInputs(lab.graph, 'dorian', lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  const encoder = new SpikeEncoder(lab, inputs, 'dorian');
  for (let i = 0; i < 3000; i++) {
    if (i % 400 === 0) encoder.noteOn(60 + (i % 12), 0.9);
    lab.step();
  }
  assert.ok(lab.stdp.eligibility.size > 0, 'precondition: traces exist');

  const restored = deserializeLab(JSON.parse(JSON.stringify(serializeLab(lab))));
  assert.equal(restored.stdp.eligibility.size, lab.stdp.eligibility.size);
  assert.deepEqual(restored.inputIds, lab.inputIds);
  // rebuild an encoder on the restored inputs and keep playing — same result
  const restoredInputs = restored.inputIds.map((id) => restored.graph.neurons.get(id));
  const enc2 = new SpikeEncoder(restored, restoredInputs, 'dorian');
  enc2.noteOn(62, 0.9);
  encoder.noteOn(62, 0.9);
  const a = spikeLog(lab, 2000);
  const b = spikeLog(restored, 2000);
  assert.equal(b, a, 'played restored organism matches original');
});

test('save/load preserves attention state', () => {
  const lab = new Lab({
    seed: 5,
    grammar: { inputNeurons: 0 },
    sim: { pulseFireProb: 0, backgroundHz: 0 },
    walk: { count: 1 },
  });
  const inputs = wireSensoryInputs(lab.graph, 'dorian', lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  lab.attachAttention(new RegionalAttention(lab.graph, 7, { strength: 0.7 }));
  lab.attention.noteHeard(2);
  lab.attention.noteHeard(2);
  lab.runSteps(600);

  const restored = deserializeLab(JSON.parse(JSON.stringify(serializeLab(lab))), {
    AttentionClass: RegionalAttention,
  });
  assert.ok(restored.attention, 'attention restored');
  assert.deepEqual(restored.attention.inputHist, lab.attention.inputHist);
  assert.equal(spikeLog(restored, 1500), spikeLog(lab, 1500));
});

test('growth continues cleanly after restore (no id collisions)', () => {
  const lab = new Lab({ seed: 21 });
  lab.runEpochs(5);
  const restored = deserializeLab(JSON.parse(JSON.stringify(serializeLab(lab))));
  const before = restored.graph.neurons.size;
  const ch = restored.branchOut(2);
  assert.ok(ch.born.length > 0);
  for (const id of ch.born) {
    assert.ok(restored.graph.neurons.has(id));
  }
  assert.equal(
    new Set([...restored.graph.neurons.keys()]).size,
    restored.graph.neurons.size,
    'no duplicate neuron ids after restore + growth'
  );
  assert.ok(restored.graph.neurons.size > before);
  restored.runEpochs(2); // and the sim still runs
});
