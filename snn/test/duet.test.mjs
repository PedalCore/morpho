import test from 'node:test';
import assert from 'node:assert/strict';

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { SCALES } from '../js/ui/audio.js';

const SCALE = 'minor pentatonic';

function duetLab(seed = 42) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true },
    walk: { count: 0 },
  });
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  const encoder = new SpikeEncoder(lab, inputs, SCALE);
  return { lab, inputs, encoder };
}

test('sensory inputs: one per scale degree, wired to matching pitch classes', () => {
  const { lab, inputs } = duetLab();
  const scale = SCALES[SCALE];
  assert.equal(inputs.length, scale.length);
  for (const input of inputs) {
    const outs = lab.graph.outgoing.get(input.id);
    assert.ok(outs.length > 0, `sensory input ${input.degreeClass} has no targets`);
    for (const s of outs) {
      const t = lab.graph.neurons.get(s.target);
      assert.equal(
        t.structDegree % scale.length,
        input.degreeClass,
        'sensory wiring must be tonotopic'
      );
    }
  }
});

test('encoder snaps off-scale notes to the nearest degree and bursts with velocity', () => {
  const { lab, encoder } = duetLab();
  // C minor pentatonic = [0,3,5,7,10]; D (pc 2) should snap to Eb (idx 1) or C (idx 0)
  const r = encoder.noteOn(62, 0.9); // D4, loud
  assert.ok(r.degree === 0 || r.degree === 1, `snapped to ${r.degree}`);
  assert.equal(lab.pendingInputFires.length, 3, 'loud note → 3-spike burst');
  const quiet = encoder.noteOn(60, 0.2); // C, quiet
  assert.equal(quiet.degree, 0);
  assert.equal(lab.pendingInputFires.length, 3 + 1, 'quiet note → single spike');
});

test('played notes drive the network: input spikes propagate to output spikes', () => {
  const { lab, encoder } = duetLab(7);
  // silence first: no drive at all
  lab.runEpochs(1);
  const silentRate = lab.activity.networkRateHz;
  // now "play" a repeated phrase for two epochs
  let outputSpikes = 0;
  lab.engine.onSpike = (n) => {
    if (n.isOutput) outputSpikes++;
  };
  for (let e = 0; e < 2 * lab.simParams.epochSteps; e++) {
    if (e % 250 === 0) encoder.noteOn(60 + [0, 3, 7][Math.floor(e / 250) % 3], 0.9);
    lab.step();
  }
  assert.ok(outputSpikes > 5, `playing should provoke responses, got ${outputSpikes}`);
  assert.ok(lab.activity.networkRateHz > silentRate, 'playing raises network activity');
});

test('fireInput delays schedule future spikes deterministically', () => {
  const { lab, inputs } = duetLab();
  lab.fireInput(inputs[0].id, 0);
  lab.fireInput(inputs[0].id, 50);
  const spikesAt = [];
  for (let i = 0; i < 80; i++) {
    const s = lab.step();
    if (s.includes(inputs[0].id)) spikesAt.push(i);
  }
  assert.deepEqual(spikesAt, [0, 50]);
});

test('repeated playing under STDP moves weights in the played network', () => {
  const { lab, encoder } = duetLab(11);
  for (let e = 0; e < 4 * lab.simParams.epochSteps; e++) {
    if (e % 300 === 0) encoder.noteOn(60, 0.9); // hammer degree 0
    lab.step();
  }
  // compare against an untouched twin of the same genotype
  const fresh = duetLab(11);
  const freshWeights = new Map([...fresh.lab.graph.synapses.values()].map((s) => [s.id, s.weight]));
  let changed = 0;
  for (const s of lab.graph.synapses.values()) {
    const w0 = freshWeights.get(s.id);
    if (w0 !== undefined && s.weight > 0 && Math.abs(s.weight - w0) > 1e-6) changed++;
  }
  assert.ok(changed > 5, `expected STDP to move weights from playing, ${changed} changed`);
});
