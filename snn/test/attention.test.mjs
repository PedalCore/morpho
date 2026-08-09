import test from 'node:test';
import assert from 'node:assert/strict';

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { RegionalAttention } from '../js/attention/attention.js';
import { SCALES } from '../js/ui/audio.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];

function attnLab(seed, strength = 0.6, bias = 'balanced') {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true, developmentEnabled: false },
    walk: { count: 0 },
  });
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  const attn = new RegionalAttention(lab.graph, scale.length, { strength, bias });
  lab.attachAttention(attn);
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE), attn };
}

test('suppress bias (default) never boosts: best region ≈ neutral, rest damped', () => {
  const { attn } = attnLab(7, 0.7, 'suppress');
  for (let i = 0; i < 6; i++) attn.noteHeard(0);
  attn.update();
  const gains = [...attn.gains.values()];
  assert.ok(gains.length > 1);
  assert.ok(Math.max(...gains) <= 1 + 1e-9, 'suppress must not amplify');
  assert.ok(Math.min(...gains) < 0.95, 'non-matching regions must be damped');
});

test('attention boosts regions matching heard material and damps the rest', () => {
  const { attn, lab } = attnLab(7);
  // "hear" degree 0 heavily
  for (let i = 0; i < 6; i++) attn.noteHeard(0);
  attn.update();
  assert.ok(attn.gains.size > 0, 'gains computed');
  // the region with the largest share of degree-0 anatomy must out-gain the
  // region with the smallest share
  let best = null;
  let worst = null;
  for (const region of lab.graph.leafRegions()) {
    let zero = 0;
    let total = 0;
    for (const id of region.members) {
      const n = lab.graph.neurons.get(id);
      if (n?.role !== 'excitatory') continue;
      total++;
      if (n.structDegree % scale.length === 0) zero++;
    }
    if (!total) continue;
    const share = zero / total;
    const g = attn.gains.get(region.path) ?? 1;
    if (!best || share > best.share) best = { share, g };
    if (!worst || share < worst.share) worst = { share, g };
  }
  assert.ok(best.share > worst.share, 'test needs regions with differing degree-0 share');
  assert.ok(
    best.g > worst.g,
    `best-matching region (share ${best.share.toFixed(2)}, gain ${best.g.toFixed(2)}) should out-gain worst (share ${worst.share.toFixed(2)}, gain ${worst.g.toFixed(2)})`
  );
  assert.ok(best.g > 1, `top region should be boosted above neutral, got ${best.g}`);
});

test('strength 0 and silence both yield neutral gains', () => {
  const { attn } = attnLab(7, 0);
  for (let i = 0; i < 6; i++) attn.noteHeard(0);
  attn.update();
  for (const g of attn.gains.values()) assert.equal(g, 1);
  const { attn: quiet } = attnLab(7, 0.8);
  quiet.update(); // nothing heard
  assert.equal(quiet.gains.size, 0, 'no gains without recent input');
});

test('modulation actually scales synaptic delivery', () => {
  const { lab } = attnLab(3, 0.8);
  const n = [...lab.graph.neurons.values()].find((x) => x.role === 'excitatory');
  // force a suppressive gain on this neuron's region
  lab.attention.gains.set(n.region, 0.5);
  const input = lab.graph.addNeuron({ role: 'input', region: 'IN' });
  lab.graph.addSynapse({ source: input.id, target: n.id, weight: 0.8, delaySteps: 1 });
  lab.engine.forceFire(input.id);
  lab.engine.step();
  lab.engine.step();
  assert.ok(n.membrane < 0.45, `delivery should be halved by gain 0.5, membrane=${n.membrane}`);
});

test('attention trickles survival energy into attended regions', () => {
  const { attn, lab } = attnLab(11, 0.8);
  const before = new Map([...lab.graph.neurons.values()].map((n) => [n.id, n.energy]));
  for (let i = 0; i < 8; i++) attn.noteHeard(2);
  attn.update();
  let boosted = 0;
  for (const n of lab.graph.neurons.values()) {
    if (n.energy > before.get(n.id)) boosted++;
  }
  assert.ok(boosted > 0, 'strongly attended regions should gain energy');
});

test('temporal mixing: fresh distinct material outweighs stale context; silence hands over to slow memory', () => {
  const { attn } = attnLab(7, 0.7, 'suppress');
  attn.params.temporalMix = true;
  // long-ago material: degree 5, decayed heavily
  for (let i = 0; i < 10; i++) attn.noteHeard(5);
  for (let i = 0; i < 12; i++) attn.update(); // ~3 s of silence
  // fresh distinct phrase: degree 0
  for (let i = 0; i < 4; i++) attn.noteHeard(0);
  attn.update();
  const ctxFresh = attn._context();
  assert.ok(
    ctxFresh[0] > ctxFresh[5],
    `fresh phrase should dominate the mixed context: ${ctxFresh.map((v) => v.toFixed(2))}`
  );
  // now fall silent: fast context decays away, slow memory keeps degree 0+5
  for (let i = 0; i < 10; i++) attn.update();
  const ctxQuiet = attn._context();
  assert.ok(
    ctxQuiet[0] > 0.01,
    'slow timescale should retain session material through silence'
  );
});

test('attention runs deterministically inside the sim loop', () => {
  const run = () => {
    const { lab, encoder } = attnLab(42, 0.6);
    const log = [];
    for (let i = 0; i < 4000; i++) {
      if (i % 400 === 0) encoder.noteOn(60, 0.9);
      const spikes = lab.step();
      if (spikes.length) log.push(`${i}:${spikes.join(',')}`);
    }
    return log.join('|');
  };
  assert.equal(run(), run());
});
