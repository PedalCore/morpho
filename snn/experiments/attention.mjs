// Attention ablation: does MA-SNN-style regional attention give the duet
// sparser, more relevant answers?  (arXiv:2209.13929's claim — sparser
// spiking AND better performance — translated to dialogue.)
//
// Same protocol as experiments/dialogue.mjs (idiom player, walker answers),
// two conditions per seed: attention OFF (strength 0) vs ON (0.7). Both run
// STDP + development. Measured per condition:
//   - relatedness (early → late, drift)
//   - notes per answer
//   - network spikes per answer  (energy accounting, SGNNBench-style)
//
// Benefit = equal-or-better relatedness at fewer spikes, and/or higher
// relatedness outright.
//
// Run: npm run experiment:attention

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { degreeHist, cosine } from '../js/duet/dialogue.js';
import { RegionalAttention } from '../js/attention/attention.js';
import { SCALES } from '../js/ui/audio.js';
import { mulberry32 } from '../js/core/rng.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];
const IDIOM = [0, 2, 3, 5];
const EXCHANGES = 60;
const NOTE_MS = 140;
const GAP_MS = 700;
const RESPONSE_MS = 2000;

function makeLab(seed, attnStrength, bias = 'balanced') {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true, developmentEnabled: true },
    walk: { count: 3, variation: 0.5 },
  });
  lab.stdp.p.tauMs = 100;
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  if (attnStrength > 0) {
    lab.attachAttention(new RegionalAttention(lab.graph, scale.length, { strength: attnStrength, bias }));
  }
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE) };
}

function session(seed, attnStrength, bias = 'balanced') {
  const { lab, encoder } = makeLab(seed, attnStrength, bias);
  const playerRng = mulberry32(seed ^ 0x5eed);
  const scores = [];
  let respDegrees = [];
  let exchangeSpikes = 0; // all non-input spikes across the whole exchange
  let inResponse = false;
  const callCounts = new Map();
  lab.engine.onSpike = (n) => {
    if (n.role !== 'input') exchangeSpikes++;
    if (!inResponse && n.role === 'excitatory') {
      callCounts.set(n.id, (callCounts.get(n.id) ?? 0) + 1);
    }
  };
  lab.walkers.onNote = (n) => {
    if (inResponse) respDegrees.push(n.structDegree % scale.length);
  };
  for (let x = 0; x < EXCHANGES; x++) {
    const call = [];
    callCounts.clear();
    exchangeSpikes = 0;
    inResponse = false;
    const len = 3 + Math.floor(playerRng() * 4);
    for (let i = 0; i < len; i++) {
      const d = IDIOM[Math.floor(playerRng() * IDIOM.length)];
      call.push(d);
      encoder.noteOn(36 + lab.key.offset + 36 + scale[d], 0.9);
      lab.runSteps(NOTE_MS);
    }
    lab.runSteps(GAP_MS);
    const top = [...callCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id]) => id)
      .slice(0, 3);
    lab.walkers.seedAt(top);
    respDegrees = [];
    inResponse = true;
    lab.runSteps(RESPONSE_MS);
    inResponse = false;
    scores.push({
      score: respDegrees.length ? cosine(degreeHist(call), degreeHist(respDegrees)) : 0,
      respNotes: respDegrees.length,
      respSpikes: exchangeSpikes,
    });
  }
  const mean = (arr, f) => (arr.length ? arr.reduce((a, s) => a + f(s), 0) / arr.length : 0);
  return {
    early: mean(scores.slice(0, 15), (s) => s.score),
    late: mean(scores.slice(-15), (s) => s.score),
    overall: mean(scores, (s) => s.score),
    notes: mean(scores, (s) => s.respNotes),
    spikes: mean(scores, (s) => s.respSpikes),
  };
}

const seeds = [7, 23, 42, 77];
const agg = { off: [], on: [], sup: [] };
for (const seed of seeds) {
  const off = session(seed, 0);
  const on = session(seed, 0.7);
  const sup = session(seed, 0.7, 'suppress');
  agg.off.push(off);
  agg.on.push(on);
  agg.sup.push(sup);
  console.log(
    `seed ${String(seed).padStart(3)}  attn-off  rel ${off.overall.toFixed(2)} (${off.early.toFixed(2)}→${off.late.toFixed(2)})  ${off.notes.toFixed(1)} notes  ${off.spikes.toFixed(0)} spikes/exchange`
  );
  console.log(
    `          attn-on   rel ${on.overall.toFixed(2)} (${on.early.toFixed(2)}→${on.late.toFixed(2)})  ${on.notes.toFixed(1)} notes  ${on.spikes.toFixed(0)} spikes/exchange`
  );
  console.log(
    `          suppress  rel ${sup.overall.toFixed(2)} (${sup.early.toFixed(2)}→${sup.late.toFixed(2)})  ${sup.notes.toFixed(1)} notes  ${sup.spikes.toFixed(0)} spikes/exchange`
  );
}
const m = (rows, f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
console.log('\n== summary (mean over seeds) ==');
console.log(
  `attn-off: relatedness ${m(agg.off, (r) => r.overall).toFixed(3)}  drift ${(m(agg.off, (r) => r.late) - m(agg.off, (r) => r.early)).toFixed(3)}  spikes/exchange ${m(agg.off, (r) => r.spikes).toFixed(0)}`
);
console.log(
  `attn-on:  relatedness ${m(agg.on, (r) => r.overall).toFixed(3)}  drift ${(m(agg.on, (r) => r.late) - m(agg.on, (r) => r.early)).toFixed(3)}  spikes/exchange ${m(agg.on, (r) => r.spikes).toFixed(0)}`
);
console.log(
  `suppress: relatedness ${m(agg.sup, (r) => r.overall).toFixed(3)}  drift ${(m(agg.sup, (r) => r.late) - m(agg.sup, (r) => r.early)).toFixed(3)}  spikes/exchange ${m(agg.sup, (r) => r.spikes).toFixed(0)}`
);
console.log('\nMA-SNN signature = attn-on holds/raises relatedness at fewer spikes per answer');
