// Attention-as-morphogen ablation: does attention-driven survival energy
// actually steer development toward the player's material?
//
// Both arms run suppress-attention (the winning config) + STDP + development
// for a long session on idiom A. The only difference: the energy trickle
// from attention into region survival is ON in one arm, OFF in the other.
//
// Structural metric: idiom coverage — the fraction of excitatory neurons
// whose birth pitch lies in the trained idiom. If attention is a morphogen,
// coverage should rise (attended anatomy survives + grows, ignored anatomy
// starves) in the trickle arm and not otherwise. Also reported: relatedness,
// grown/pruned counts.
//
// Run: npm run experiment:morphogen

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { degreeHist, cosine } from '../js/duet/dialogue.js';
import { RegionalAttention } from '../js/attention/attention.js';
import { SCALES } from '../js/ui/audio.js';
import { mulberry32 } from '../js/core/rng.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];
const IDIOM = [0, 2, 3, 5];
const IDIOM_SET = new Set(IDIOM);
const EXCHANGES = 120; // long session (~10 sim minutes)
const NOTE_MS = 140;
const GAP_MS = 700;
const RESPONSE_MS = 2000;

function idiomCoverage(lab) {
  let inIdiom = 0;
  let total = 0;
  for (const n of lab.graph.neurons.values()) {
    if (n.role !== 'excitatory') continue;
    total++;
    if (IDIOM_SET.has(n.structDegree % scale.length)) inIdiom++;
  }
  return total ? inIdiom / total : 0;
}

function session(seed, trickleOn) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true, developmentEnabled: true },
    walk: { count: 3, variation: 0.5 },
  });
  lab.stdp.p.tauMs = 100;
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  lab.attachAttention(
    new RegionalAttention(lab.graph, scale.length, {
      strength: 0.7,
      bias: 'suppress',
      energyTrickle: trickleOn ? 0.02 : 0,
    })
  );
  const encoder = new SpikeEncoder(lab, inputs, SCALE);
  const playerRng = mulberry32(seed ^ 0x5eed);

  const startCoverage = idiomCoverage(lab);
  let respDegrees = [];
  let inResponse = false;
  const callCounts = new Map();
  lab.engine.onSpike = (n) => {
    if (!inResponse && n.role === 'excitatory') {
      callCounts.set(n.id, (callCounts.get(n.id) ?? 0) + 1);
    }
  };
  lab.walkers.onNote = (n) => {
    if (inResponse) respDegrees.push(n.structDegree % scale.length);
  };

  const scores = [];
  for (let x = 0; x < EXCHANGES; x++) {
    callCounts.clear();
    inResponse = false;
    const len = 3 + Math.floor(playerRng() * 4);
    const call = Array.from({ length: len }, () => IDIOM[Math.floor(playerRng() * IDIOM.length)]);
    for (const d of call) {
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
    scores.push(respDegrees.length ? cosine(degreeHist(call), degreeHist(respDegrees)) : 0);
  }
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  return {
    startCoverage,
    endCoverage: idiomCoverage(lab),
    lateRel: mean(scores.slice(-30)),
    grown: lab.dev.grownTotal,
    pruned: lab.dev.prunedTotal,
    neurons: lab.graph.neurons.size,
  };
}

const seeds = [7, 23, 42, 77];
const agg = { on: [], off: [] };
for (const seed of seeds) {
  const on = session(seed, true);
  const off = session(seed, false);
  agg.on.push(on);
  agg.off.push(off);
  const f = (r) =>
    `coverage ${r.startCoverage.toFixed(2)}→${r.endCoverage.toFixed(2)}  rel(late) ${r.lateRel.toFixed(2)}  grown/pruned ${r.grown}/${r.pruned}  n=${r.neurons}`;
  console.log(`seed ${String(seed).padStart(3)}  trickle-on   ${f(on)}`);
  console.log(`          trickle-off  ${f(off)}`);
}
const m = (rows, f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
console.log('\n== summary (mean over seeds) ==');
console.log(
  `trickle-on:  Δcoverage ${(m(agg.on, (r) => r.endCoverage) - m(agg.on, (r) => r.startCoverage)).toFixed(3)}  late relatedness ${m(agg.on, (r) => r.lateRel).toFixed(3)}`
);
console.log(
  `trickle-off: Δcoverage ${(m(agg.off, (r) => r.endCoverage) - m(agg.off, (r) => r.startCoverage)).toFixed(3)}  late relatedness ${m(agg.off, (r) => r.lateRel).toFixed(3)}`
);
console.log('\nmorphogen signature = coverage rises (anatomy reshapes toward the idiom) only with the trickle on');
