// Specialization control: does the organism learn *the player*, or does it
// just mirror anything played at it?
//
// Train on idiom A. Periodically freeze plasticity and probe with fixed
// phrases from idiom A and from unseen idiom B. Specialization index =
// relatedness(A-probe) − relatedness(B-probe). If it grows over the session
// (and doesn't with STDP off), the organism genuinely specialized toward
// the trained material rather than reflecting whatever it hears.
//
// Run: npm run experiment:specialization

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { degreeHist, cosine } from '../js/duet/dialogue.js';
import { SCALES } from '../js/ui/audio.js';
import { mulberry32 } from '../js/core/rng.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];
const IDIOM_A = [0, 2, 3, 5]; // trained material
const IDIOM_B = [1, 4, 6]; // unseen probe material
const PROBE_A = [0, 2, 3, 2, 0]; // fixed probe phrases
const PROBE_B = [1, 4, 6, 4, 1];
const TRAIN_EXCHANGES = 50;
const PROBE_EVERY = 10;
const NOTE_MS = 140;
const GAP_MS = 700;
const RESPONSE_MS = 2000;

function makeLab(seed, stdpOn) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: stdpOn, developmentEnabled: true },
    walk: { count: 3, variation: 0.5 },
  });
  lab.stdp.p.tauMs = 100;
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE) };
}

function exchange(lab, encoder, degrees, state) {
  state.callCounts.clear();
  state.inResponse = false;
  for (const d of degrees) {
    encoder.noteOn(36 + lab.key.offset + 36 + scale[d], 0.9);
    lab.runSteps(NOTE_MS);
  }
  lab.runSteps(GAP_MS);
  const top = [...state.callCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => id)
    .slice(0, 3);
  lab.walkers.seedAt(top);
  state.respDegrees = [];
  state.inResponse = true;
  lab.runSteps(RESPONSE_MS);
  state.inResponse = false;
  return state.respDegrees.length
    ? cosine(degreeHist(degrees), degreeHist(state.respDegrees))
    : 0;
}

function probeBlock(lab, encoder, state) {
  // plasticity + development frozen during probes
  const stdpWas = lab.simParams.stdpEnabled;
  const devWas = lab.simParams.developmentEnabled;
  lab.simParams.stdpEnabled = false;
  lab.simParams.developmentEnabled = false;
  const relA = exchange(lab, encoder, PROBE_A, state);
  const relB = exchange(lab, encoder, PROBE_B, state);
  lab.simParams.stdpEnabled = stdpWas;
  lab.simParams.developmentEnabled = devWas;
  return { relA, relB, index: relA - relB };
}

function session(seed, stdpOn) {
  const { lab, encoder } = makeLab(seed, stdpOn);
  const playerRng = mulberry32(seed ^ 0x5eed);
  const state = { callCounts: new Map(), respDegrees: [], inResponse: false };
  lab.engine.onSpike = (n) => {
    if (!state.inResponse && n.role === 'excitatory') {
      state.callCounts.set(n.id, (state.callCounts.get(n.id) ?? 0) + 1);
    }
  };
  lab.walkers.onNote = (n) => {
    if (state.inResponse) state.respDegrees.push(n.structDegree % scale.length);
  };

  const blocks = [probeBlock(lab, encoder, state)]; // pre-training baseline
  for (let x = 0; x < TRAIN_EXCHANGES; x++) {
    const len = 3 + Math.floor(playerRng() * 4);
    const call = Array.from({ length: len }, () => IDIOM_A[Math.floor(playerRng() * IDIOM_A.length)]);
    exchange(lab, encoder, call, state);
    if ((x + 1) % PROBE_EVERY === 0) blocks.push(probeBlock(lab, encoder, state));
  }
  return blocks;
}

const seeds = [7, 23, 42, 77];
const agg = { on: [], off: [] };
for (const seed of seeds) {
  const on = session(seed, true);
  const off = session(seed, false);
  agg.on.push(on);
  agg.off.push(off);
  const fmt = (blocks) => blocks.map((b) => b.index.toFixed(2)).join(' ');
  console.log(`seed ${String(seed).padStart(3)}  stdp-on  index: ${fmt(on)}`);
  console.log(`          stdp-off index: ${fmt(off)}`);
}

const meanAt = (rows, i, f) => rows.reduce((a, r) => a + f(r[i]), 0) / rows.length;
const nBlocks = agg.on[0].length;
console.log('\n== mean specialization index (relA − relB) per probe block ==');
console.log('block:        ' + Array.from({ length: nBlocks }, (_, i) => (i === 0 ? 'pre ' : `t${i}  `)).join(' '));
console.log('stdp-on:      ' + Array.from({ length: nBlocks }, (_, i) => meanAt(agg.on, i, (b) => b.index).toFixed(2)).join('  '));
console.log('stdp-off:     ' + Array.from({ length: nBlocks }, (_, i) => meanAt(agg.off, i, (b) => b.index).toFixed(2)).join('  '));
console.log('\nmean relA / relB final block:');
console.log(`stdp-on:  A ${meanAt(agg.on, nBlocks - 1, (b) => b.relA).toFixed(2)}  B ${meanAt(agg.on, nBlocks - 1, (b) => b.relB).toFixed(2)}`);
console.log(`stdp-off: A ${meanAt(agg.off, nBlocks - 1, (b) => b.relA).toFixed(2)}  B ${meanAt(agg.off, nBlocks - 1, (b) => b.relB).toFixed(2)}`);
console.log('\nlearning-the-player = index grows over blocks with stdp-on, flat with stdp-off');
