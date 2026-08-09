// R-STDP sequence-learning probe: does reward-gated plasticity crack the
// sequence learning that immediate pair-STDP could not? (v5 finding:
// trained ≈ scrambled — order didn't matter.)
//
// Same motif-completion protocol as experiments/learning.mjs, three arms:
//   immediate  — classic STDP (v5 baseline)
//   rewarded   — R-STDP, teacher reward right after each motif repetition
//   rand-rew   — R-STDP, same number of rewards at decorrelated times
//                (controls for "reward exists" vs "reward is contingent")
//
// Sequence learning shows up as: rewarded Δlift > immediate AND > rand-rew.
//
// Run: npm run experiment:rstdp

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { SCALES } from '../js/ui/audio.js';
import { mulberry32 } from '../js/core/rng.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];
const TAU_MS = 100;
const NOTE_MS = 120;
const REPS = 50;
const MOTIF = [0, 2, 4, 6];
const CUE = MOTIF.slice(0, 2);
const CONTINUATION = new Set(MOTIF.slice(2));
const CUE_SET = new Set(CUE);

function makeLab(seed, mode) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: true, developmentEnabled: false },
    walk: { count: 0 },
  });
  lab.stdp.p.tauMs = TAU_MS;
  lab.stdp.p.mode = mode;
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE) };
}

const degToMidi = (lab, d) => 36 + lab.key.offset + 36 + scale[d];

function playSeq(lab, encoder, degrees) {
  for (const d of degrees) {
    encoder.noteOn(degToMidi(lab, d), 0.9);
    lab.runSteps(NOTE_MS);
  }
}

function probe(lab, encoder) {
  const was = lab.simParams.stdpEnabled;
  lab.simParams.stdpEnabled = false;
  const hist = new Array(scale.length).fill(0);
  const prev = lab.engine.onSpike;
  lab.engine.onSpike = (n) => {
    if (n.isOutput) hist[n.structDegree % scale.length]++;
  };
  playSeq(lab, encoder, CUE);
  lab.runSteps(1200);
  lab.engine.onSpike = prev;
  lab.simParams.stdpEnabled = was;
  let cont = 0;
  let nonCue = 0;
  hist.forEach((c, d) => {
    if (CUE_SET.has(d)) return;
    nonCue += c;
    if (CONTINUATION.has(d)) cont += c;
  });
  if (!nonCue) return 0;
  return cont / nonCue / (CONTINUATION.size / (scale.length - CUE_SET.size));
}

function condition(seed, mode, rewardTiming) {
  const { lab, encoder } = makeLab(seed, mode);
  const rng = mulberry32(seed ^ 0xfeed);
  const before = probe(lab, encoder);
  for (let r = 0; r < REPS; r++) {
    playSeq(lab, encoder, MOTIF);
    if (rewardTiming === 'contingent') {
      lab.reward(1); // teacher: "that phrase — yes"
      lab.runSteps(500);
    } else if (rewardTiming === 'random') {
      // same reward budget, decorrelated: reward lands mid-silence
      lab.runSteps(200 + Math.floor(rng() * 3000));
      lab.reward(1);
      lab.runSteps(300);
    } else {
      lab.runSteps(500);
    }
  }
  const after = probe(lab, encoder);
  return { before, after, delta: after - before };
}

const seeds = [11, 23, 42, 77, 99];
const agg = { imm: [], rew: [], rand: [] };
for (const seed of seeds) {
  const imm = condition(seed, 'immediate', 'none');
  const rew = condition(seed, 'reward', 'contingent');
  const rand = condition(seed, 'reward', 'random');
  agg.imm.push(imm);
  agg.rew.push(rew);
  agg.rand.push(rand);
  console.log(
    `seed ${String(seed).padStart(3)}  immediate Δ ${imm.delta.toFixed(3)}  rewarded Δ ${rew.delta.toFixed(3)} (${rew.before.toFixed(2)}→${rew.after.toFixed(2)})  rand-reward Δ ${rand.delta.toFixed(3)}`
  );
}
const m = (rows) => rows.reduce((a, r) => a + r.delta, 0) / rows.length;
console.log('\n== summary (mean Δ continuation-lift) ==');
console.log(`immediate STDP:     ${m(agg.imm).toFixed(3)}`);
console.log(`R-STDP contingent:  ${m(agg.rew).toFixed(3)}`);
console.log(`R-STDP random:      ${m(agg.rand).toFixed(3)}`);
console.log('\nsequence learning = contingent above BOTH immediate and random-reward');
