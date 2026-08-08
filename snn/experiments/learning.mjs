// Learning probe: can the duet brain learn a motif?
//
// Protocol (per seed):
//   1. baseline probe — play the CUE (first 2 motif notes), record which
//      output degrees fire in the response window (STDP frozen during probes)
//   2. training — play the full MOTIF R times (STDP active, development off
//      so we isolate synaptic learning)
//   3. probe again — same cue
//
// Controls:
//   - STDP-OFF twin: same genotype, same training, no plasticity
//   - SCRAMBLED twin: STDP on, trained on the motif's degrees in a different
//     order (same note statistics, different transitions)
//
// Metric: continuation lift = P(response spike ∈ continuation degrees,
// among non-cue spikes) ÷ chance. Learning the *sequence* (not just usage)
// shows up as: Δlift(trained) > Δlift(stdp-off) and > Δlift(scrambled).
//
// Run: npm run experiment:learning [-- tauMs noteMs reps]

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { SCALES } from '../js/ui/audio.js';

const SCALE = 'dorian'; // 7 degrees → motif can leave degrees unused
const scale = SCALES[SCALE];

const TAU_MS = Number(process.argv[2] ?? 100); // STDP trace (default longer than live 25ms)
const NOTE_MS = Number(process.argv[3] ?? 120); // inter-note interval while training
const REPS = Number(process.argv[4] ?? 50);

const MOTIF = [0, 2, 4, 6]; // cue [0,2] → continuation {4,6}; degrees 1,3,5 unused
const CUE = MOTIF.slice(0, 2);
const CONTINUATION = new Set(MOTIF.slice(2));
const CUE_SET = new Set(CUE);

function makeLab(seed, stdpOn) {
  const lab = new Lab({
    seed,
    grammar: { inputNeurons: 0, outputFraction: 0.5 },
    sim: { pulseFireProb: 0, backgroundHz: 0, stdpEnabled: stdpOn, developmentEnabled: false },
    walk: { count: 0 },
  });
  lab.stdp.p.tauMs = TAU_MS;
  const inputs = wireSensoryInputs(lab.graph, SCALE, lab.streams.build);
  lab.inputIds = inputs.map((n) => n.id);
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE) };
}

const degToMidi = (lab, d, oct = 3) => 36 + lab.key.offset + oct * 12 + scale[d];

function playSeq(lab, encoder, degrees, noteMs) {
  for (const d of degrees) {
    encoder.noteOn(degToMidi(lab, d), 0.9);
    lab.runSteps(noteMs);
  }
}

function probe(lab, encoder, windowMs = 1200) {
  const was = lab.simParams.stdpEnabled;
  lab.simParams.stdpEnabled = false; // probes must not train
  const hist = new Array(scale.length).fill(0);
  const prev = lab.engine.onSpike;
  lab.engine.onSpike = (n) => {
    if (n.isOutput) hist[n.structDegree % scale.length]++;
  };
  playSeq(lab, encoder, CUE, NOTE_MS);
  lab.runSteps(windowMs);
  lab.engine.onSpike = prev;
  lab.simParams.stdpEnabled = was;
  return hist;
}

function continuationLift(hist) {
  let cont = 0;
  let nonCue = 0;
  hist.forEach((c, d) => {
    if (CUE_SET.has(d)) return;
    nonCue += c;
    if (CONTINUATION.has(d)) cont += c;
  });
  if (!nonCue) return { lift: 0, nonCue };
  const chance = CONTINUATION.size / (scale.length - CUE_SET.size);
  return { lift: cont / nonCue / chance, nonCue };
}

function condition(seed, stdpOn, trainDegrees) {
  const { lab, encoder } = makeLab(seed, stdpOn);
  const before = continuationLift(probe(lab, encoder));
  for (let r = 0; r < REPS; r++) {
    playSeq(lab, encoder, trainDegrees, NOTE_MS);
    lab.runSteps(500); // settle between repetitions
  }
  const after = continuationLift(probe(lab, encoder));
  return { before, after, delta: after.lift - before.lift };
}

const SCRAMBLED = [4, 0, 6, 2]; // same degrees, different transitions

const seeds = [11, 23, 42, 77, 99];
const rows = [];
for (const seed of seeds) {
  const trained = condition(seed, true, MOTIF);
  const stdpOff = condition(seed, false, MOTIF);
  const scrambled = condition(seed, true, SCRAMBLED);
  rows.push({ seed, trained, stdpOff, scrambled });
  console.log(
    `seed ${String(seed).padStart(3)}  ` +
      `trained Δlift ${trained.delta.toFixed(3)} (${trained.before.lift.toFixed(2)}→${trained.after.lift.toFixed(2)}, n=${trained.after.nonCue})  ` +
      `stdp-off Δ ${stdpOff.delta.toFixed(3)}  scrambled Δ ${scrambled.delta.toFixed(3)}`
  );
}

const mean = (f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
console.log('\n== summary ==');
console.log(`params: tauMs=${TAU_MS} noteMs=${NOTE_MS} reps=${REPS} motif=${MOTIF} cue=${CUE}`);
console.log(`mean Δlift  trained:   ${mean((r) => r.trained.delta).toFixed(3)}`);
console.log(`mean Δlift  stdp-off:  ${mean((r) => r.stdpOff.delta).toFixed(3)}`);
console.log(`mean Δlift  scrambled: ${mean((r) => r.scrambled.delta).toFixed(3)}`);
console.log(
  '\nlearning signature = trained above BOTH controls (sequence learned, not just usage)'
);
