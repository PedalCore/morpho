// Style capture: train an organism on Beethoven (Ode to Joy theme, public
// domain) and test whether SIMPLE novel input then provokes responses closer
// to Beethoven's pitch distribution and rhythm than an untrained twin's.
//
// This is the "feed in a MIDI score, respond in the style" question in
// miniature — measurable, controlled, honest. Style here = degree
// distribution + IOI distribution, not sequence reproduction (which v5/v8
// showed this substrate does not do).
//
// Arms: trained (STDP+attention+development on the theme, 30 passes)
//       fresh twin (same genotype, never heard the theme)
// Probe: three 3-note cues NOT from the theme's opening, walkers answer.
// Metric: cosine(response, THEME) for degrees and rhythm.
//
// Run: npm run experiment:style

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { degreeHist, ioiHist, cosine } from '../js/duet/dialogue.js';
import { RegionalAttention } from '../js/attention/attention.js';
import { SCALES } from '../js/ui/audio.js';

const SCALE = 'major';
const scale = SCALES[SCALE];

// Ode to Joy, scale degrees in major (0=do): E E F G G F E D C C D E E D D …
const THEME = [
  { d: 2, ms: 400 }, { d: 2, ms: 400 }, { d: 3, ms: 400 }, { d: 4, ms: 400 },
  { d: 4, ms: 400 }, { d: 3, ms: 400 }, { d: 2, ms: 400 }, { d: 1, ms: 400 },
  { d: 0, ms: 400 }, { d: 0, ms: 400 }, { d: 1, ms: 400 }, { d: 2, ms: 400 },
  { d: 2, ms: 600 }, { d: 1, ms: 200 }, { d: 1, ms: 800 },
  { d: 2, ms: 400 }, { d: 2, ms: 400 }, { d: 3, ms: 400 }, { d: 4, ms: 400 },
  { d: 4, ms: 400 }, { d: 3, ms: 400 }, { d: 2, ms: 400 }, { d: 1, ms: 400 },
  { d: 0, ms: 400 }, { d: 0, ms: 400 }, { d: 1, ms: 400 }, { d: 2, ms: 400 },
  { d: 1, ms: 600 }, { d: 0, ms: 200 }, { d: 0, ms: 800 },
];
const THEME_DEG_HIST = degreeHist(THEME.map((n) => n.d));
const themeTimes = [];
THEME.reduce((t, n) => (themeTimes.push(t), t + n.ms), 0);
const THEME_IOI_HIST = ioiHist(themeTimes);

const TRAIN_PASSES = 30;
const PROBES = [
  [0, 2, 4], // simple triad walk (in theme material)
  [5, 4, 2], // starts OUTSIDE the theme's degree set
  [1, 3, 1], // partly outside
];

function makeLab(seed) {
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
      temporalMix: true,
    })
  );
  return { lab, encoder: new SpikeEncoder(lab, inputs, SCALE) };
}

function attachCollectors(lab, state) {
  lab.engine.onSpike = (n) => {
    if (!state.inResponse && n.role === 'excitatory') {
      state.callCounts.set(n.id, (state.callCounts.get(n.id) ?? 0) + 1);
    }
  };
  lab.walkers.onNote = (n) => {
    if (state.inResponse) {
      state.respDegrees.push(n.structDegree % scale.length);
      state.respTimes.push(lab.engine.stepCount);
    }
  };
}

function probeStyle(lab, encoder, state) {
  const stdpWas = lab.simParams.stdpEnabled;
  const devWas = lab.simParams.developmentEnabled;
  lab.simParams.stdpEnabled = false;
  lab.simParams.developmentEnabled = false;
  let deg = 0;
  let rhy = 0;
  for (const cue of PROBES) {
    state.callCounts.clear();
    state.inResponse = false;
    for (const d of cue) {
      encoder.noteOn(36 + lab.key.offset + 36 + scale[d], 0.9);
      lab.runSteps(300);
    }
    lab.runSteps(700);
    const top = [...state.callCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([id]) => id)
      .slice(0, 3);
    lab.walkers.seedAt(top);
    lab.walkers.setPhrase(300);
    state.respDegrees = [];
    state.respTimes = [];
    state.inResponse = true;
    lab.runSteps(3000);
    state.inResponse = false;
    deg += state.respDegrees.length
      ? cosine(degreeHist(state.respDegrees), THEME_DEG_HIST)
      : 0;
    rhy += state.respTimes.length > 2 ? cosine(ioiHist(state.respTimes), THEME_IOI_HIST) : 0;
  }
  lab.simParams.stdpEnabled = stdpWas;
  lab.simParams.developmentEnabled = devWas;
  return { deg: deg / PROBES.length, rhy: rhy / PROBES.length };
}

function run(seed) {
  const state = { callCounts: new Map(), respDegrees: [], respTimes: [], inResponse: false };
  // trained organism
  const { lab, encoder } = makeLab(seed);
  attachCollectors(lab, state);
  for (let pass = 0; pass < TRAIN_PASSES; pass++) {
    for (const n of THEME) {
      encoder.noteOn(36 + lab.key.offset + 36 + scale[n.d], 0.85);
      lab.runSteps(n.ms);
    }
    lab.runSteps(1200); // breath between passes
  }
  const trained = probeStyle(lab, encoder, state);
  // untrained twin
  const twin = makeLab(seed);
  attachCollectors(twin.lab, state);
  const fresh = probeStyle(twin.lab, twin.encoder, state);
  return { trained, fresh };
}

const seeds = [7, 23, 42, 77];
let sums = { td: 0, tr: 0, fd: 0, fr: 0 };
for (const seed of seeds) {
  const r = run(seed);
  sums.td += r.trained.deg;
  sums.tr += r.trained.rhy;
  sums.fd += r.fresh.deg;
  sums.fr += r.fresh.rhy;
  console.log(
    `seed ${String(seed).padStart(3)}  trained: style-deg ${r.trained.deg.toFixed(2)}  style-rhythm ${r.trained.rhy.toFixed(2)}   fresh twin: ${r.fresh.deg.toFixed(2)} / ${r.fresh.rhy.toFixed(2)}`
  );
}
const n = seeds.length;
console.log('\n== summary (mean, response similarity to the Beethoven theme) ==');
console.log(`trained:    degrees ${(sums.td / n).toFixed(3)}   rhythm ${(sums.tr / n).toFixed(3)}`);
console.log(`fresh twin: degrees ${(sums.fd / n).toFixed(3)}   rhythm ${(sums.fr / n).toFixed(3)}`);
console.log('\nstyle capture = trained above fresh on the SAME cues');
