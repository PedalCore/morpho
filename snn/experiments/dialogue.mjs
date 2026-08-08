// Dialogue experiment: simulate a call-and-response session and measure
// whether the organism's answers become MORE related to the player's
// material over time — the actual question behind "is it learning?"
//
// A simulated player with a fixed idiom (a biased subset of degrees) plays
// short calls; after each call we collect the response window's output
// degrees and score relatedness (degree-histogram cosine, as in the UI).
// Compare early vs late exchanges, STDP on vs off, across seeds.
//
// Run: npm run experiment:dialogue

import { Lab } from '../js/sim/lab.js';
import { wireSensoryInputs, SpikeEncoder } from '../js/duet/sensory.js';
import { degreeHist, cosine } from '../js/duet/dialogue.js';
import { SCALES } from '../js/ui/audio.js';
import { mulberry32 } from '../js/core/rng.js';

const SCALE = 'dorian';
const scale = SCALES[SCALE];
const IDIOM = [0, 2, 3, 5]; // the player's material — answers should gravitate here
const EXCHANGES = 60;
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

// Answers come from walkers seeded on the anatomy the call activated —
// their traversal follows the (STDP-shaped) weights.
function session(seed, stdpOn) {
  const { lab, encoder } = makeLab(seed, stdpOn);
  const playerRng = mulberry32(seed ^ 0x5eed);
  const scores = [];
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
  for (let x = 0; x < EXCHANGES; x++) {
    const call = [];
    callCounts.clear();
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
    const resp = respDegrees.slice();
    scores.push({
      score: resp.length ? cosine(degreeHist(call), degreeHist(resp)) : 0,
      respNotes: resp.length,
    });
  }
  const meanOf = (arr) => (arr.length ? arr.reduce((a, s) => a + s.score, 0) / arr.length : 0);
  const answered = scores.filter((s) => s.respNotes > 0);
  return {
    early: meanOf(scores.slice(0, 15)),
    late: meanOf(scores.slice(-15)),
    answeredFrac: answered.length / scores.length,
    meanRespNotes: scores.reduce((a, s) => a + s.respNotes, 0) / scores.length,
  };
}

const seeds = [7, 23, 42, 77];
let sumOnDelta = 0;
let sumOffDelta = 0;
for (const seed of seeds) {
  const on = session(seed, true);
  const off = session(seed, false);
  sumOnDelta += on.late - on.early;
  sumOffDelta += off.late - off.early;
  console.log(
    `seed ${String(seed).padStart(3)}  stdp-on  early ${on.early.toFixed(2)} → late ${on.late.toFixed(2)}  (answered ${(on.answeredFrac * 100).toFixed(0)}%, ${on.meanRespNotes.toFixed(1)} notes/resp)`
  );
  console.log(
    `          stdp-off early ${off.early.toFixed(2)} → late ${off.late.toFixed(2)}  (answered ${(off.answeredFrac * 100).toFixed(0)}%, ${off.meanRespNotes.toFixed(1)} notes/resp)`
  );
}
console.log('\n== summary ==');
console.log(`mean relatedness drift (late − early)  stdp-on:  ${(sumOnDelta / seeds.length).toFixed(3)}`);
console.log(`mean relatedness drift (late − early)  stdp-off: ${(sumOffDelta / seeds.length).toFixed(3)}`);
console.log('\npositive drift with stdp-on above stdp-off = the dialogue is converging on the player');
