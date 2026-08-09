import { Lab, KEY_NAMES } from './sim/lab.js';
import { Renderer, drawRateSparkline } from './ui/renderer.js';
import { AudioEngine, SCALES, TUNINGS } from './ui/audio.js';
import { FifthsWheel } from './ui/fifths.js';
import { wireSensoryInputs, SpikeEncoder, NOTE_NAMES } from './duet/sensory.js';
import { DialogueTracker } from './duet/dialogue.js';
import { MidiIO } from './io/midi.js';
import { RegionalAttention } from './attention/attention.js';
import { serializeLab, deserializeLab } from './sim/serialize.js';
import { parseSMF } from './io/smf.js';

const $ = (id) => document.getElementById(id);
// duet mode: no metronome drive — the human plays the organism via MIDI/pads
// attention mode: duet + MA-SNN-style regional attention (attention.html)
const MODE = document.body.dataset.mode ?? 'lab';
const ATTN = MODE === 'attention';
const DUET = MODE === 'duet' || ATTN;

const audio = new AudioEngine();
const midiIO = new MidiIO();
let lab = null;
let renderer = null;
let fifthsWheel = null;
let encoder = null;
let sensoryInputs = [];
let dialogue = null;
let callCounts = new Map(); // neuronId → spikes during the current human call
// MIDI-file training: notes fed into the sensory layer on sim time
let trainNotes = null; // [{tMs, note, vel}]
let trainIdx = 0;
let trainStartStep = 0;
let running = false;
let speed = 1;
let epochMarks = []; // {i, born, pruned} aligned to rateHistory indices
let lastTime = 0;
let accum = 0;

function build(seed) {
  const fresh = new Lab({
    seed,
    sim: {
      pulsePeriodMs: parseInt($('tempo').value, 10),
      developmentEnabled: $('devToggle').checked,
      modProb: parseFloat($('keyDrift').value),
      drivePattern: $('drive').value,
      stdpEnabled: $('stdp').checked,
      // duet: the human is the drive — no metronome pulses, just a whisper
      // of background so the organism "dreams" its vocabulary when idle
      ...(DUET ? { pulseFireProb: 0, backgroundHz: 0.25 } : {}),
    },
    grammar: DUET ? { inputNeurons: 0, outputFraction: 0.5 } : {},
    walk: {
      count: parseInt($('walkers').value, 10),
      variation: parseFloat($('variation').value),
      momentum: parseFloat($('momentum').value),
      stepDivisor: parseInt($('walkRate').value, 10),
      rubato: parseFloat($('rubato').value),
    },
  });
  if (DUET) {
    const inputs = wireSensoryInputs(fresh.graph, audio.scaleName, fresh.streams.build);
    fresh.inputIds = inputs.map((n) => n.id);
    if (ATTN) {
      fresh.attachAttention(
        new RegionalAttention(fresh.graph, SCALES[audio.scaleName].length, {
          strength: parseFloat($('attnStrength')?.value ?? 0.6),
          temporalMix: true, // STSA-style: attend over context depth too
        })
      );
    }
  }
  adopt(fresh);
}

// Wire an organism (freshly grown OR restored from a save) into the page.
function adopt(newLab) {
  lab = newLab;
  if (DUET) {
    sensoryInputs = lab.inputIds.map((id) => lab.graph.neurons.get(id)).filter(Boolean);
    encoder = new SpikeEncoder(lab, sensoryInputs, audio.scaleName);
    dialogue = new DialogueTracker();
    dialogue.onExchange = updateDialogue;
    dialogue.onCallStart = () => callCounts.clear();
    // when your call ends, the walkers are dropped onto the anatomy your
    // call just lit up — their traversal is the answer
    dialogue.onResponseStart = () => {
      if (!$('qa')?.checked || lab.walkers.params.count === 0) return;
      const now = lab.engine.stepCount;
      const top = [...callCounts.entries()]
        .map(([id, count]) => {
          const n = lab.graph.neurons.get(id);
          // attention mode: recency-weight the call activity (temporal
          // attention) — the answer picks up the tail of the question
          const w =
            ATTN && n?.lastSpikeStep >= 0
              ? count * Math.exp(-(now - n.lastSpikeStep) / 800)
              : count;
          return [id, w, n];
        })
        .filter(([, , n]) => n?.role === 'excitatory')
        .sort((a, b) => b[1] - a[1])
        .map(([id]) => id)
        .slice(0, Math.max(2, lab.walkers.params.count));
      if (top.length) lab.walkers.seedAt(top);
      // answer at the pace of the question
      lab.walkers.setPhrase(dialogue.meanCallIOI());
    };
    rebuildPads();
    updateDialogue();
  }
  audio.pulseMs = lab.simParams.pulsePeriodMs;
  lab.walkers.posOf = (id) => {
    const n = lab.graph.neurons.get(id);
    return n ? renderer.positionOf(lab.graph, n) : null;
  };
  renderer = new Renderer($('net'));
  epochMarks = [];

  lab.engine.onSpike = (n) => {
    if (DUET && dialogue?.state === 'call' && n.role === 'excitatory') {
      callCounts.set(n.id, (callCounts.get(n.id) ?? 0) + 1);
    }
    if (DUET && n.role !== 'input') dialogue?.spike(); // energy accounting
    if (!n.isOutput) return;
    // Q&A gating: while you're mid-phrase, the model holds its voice and
    // answers in the gap you leave (spikes still happen — only audio waits)
    const qaGated =
      DUET && $('qa')?.checked && dialogue.humanActive(lab.engine.stepCount);
    if (audio.enabled && !qaGated) {
      const fanout = lab.graph.outgoing.get(n.id)?.length ?? 0;
      audio.noteOn(n, renderer.panOf(lab.graph, n), fanout);
    }
    if (DUET) {
      const scale = SCALES[audio.scaleName];
      const degree = n.structDegree % scale.length;
      dialogue.modelNote(degree, lab.engine.stepCount);
      if (midiIO.output && !qaGated) {
        midiIO.send(36 + lab.key.offset + n.octave * 12 + scale[degree], 0.7);
      }
      flashPad(n.structDegree, 'model');
    }
  };

  lab.walkers.onNote = (n, walkerIndex) => {
    // in q&a mode walkers speak only during the response window
    const inResponse = !DUET || !$('qa')?.checked || dialogue.state === 'response';
    if (DUET) {
      const scale = SCALES[audio.scaleName];
      const degree = n.structDegree % scale.length;
      dialogue.modelNote(degree, lab.engine.stepCount);
      if (inResponse) {
        flashPad(n.structDegree, 'model');
        if (midiIO.output) midiIO.send(36 + lab.key.offset + n.octave * 12 + scale[degree], 0.6);
      }
    }
    if (audio.enabled && inResponse) {
      audio.walkerNote(n, renderer.panOf(lab.graph, n), walkerIndex);
    }
  };

  // a fresh organism starts in C; a restored one keeps its harmonic state
  audio.keyOffset = lab.key.offset;
  if (fifthsWheel) fifthsWheel.setKey(lab.key.fifths);
  lab.onKeyChange = ({ neuronId, name, rule, fifths, offset }) => {
    audio.keyOffset = offset;
    renderer.markKeyChange(lab.graph, neuronId);
    if (fifthsWheel) fifthsWheel.setKey(fifths, rule);
    if (DUET) rebuildPads(); // pad note names follow the key
    updateStats();
    updateLog();
  };

  lab.onEpoch = (l) => {
    const { born, pruned, subdivided } = l.lastEpochChanges;
    for (const id of born) {
      renderer.markBirth(l.graph, id);
      audio.birthChime(0);
    }
    for (const id of pruned) {
      renderer.markDeath(id);
      audio.pruneThud(0);
    }
    for (const sub of subdivided) {
      const children = sub.children
        .map((p) => l.graph.regions.get(p))
        .filter(Boolean);
      renderer.invalidatePositions(children.flatMap((c) => [...c.members]));
      const depth = children[0]?.depth ?? 2;
      audio.divisionArpeggio(Math.min(depth, 5), 0);
    }
    const i = l.activity.rateHistory.length - 1;
    if (born.length || pruned.length) {
      epochMarks.push({ i, born: born.length, pruned: pruned.length });
    }
    // history is a bounded shifting buffer — keep marks aligned
    if (l.activity.rateHistory.length === l.activity.maxHistory) {
      epochMarks = epochMarks.map((m) => ({ ...m, i: m.i - 1 })).filter((m) => m.i >= 0);
    }
    updateStats();
    updateLog();
  };

  if (DUET) {
    rebuildPads();
    updateDialogue();
  }
  updateStats();
  updateLog();
}

// ---- organism persistence ----

const STORE_KEY = `organism-${MODE}`;

function flashBtn(id, text) {
  const btn = $(id);
  if (!btn) return;
  const orig = btn.dataset.label ?? btn.textContent;
  btn.dataset.label = orig;
  btn.textContent = text;
  clearTimeout(btn._revert);
  btn._revert = setTimeout(() => (btn.textContent = orig), 1200);
}

function saveOrganism() {
  const data = serializeLab(lab);
  data.savedAt = new Date().toISOString();
  localStorage.setItem(STORE_KEY, JSON.stringify(data));
  flashBtn('saveBtn', '✓ saved');
}

function loadOrganism() {
  const raw = localStorage.getItem(STORE_KEY);
  if (!raw) {
    flashBtn('loadBtn', 'no save');
    return;
  }
  adopt(deserializeLab(JSON.parse(raw), { AttentionClass: RegionalAttention }));
  flashBtn('loadBtn', '✓ loaded');
}

function exportOrganism() {
  const data = serializeLab(lab);
  data.savedAt = new Date().toISOString();
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `organism-${lab.seed}-e${lab.epoch}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function importOrganism(file) {
  file.text().then((text) => {
    try {
      adopt(deserializeLab(JSON.parse(text), { AttentionClass: RegionalAttention }));
      flashBtn('importBtn', '✓ imported');
    } catch (err) {
      flashBtn('importBtn', 'bad file');
    }
  });
}

function updateStats() {
  const r = lab.report();
  $('stats').innerHTML = `
    <div><span>epoch</span><b>${r.epoch}</b></div>
    <div><span>neurons</span><b>${r.neurons}</b></div>
    <div><span>synapses</span><b>${r.synapses}</b></div>
    <div><span>E / I</span><b>${r.excitatory} / ${r.inhibitory}</b></div>
    <div><span>voices (outputs)</span><b>${r.outputs}</b></div>
    <div><span>leaf regions</span><b>${r.leafRegions}</b></div>
    <div><span>octave spread</span><b>${r.octaveSpread}</b></div>
    <div><span>key</span><b>${r.key}${r.keyChanges ? ` (${r.keyChanges} changes)` : ''}</b></div>
    <div><span>mean rate</span><b>${r.meanRateHz.toFixed(1)} Hz</b></div>
    <div><span>grown / pruned</span><b>${r.grownTotal} / ${r.prunedTotal}</b></div>
    <div><span>region divisions</span><b>${r.subdividedTotal}</b></div>
  `;
  drawRateSparkline(
    $('spark'),
    lab.activity.rateHistory,
    [lab.dev.params.lowRateHz, lab.dev.params.highRateHz],
    epochMarks
  );
}

function updateLog() {
  const items = lab.dev.events.slice(-9).reverse();
  $('log').innerHTML = items
    .map((e) => {
      if (e.type === 'NeuronBorn') {
        const voice = e.isOutput ? ' ♪' : '';
        return `<li class="born">e${e.epoch} ✚ ${e.role} born in ${e.region}${voice} <em>(${e.rate.toFixed(1)} Hz)</em></li>`;
      }
      if (e.type === 'SynapseGrown') {
        return `<li class="born">e${e.epoch} ⇢ afferent grown ${e.from} → ${e.region} <em>(${e.rate.toFixed(1)} Hz)</em></li>`;
      }
      if (e.type === 'RegionExpanded') {
        return `<li class="divided">e${e.epoch} ◈ ${e.region} divided → ${e.children.join(', ')} <em>(${e.size} cells)</em></li>`;
      }
      if (e.type === 'RegionSprouted') {
        return `<li class="divided">e${e.epoch} ⑂ ${e.region} branched → ${e.children.join(', ')} <em>(+${e.newNeurons} neurons)</em></li>`;
      }
      if (e.type === 'KeyChanged') {
        const src = e.id === 'manual' ? '' : `, node ${e.id}`;
        return `<li class="keychange">e${e.epoch} ♮ key → ${e.key} <em>(${e.rule}${src})</em></li>`;
      }
      if (e.type === 'Reinforced') {
        return `<li class="born">e${e.epoch} ✚ reinforced ${e.count} active neurons</li>`;
      }
      return `<li class="pruned">e${e.epoch} ✕ neuron ${e.id} pruned from ${e.region} <em>(energy ${e.energy?.toFixed(2) ?? '?'})</em></li>`;
    })
    .join('');
}

// ---- duet mode: pads, MIDI, reinforcement ----

const PAD_KEYS = 'asdfghjklqwertyuiop';
let padOctave = 2; // lower pad row; upper row is padOctave + 1

function rebuildPads() {
  const padbar = $('pads');
  if (!padbar) return;
  const scale = SCALES[audio.scaleName];
  padbar.innerHTML = '';
  const octLabel = $('padOctVal');
  if (octLabel) octLabel.textContent = `${padOctave}·${padOctave + 1}`;
  [padOctave, padOctave + 1].forEach((oct, row) => {
    for (let d = 0; d < scale.length; d++) {
      const idx = row * scale.length + d;
      const b = document.createElement('button');
      b.className = 'pad';
      b.dataset.degree = d;
      const pc = (scale[d] + lab.key.offset) % 12;
      b.innerHTML = `${NOTE_NAMES[pc]}<small>${PAD_KEYS[idx] ?? ''}</small>`;
      b.addEventListener('pointerdown', () => playHuman(oct, d));
      padbar.appendChild(b);
    }
  });
}

function playHuman(octave, degree) {
  if (!lab || !encoder) return;
  const scale = SCALES[audio.scaleName];
  const midi = 36 + lab.key.offset + octave * 12 + scale[degree];
  encoder.noteOn(midi, 0.8);
  dialogue?.humanNote(degree, lab.engine.stepCount);
  audio.humanNote(octave, degree);
  flashPad(degree, 'human');
}

function updateDialogue() {
  const el = $('dialogue');
  if (!el || !dialogue) return;
  const last = dialogue.exchanges[dialogue.exchanges.length - 1];
  let extra = '';
  if (ATTN) {
    const top = lab.attention?.topRegion;
    extra = `
      <div><span>spikes / answer</span><b>${last ? last.respSpikes : '—'}</b></div>
      <div><span>attending</span><b>${top ? `${top.path} ×${top.gain.toFixed(2)}` : '—'}</b></div>
    `;
  }
  el.innerHTML = `
    <div><span>exchanges</span><b>${dialogue.exchanges.length}</b></div>
    <div><span>last call → resp</span><b>${last ? `${last.callNotes} → ${last.respNotes}` : '—'}</b></div>
    <div><span>last relatedness</span><b>${last ? last.score.toFixed(2) : '—'}</b></div>
    <div><span>rhythm match</span><b>${last && last.rhythm ? last.rhythm.toFixed(2) : '—'}</b></div>
    <div><span>recent avg (10)</span><b>${dialogue.recentMean(10).toFixed(2)}</b></div>
    ${extra}
  `;
}

function flashPad(structDegree, kind) {
  const pads = $('pads');
  if (!pads) return;
  const scale = SCALES[audio.scaleName];
  const d = structDegree % scale.length;
  for (const b of pads.querySelectorAll(`.pad[data-degree="${d}"]`)) {
    b.classList.add(kind);
    setTimeout(() => b.classList.remove(kind), 180);
  }
}

function reinforceRecent() {
  if (!lab) return;
  let count = 0;
  const now = lab.engine.stepCount;
  for (const n of lab.graph.neurons.values()) {
    if (n.role === 'input') continue;
    if (n.lastSpikeStep >= 0 && now - n.lastSpikeStep < 2500) {
      n.energy = Math.min(1.5, n.energy + 0.35);
      renderer.markBirth(lab.graph, n.id); // green reward ripple on each one
      count++;
    }
  }
  lab.reward(1); // R-STDP: converts eligibility traces to weight change (no-op in immediate mode)
  lab.dev.log({ epoch: lab.epoch, type: 'Reinforced', count });
  updateLog();

  // visible button feedback: how many neurons got the reward
  const btn = $('reinforce');
  if (btn) {
    btn.textContent = count ? `✚ reinforced ${count}` : '✚ nothing fired recently';
    btn.classList.add('flash');
    clearTimeout(btn._revert);
    btn._revert = setTimeout(() => {
      btn.textContent = '✚ reinforce';
      btn.classList.remove('flash');
    }, 1100);
  }
}

function setupDuet() {
  $('reinforce')?.addEventListener('click', reinforceRecent);

  // computer keyboard: two pad rows
  window.addEventListener('keydown', (e) => {
    if (e.repeat || e.metaKey || e.ctrlKey) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    const idx = PAD_KEYS.indexOf(e.key.toLowerCase());
    if (idx < 0 || !lab) return;
    const len = SCALES[audio.scaleName].length;
    if (idx >= len * 2) return;
    playHuman(idx < len ? padOctave : padOctave + 1, idx % len);
  });

  const shiftPadOctave = (d) => {
    padOctave = Math.max(0, Math.min(4, padOctave + d));
    rebuildPads();
  };
  $('padOctDown')?.addEventListener('click', () => shiftPadOctave(-1));
  $('padOctUp')?.addEventListener('click', () => shiftPadOctave(1));
  // z / x also shift the playing octave, like most soft keyboards
  window.addEventListener('keydown', (e) => {
    if (e.repeat || e.metaKey || e.ctrlKey) return;
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (e.key === 'z') shiftPadOctave(-1);
    if (e.key === 'x') shiftPadOctave(1);
  });

  // human MIDI in → spikes; model out → MIDI hardware
  midiIO.onNote = (note, vel) => {
    if (!encoder) return;
    const r = encoder.noteOn(note, vel);
    if (r) {
      dialogue?.humanNote(r.degree, lab.engine.stepCount);
      flashPad(r.degree, 'human');
    }
  };

  // train from a MIDI file: the file becomes the player
  $('midiTrainBtn')?.addEventListener('click', () => {
    if (trainNotes) {
      trainNotes = null; // acts as a stop button while training
      flashBtn('midiTrainBtn', 'stopped');
      return;
    }
    $('midiFile').click();
  });
  $('midiFile')?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file) return;
    file.arrayBuffer().then((buf) => {
      try {
        const parsed = parseSMF(buf);
        if (!parsed.notes.length) throw new Error('no notes');
        trainNotes = parsed.notes;
        trainIdx = 0;
        trainStartStep = lab.engine.stepCount;
        running = true;
        $('runBtn').textContent = '⏸ pause';
        flashBtn('midiTrainBtn', `${parsed.notes.length} notes…`);
      } catch {
        flashBtn('midiTrainBtn', 'bad midi file');
      }
    });
  });

  $('midiBtn')?.addEventListener('click', async () => {
    if (!midiIO.supported) {
      $('midiBtn').textContent = 'no midi support';
      return;
    }
    try {
      await midiIO.init();
      const fill = (sel, list) => {
        sel.innerHTML = '<option value="">—</option>';
        for (const d of list) {
          const o = document.createElement('option');
          o.value = d.id;
          o.textContent = d.name;
          sel.appendChild(o);
        }
        sel.style.display = '';
      };
      fill($('midiIn'), midiIO.inputs());
      fill($('midiOut'), midiIO.outputs());
      $('midiBtn').textContent = `midi ✓ (${midiIO.inputs().length} in / ${midiIO.outputs().length} out)`;
      $('midiIn').addEventListener('change', (e) => midiIO.bindInput(e.target.value));
      $('midiOut').addEventListener('change', (e) => midiIO.bindOutput(e.target.value));
      if (midiIO.inputs().length) {
        $('midiIn').value = midiIO.inputs()[0].id;
        midiIO.bindInput(midiIO.inputs()[0].id);
      }
    } catch {
      $('midiBtn').textContent = 'midi blocked';
    }
  });
}

function frame(t) {
  requestAnimationFrame(frame);
  if (!lab) return;
  if (running) {
    if (!lastTime) lastTime = t;
    accum += Math.min(t - lastTime, 100) * speed;
    lastTime = t;
    const steps = Math.min(Math.floor(accum), 120);
    accum -= steps;
    for (let i = 0; i < steps; i++) {
      // MIDI-file training: replay the file into the organism on sim time
      // (crank the speed control to train faster than real time)
      if (trainNotes) {
        const simMs = lab.engine.stepCount - trainStartStep;
        while (trainIdx < trainNotes.length && trainNotes[trainIdx].tMs <= simMs) {
          const ev = trainNotes[trainIdx++];
          const r = encoder?.noteOn(ev.note, ev.vel);
          if (r) {
            dialogue?.humanNote(r.degree, lab.engine.stepCount);
            flashPad(r.degree, 'human');
          }
        }
        if (trainIdx >= trainNotes.length) {
          trainNotes = null;
          flashBtn('midiTrainBtn', '✓ done');
        } else if (lab.engine.stepCount % 1000 === 0) {
          const btn = $('midiTrainBtn');
          if (btn) btn.textContent = `training ${Math.round((trainIdx / trainNotes.length) * 100)}%`;
        }
      }
      lab.step();
    }
    if (DUET && dialogue) dialogue.tick(lab.engine.stepCount);
  } else {
    lastTime = t;
  }
  renderer.draw(lab.graph, lab.engine, lab.walkers, ATTN ? lab.attention : null);
}

function fitCanvas() {
  const c = $('net');
  const rect = c.parentElement.getBoundingClientRect();
  c.width = rect.width;
  c.height = rect.height;
  const s = $('spark');
  s.width = s.parentElement.getBoundingClientRect().width - 24;
  s.height = 64;
}

window.addEventListener('resize', fitCanvas);

window.addEventListener('DOMContentLoaded', () => {
  const scaleSel = $('scale');
  for (const name of Object.keys(SCALES)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === audio.scaleName) opt.selected = true;
    scaleSel.appendChild(opt);
  }
  const tuningSel = $('tuning');
  for (const name of Object.keys(TUNINGS)) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === audio.tuningName) opt.selected = true;
    tuningSel.appendChild(opt);
  }
  tuningSel.addEventListener('change', (e) => {
    audio.tuningName = e.target.value;
  });

  const setOctave = (d) => {
    audio.octaveShift = Math.max(-2, Math.min(3, audio.octaveShift + d));
    $('octVal').textContent = audio.octaveShift > 0 ? `+${audio.octaveShift}` : `${audio.octaveShift}`;
  };
  $('octDown').addEventListener('click', () => setOctave(-1));
  $('octUp').addEventListener('click', () => setOctave(1));

  $('octWidth').addEventListener('input', (e) => {
    audio.octaveWidth = parseInt(e.target.value, 10);
    $('octWidthVal').textContent = e.target.value;
  });

  $('keyDrift').addEventListener('input', (e) => {
    lab.simParams.modProb = parseFloat(e.target.value);
    $('keyDriftVal').textContent = e.target.value;
  });

  $('walkerSpread').addEventListener('change', (e) => {
    audio.walkerSpread = e.target.checked;
  });

  fitCanvas();
  build(parseInt($('seed').value, 10));
  if (DUET) setupDuet();

  // interactive circle of fifths: shows the key, click to set it manually
  fifthsWheel = new FifthsWheel($('viewwrap'), (i) => {
    lab.key.fifths = i;
    lab.key.offset = (7 * i) % 12;
    lab.lastModStep = lab.engine.stepCount; // grace period before modulators drift again
    audio.keyOffset = lab.key.offset;
    fifthsWheel.setKey(i);
    lab.dev.log({ epoch: lab.epoch, type: 'KeyChanged', id: 'manual', rule: 'selected', key: KEY_NAMES[i] });
    if (DUET) rebuildPads();
    updateStats();
    updateLog();
  });
  fifthsWheel.setKey(lab.key.fifths);

  requestAnimationFrame(frame);

  $('runBtn').addEventListener('click', () => {
    running = !running;
    $('runBtn').textContent = running ? '⏸ pause' : '▶ run';
  });

  $('audioToggle').addEventListener('change', async (e) => {
    if (e.target.checked) {
      await audio.enable();
      // apply current slider state to the freshly built fx graph
      audio.setFx('reverb', parseFloat($('fxReverb').value));
      audio.setFx('delay', parseFloat($('fxDelay').value));
      audio.setFx('chorus', parseFloat($('fxChorus').value));
      audio.setDelayFromPulse(parseInt($('tempo').value, 10));
    } else {
      audio.disable();
    }
  });

  $('devToggle').addEventListener('change', (e) => {
    lab.simParams.developmentEnabled = e.target.checked;
  });

  $('devSounds').addEventListener('change', (e) => {
    audio.structuralSounds = e.target.checked;
  });

  $('spikeNotes').addEventListener('change', (e) => {
    audio.spikeNotes = e.target.checked;
  });

  $('rebuild').addEventListener('click', () => {
    build(parseInt($('seed').value, 10) || 42);
  });

  $('branch').addEventListener('click', () => {
    const ch = lab.branchOut(2);
    renderer.invalidatePositions(ch.moved);
    for (const id of ch.born) renderer.markBirth(lab.graph, id);
    for (const sub of ch.sprouted) {
      const depth = lab.graph.regions.get(sub.children[0])?.depth ?? 2;
      audio.divisionArpeggio(Math.min(depth, 5), 0);
    }
    updateStats();
    updateLog();
  });

  $('reseed').addEventListener('click', () => {
    $('seed').value = Math.floor(Math.random() * 100000);
    build(parseInt($('seed').value, 10));
  });

  $('saveBtn')?.addEventListener('click', saveOrganism);
  $('loadBtn')?.addEventListener('click', loadOrganism);
  $('exportBtn')?.addEventListener('click', exportOrganism);
  $('importBtn')?.addEventListener('click', () => $('importFile').click());
  $('importFile')?.addEventListener('change', (e) => {
    if (e.target.files[0]) importOrganism(e.target.files[0]);
    e.target.value = '';
  });

  scaleSel.addEventListener('change', (e) => {
    audio.scaleName = e.target.value;
    if (DUET) rebuildPads(); // note: sensory wiring stays from build time — press grow to re-wire
  });

  $('tempo').addEventListener('input', (e) => {
    lab.simParams.pulsePeriodMs = parseInt(e.target.value, 10);
    $('tempoVal').textContent = `${e.target.value} ms`;
    audio.setDelayFromPulse(parseInt(e.target.value, 10));
    audio.pulseMs = parseInt(e.target.value, 10);
  });

  $('quantize').addEventListener('input', (e) => {
    audio.quantize = parseFloat(e.target.value);
  });

  $('grid').addEventListener('change', (e) => {
    audio.gridDiv = parseInt(e.target.value, 10);
  });

  $('drive').addEventListener('change', (e) => {
    lab.simParams.drivePattern = e.target.value;
  });

  $('stdp').addEventListener('change', (e) => {
    lab.simParams.stdpEnabled = e.target.checked;
  });

  $('momentum').addEventListener('input', (e) => {
    lab.walkers.params.momentum = parseFloat(e.target.value);
  });

  $('walkRate').addEventListener('change', (e) => {
    lab.walkers.params.stepDivisor = parseInt(e.target.value, 10);
  });

  $('rubato').addEventListener('input', (e) => {
    lab.walkers.params.rubato = parseFloat(e.target.value);
  });

  $('attnStrength')?.addEventListener('input', (e) => {
    if (lab.attention) lab.attention.params.strength = parseFloat(e.target.value);
  });

  // steer: hovering the network pulls walkers toward the cursor
  const net = $('net');
  net.addEventListener('pointermove', (e) => {
    if (!$('steer').checked || !renderer.view) {
      lab.walkers.attractor = null;
      return;
    }
    const rect = net.getBoundingClientRect();
    const { S, ox, oy } = renderer.view;
    lab.walkers.attractor = {
      x: (e.clientX - rect.left - ox) / S,
      y: (e.clientY - rect.top - oy) / S,
      strength: 6,
    };
  });
  net.addEventListener('pointerleave', () => {
    lab.walkers.attractor = null;
  });
  $('steer').addEventListener('change', (e) => {
    if (!e.target.checked) lab.walkers.attractor = null;
  });

  for (const [id, name] of [
    ['fxReverb', 'reverb'],
    ['fxDelay', 'delay'],
    ['fxChorus', 'chorus'],
  ]) {
    $(id).addEventListener('input', (e) => {
      audio.setFx(name, parseFloat(e.target.value));
    });
  }

  $('density').addEventListener('input', (e) => {
    audio.minRetriggerMs = parseInt(e.target.value, 10);
  });

  $('strum').addEventListener('input', (e) => {
    audio.strumMs = parseInt(e.target.value, 10);
  });

  $('walkers').addEventListener('input', (e) => {
    lab.walkers.setCount(parseInt(e.target.value, 10));
    $('walkersVal').textContent = e.target.value;
  });

  $('variation').addEventListener('input', (e) => {
    lab.walkers.params.variation = parseFloat(e.target.value);
  });

  $('speedSel').addEventListener('change', (e) => {
    speed = parseFloat(e.target.value);
  });
});
