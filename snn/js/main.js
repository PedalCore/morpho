import { Lab, KEY_NAMES } from './sim/lab.js';
import { Renderer, drawRateSparkline } from './ui/renderer.js';
import { AudioEngine, SCALES, TUNINGS } from './ui/audio.js';
import { FifthsWheel } from './ui/fifths.js';
import { wireSensoryInputs, SpikeEncoder, NOTE_NAMES } from './duet/sensory.js';
import { MidiIO } from './io/midi.js';

const $ = (id) => document.getElementById(id);
// duet mode: no metronome drive — the human plays the organism via MIDI/pads
const DUET = document.body.dataset.mode === 'duet';

const audio = new AudioEngine();
const midiIO = new MidiIO();
let lab = null;
let renderer = null;
let fifthsWheel = null;
let encoder = null;
let sensoryInputs = [];
let running = false;
let speed = 1;
let epochMarks = []; // {i, born, pruned} aligned to rateHistory indices
let lastTime = 0;
let accum = 0;

function build(seed) {
  lab = new Lab({
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
    },
  });
  if (DUET) {
    sensoryInputs = wireSensoryInputs(lab.graph, audio.scaleName, lab.streams.build);
    lab.inputIds = sensoryInputs.map((n) => n.id);
    encoder = new SpikeEncoder(lab, sensoryInputs, audio.scaleName);
    rebuildPads();
  }
  audio.pulseMs = lab.simParams.pulsePeriodMs;
  lab.walkers.posOf = (id) => {
    const n = lab.graph.neurons.get(id);
    return n ? renderer.positionOf(lab.graph, n) : null;
  };
  renderer = new Renderer($('net'));
  epochMarks = [];

  lab.engine.onSpike = (n) => {
    if (!n.isOutput) return;
    if (audio.enabled) {
      const fanout = lab.graph.outgoing.get(n.id)?.length ?? 0;
      audio.noteOn(n, renderer.panOf(lab.graph, n), fanout);
    }
    if (DUET) {
      if (midiIO.output) {
        const scale = SCALES[audio.scaleName];
        const midi = 36 + lab.key.offset + n.octave * 12 + scale[n.structDegree % scale.length];
        midiIO.send(midi, 0.7);
      }
      flashPad(n.structDegree, 'model');
    }
  };

  lab.walkers.onNote = (n, walkerIndex) => {
    if (audio.enabled) audio.walkerNote(n, renderer.panOf(lab.graph, n), walkerIndex);
  };

  audio.keyOffset = 0; // fresh organism starts back in C
  if (fifthsWheel) fifthsWheel.setKey(0);
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

  updateStats();
  updateLog();
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
  audio.humanNote(octave, degree);
  flashPad(degree, 'human');
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
    encoder.noteOn(note, vel);
    const scale = SCALES[audio.scaleName];
    flashPad(0, 'human'); // generic flash; precise degree below
    const rel = (((note % 12) - lab.key.offset) % 12 + 12) % 12;
    let best = 0;
    let bd = 99;
    scale.forEach((s, i) => {
      const d = Math.min((rel - s + 12) % 12, (s - rel + 12) % 12);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    flashPad(best, 'human');
  };

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
    for (let i = 0; i < steps; i++) lab.step();
  } else {
    lastTime = t;
  }
  renderer.draw(lab.graph, lab.engine, lab.walkers);
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
