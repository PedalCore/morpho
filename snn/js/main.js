import { Lab } from './sim/lab.js';
import { Renderer, drawRateSparkline } from './ui/renderer.js';
import { AudioEngine, SCALES, TUNINGS } from './ui/audio.js';

const $ = (id) => document.getElementById(id);

const audio = new AudioEngine();
let lab = null;
let renderer = null;
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
    },
    walk: {
      count: parseInt($('walkers').value, 10),
      variation: parseFloat($('variation').value),
    },
  });
  renderer = new Renderer($('net'));
  epochMarks = [];

  lab.engine.onSpike = (n) => {
    if (n.isOutput && audio.enabled) {
      const fanout = lab.graph.outgoing.get(n.id)?.length ?? 0;
      audio.noteOn(n, renderer.panOf(lab.graph, n), fanout);
    }
  };

  lab.walkers.onNote = (n, walkerIndex) => {
    if (audio.enabled) audio.walkerNote(n, renderer.panOf(lab.graph, n), walkerIndex);
  };

  audio.keyOffset = 0; // fresh organism starts back in C
  lab.onKeyChange = ({ neuronId, name, offset }) => {
    audio.keyOffset = offset;
    renderer.markKeyChange(lab.graph, neuronId);
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
        return `<li class="keychange">e${e.epoch} ♮ key → ${e.key} <em>(${e.rule}, node ${e.id})</em></li>`;
      }
      return `<li class="pruned">e${e.epoch} ✕ neuron ${e.id} pruned from ${e.region} <em>(energy ${e.energy?.toFixed(2) ?? '?'})</em></li>`;
    })
    .join('');
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

  $('keyDrift').addEventListener('input', (e) => {
    lab.simParams.modProb = parseFloat(e.target.value);
    $('keyDriftVal').textContent = e.target.value;
  });

  $('walkerSpread').addEventListener('change', (e) => {
    audio.walkerSpread = e.target.checked;
  });

  fitCanvas();
  build(parseInt($('seed').value, 10));
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
  });

  $('tempo').addEventListener('input', (e) => {
    lab.simParams.pulsePeriodMs = parseInt(e.target.value, 10);
    $('tempoVal').textContent = `${e.target.value} ms`;
    audio.setDelayFromPulse(parseInt(e.target.value, 10));
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
