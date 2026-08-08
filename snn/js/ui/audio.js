// Sonification layer. Pitch is structural and fixed at each neuron's birth
// (plugin briefs: recursion depth → register, structural position → scale
// degree), so the anatomy is the note pool:
//   - deep microcircuits chirp high, shallow regions sit low
//   - region subdivision opens a new, higher register
//   - growth adds pitches, pruning removes them
//
// Near-simultaneous notes are STRUMMED: a shared scheduling cursor fans
// chords out into fast arpeggios (spacing adjustable, low→high by arrival).
//
// Audio state is kept entirely out of the simulation: the sim is
// deterministic with audio on or off.

export const SCALES = {
  major: [0, 2, 4, 5, 7, 9, 11],
  'natural minor': [0, 2, 3, 5, 7, 8, 10],
  'major pentatonic': [0, 2, 4, 7, 9],
  'minor pentatonic': [0, 3, 5, 7, 10],
  dorian: [0, 2, 3, 5, 7, 9, 10],
  lydian: [0, 2, 4, 6, 7, 9, 11],
  'phrygian dominant': [0, 1, 4, 5, 7, 8, 10],
  'harmonic minor': [0, 2, 3, 5, 7, 8, 11],
  'whole tone': [0, 2, 4, 6, 8, 10],
  hirajoshi: [0, 2, 3, 7, 8],
};

// Tuning systems. Scales stay defined in 12-TET semitones; EDO tunings map
// each semitone to its nearest step (24-TET is exact, 19/31-TET give the
// meantone-flavoured versions), just intonation maps to 5-limit ratios.
// The circle-of-fifths key offset is retuned the same way, so modulations
// land inside the chosen tuning.
export const TUNINGS = {
  '12-TET': { divisions: 12 },
  '19-TET': { divisions: 19 },
  '24-TET (quarter tones)': { divisions: 24 },
  '31-TET': { divisions: 31 },
  'just intonation (5-limit)': {
    ratios: [1 / 1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8],
  },
};

const ROOT_HZ = 440 * Math.pow(2, (36 - 69) / 12); // C2 — depth spans up to 5 octaves above

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.enabled = false;
    this.scaleName = 'minor pentatonic';
    this.tuningName = '12-TET';
    this.keyOffset = 0; // semitone shift set by the sim's circle-of-fifths state
    this.octaveShift = 0; // global transpose, in octaves
    this.octaveWidth = 5; // register span: structural octaves 1–5 compressed into this many
    this.walkerSpread = false; // give each walker its own register band
    this.structuralSounds = true; // birth chimes / prune thuds / division arpeggios
    this.spikeNotes = true; // output-neuron spike voices (off → walkers only)
    this.minRetriggerMs = 90; // per-neuron note gate (density control)
    this.strumMs = 24; // fan-out spacing for near-simultaneous notes
    this.quantize = 0; // 0 = raw spike timing, 1 = hard-snapped to the grid
    this.gridDiv = 2; // grid step = pulse / gridDiv (2 = eighth notes)
    this.pulseMs = 340; // kept in sync with the sim's pulse period
    this.lastNoteTime = new Map(); // neuronId -> ctx time
    this.strumCursor = 0; // shared "next available onset" time
    this.voiceCount = 0;
    this.maxVoices = 28;
  }

  async enable() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = this.ctx;
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18;
      comp.ratio.value = 6;
      comp.connect(ctx.destination);

      // all voices feed master; master splits into dry + fx sends
      this.master = ctx.createGain();
      this.master.gain.value = 0.5;

      const dry = ctx.createGain();
      dry.gain.value = 1;
      this.master.connect(dry);
      dry.connect(comp);

      // reverb send → convolver (generated decaying-noise impulse)
      this.reverbSend = ctx.createGain();
      this.reverbSend.gain.value = 0.35;
      const reverb = ctx.createConvolver();
      reverb.buffer = this._makeImpulse(2.6, 2.8);
      this.master.connect(this.reverbSend);
      this.reverbSend.connect(reverb);
      reverb.connect(comp);

      // delay send → feedback delay with darkened repeats
      this.delaySend = ctx.createGain();
      this.delaySend.gain.value = 0.25;
      this.delayNode = ctx.createDelay(2.0);
      this.delayNode.delayTime.value = 0.255; // dotted pulse; follows tempo
      const fbFilter = ctx.createBiquadFilter();
      fbFilter.type = 'lowpass';
      fbFilter.frequency.value = 2800;
      const feedback = ctx.createGain();
      feedback.gain.value = 0.38;
      this.master.connect(this.delaySend);
      this.delaySend.connect(this.delayNode);
      this.delayNode.connect(fbFilter);
      fbFilter.connect(feedback);
      feedback.connect(this.delayNode);
      this.delayNode.connect(comp);

      // chorus send → LFO-modulated short delay
      this.chorusSend = ctx.createGain();
      this.chorusSend.gain.value = 0;
      const chorusDelay = ctx.createDelay(0.06);
      chorusDelay.delayTime.value = 0.018;
      const lfo = ctx.createOscillator();
      lfo.type = 'sine';
      lfo.frequency.value = 0.55;
      const lfoDepth = ctx.createGain();
      lfoDepth.gain.value = 0.005;
      lfo.connect(lfoDepth);
      lfoDepth.connect(chorusDelay.delayTime);
      lfo.start();
      this.master.connect(this.chorusSend);
      this.chorusSend.connect(chorusDelay);
      chorusDelay.connect(comp);
    }
    await this.ctx.resume();
    this.enabled = true;
  }

  _makeImpulse(seconds, decay) {
    const rate = this.ctx.sampleRate;
    const len = Math.floor(rate * seconds);
    const buf = this.ctx.createBuffer(2, len, rate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
      }
    }
    return buf;
  }

  setFx(name, value) {
    const node = { reverb: this.reverbSend, delay: this.delaySend, chorus: this.chorusSend }[name];
    if (node) node.gain.value = value;
  }

  // delay echoes track the organism's pulse (dotted feel)
  setDelayFromPulse(pulseMs) {
    if (this.delayNode) this.delayNode.delayTime.value = Math.min(1.9, (pulseMs * 0.75) / 1000);
  }

  disable() {
    this.enabled = false;
    if (this.ctx) this.ctx.suspend();
  }

  // octave + scale degree index → Hz, through the active scale, tuning and key.
  // Structural octaves (1–5, from region depth) are first compressed into the
  // chosen register width, then shifted by the global octave transpose;
  // extraOctave (walker spread) is applied after compression so voice
  // separation survives a narrow width.
  freqFor(octave, structDegree, extraOctave = 0) {
    const scale = SCALES[this.scaleName];
    const degree = ((structDegree % scale.length) + scale.length) % scale.length;
    const semis = scale[degree];
    const tuning = TUNINGS[this.tuningName];
    const structural = Math.max(1, Math.min(5, octave));
    const compressed = 1 + Math.round(((structural - 1) / 4) * (this.octaveWidth - 1));
    const centering = Math.floor((5 - this.octaveWidth) / 2); // narrow widths sit mid-register
    const oct = Math.max(0, Math.min(6, compressed + centering + extraOctave + this.octaveShift));
    if (tuning.ratios) {
      return ROOT_HZ * Math.pow(2, oct) * tuning.ratios[semis % 12] * tuning.ratios[this.keyOffset % 12];
    }
    const d = tuning.divisions;
    const steps = Math.round((semis * d) / 12) + Math.round((this.keyOffset * d) / 12);
    return ROOT_HZ * Math.pow(2, oct + steps / d);
  }

  neuronFreq(n) {
    return this.freqFor(n.octave, n.structDegree);
  }

  // Reserve an onset at or after `now`: first pull it toward the musical
  // grid (quantize strength), then respect the strum spacing so chords fan
  // out into arpeggios.
  strumSlot(now) {
    let at = now;
    if (this.quantize > 0) {
      const g = this.pulseMs / this.gridDiv / 1000;
      const snapped = Math.ceil(now / g) * g; // next grid line
      at = now + (snapped - now) * this.quantize;
    }
    const spacing = this.strumMs / 1000;
    at = Math.max(at, this.strumCursor);
    if (at - now > 0.6) return -1; // queue saturated — drop
    this.strumCursor = at + spacing;
    return at;
  }

  _voice({ freq, at, pan, amp, dur, type, filterMul = 6 }) {
    const osc = this.ctx.createOscillator();
    osc.type = type;
    osc.frequency.value = freq;
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = Math.min(freq * filterMul, 9500);
    filter.Q.value = 0.7;
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(amp, at + 0.006);
    gain.gain.exponentialRampToValueAtTime(0.0004, at + dur);
    const panner = this.ctx.createStereoPanner();
    panner.pan.value = Math.max(-0.9, Math.min(0.9, pan));
    osc.connect(filter);
    filter.connect(gain);
    gain.connect(panner);
    panner.connect(this.master);
    this.voiceCount++;
    osc.start(at);
    osc.stop(at + dur + 0.05);
    osc.onended = () => {
      this.voiceCount--;
      osc.disconnect();
      filter.disconnect();
      gain.disconnect();
      panner.disconnect();
    };
  }

  // Spike-driven note from an output neuron. fanout biases velocity and
  // duration (plugin brief: high fanout → accented + shorter).
  noteOn(neuron, pan = 0, fanout = 3) {
    if (!this.enabled || !this.ctx || !this.spikeNotes) return false;
    const now = this.ctx.currentTime;
    const last = this.lastNoteTime.get(neuron.id) ?? -1;
    if (now - last < this.minRetriggerMs / 1000) return false;
    if (this.voiceCount >= this.maxVoices) return false;
    const at = this.strumSlot(now);
    if (at < 0) return false;
    this.lastNoteTime.set(neuron.id, now);

    const f = Math.min(fanout, 10) / 10;
    this._voice({
      freq: this.neuronFreq(neuron),
      at,
      pan,
      amp: 0.07 + 0.06 * f,
      dur: 0.7 - 0.4 * f,
      type: 'triangle',
    });
    return true;
  }

  // Walker melody voice — distinct softer timbre so the stochastic melodic
  // line reads separately from the spike texture. With walkerSpread on, each
  // walker is transposed into its own register band so voices don't pile up
  // on the same octave.
  walkerNote(neuron, pan = 0, walkerIndex = 0) {
    if (!this.enabled || !this.ctx) return false;
    if (this.voiceCount >= this.maxVoices) return false;
    const now = this.ctx.currentTime;
    const at = this.strumSlot(now);
    if (at < 0) return false;
    const shift = this.walkerSpread ? [0, 1, -1, 2][walkerIndex % 4] : 0;
    this._voice({
      freq: this.freqFor(neuron.octave, neuron.structDegree, shift),
      at,
      pan,
      amp: 0.06,
      dur: 0.32,
      type: 'sine',
      filterMul: 4,
    });
    return true;
  }

  // Human monitoring voice for on-screen pads (duet mode) — immediate, no
  // strum queue, so playing feels responsive.
  humanNote(octave, structDegree) {
    if (!this.enabled || !this.ctx) return;
    this._voice({
      freq: this.freqFor(octave, structDegree),
      at: this.ctx.currentTime,
      pan: 0,
      amp: 0.09,
      dur: 0.3,
      type: 'sine',
      filterMul: 5,
    });
  }

  // Structural events get their own quiet sounds so development is audible.
  birthChime(pan = 0) {
    if (!this.enabled || !this.ctx || !this.structuralSounds) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(1400, now);
    osc.frequency.exponentialRampToValueAtTime(2400, now + 0.25);
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.045, now);
    gain.gain.exponentialRampToValueAtTime(0.0004, now + 0.4);
    const panner = this.ctx.createStereoPanner();
    panner.pan.value = pan;
    osc.connect(gain);
    gain.connect(panner);
    panner.connect(this.master);
    osc.start(now);
    osc.stop(now + 0.45);
    osc.onended = () => {
      osc.disconnect();
      gain.disconnect();
      panner.disconnect();
    };
  }

  pruneThud(pan = 0) {
    if (!this.enabled || !this.ctx || !this.structuralSounds) return;
    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(190, now);
    osc.frequency.exponentialRampToValueAtTime(60, now + 0.3);
    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.09, now);
    gain.gain.exponentialRampToValueAtTime(0.0004, now + 0.35);
    const panner = this.ctx.createStereoPanner();
    panner.pan.value = pan;
    osc.connect(gain);
    gain.connect(panner);
    panner.connect(this.master);
    osc.start(now);
    osc.stop(now + 0.4);
    osc.onended = () => {
      osc.disconnect();
      gain.disconnect();
      panner.disconnect();
    };
  }

  // Region subdivision — "Expanded → arpeggio" from the plugin brief's
  // lifecycle table: a quick ascending run announcing the new register.
  divisionArpeggio(baseOctave = 2, pan = 0) {
    if (!this.enabled || !this.ctx || !this.structuralSounds) return;
    const scale = SCALES[this.scaleName];
    const now = this.ctx.currentTime;
    for (let i = 0; i < 4; i++) {
      this._voice({
        freq: this.freqFor(baseOctave + Math.floor(i / scale.length), i),
        at: now + i * 0.05,
        pan,
        amp: 0.05,
        dur: 0.3,
        type: 'triangle',
      });
    }
  }
}
