// Thin Web MIDI wrapper: pick an input to play the organism from a keyboard,
// pick an output to let the organism play your hardware/DAW.

export class MidiIO {
  constructor() {
    this.access = null;
    this.input = null;
    this.output = null;
    this.onNote = null; // (midiNote, velocity01) => void
  }

  get supported() {
    return typeof navigator !== 'undefined' && !!navigator.requestMIDIAccess;
  }

  async init() {
    this.access = await navigator.requestMIDIAccess();
    return this;
  }

  inputs() {
    return this.access ? [...this.access.inputs.values()] : [];
  }

  outputs() {
    return this.access ? [...this.access.outputs.values()] : [];
  }

  bindInput(id) {
    if (this.input) this.input.onmidimessage = null;
    this.input = this.inputs().find((i) => i.id === id) ?? null;
    if (this.input) {
      this.input.onmidimessage = (e) => {
        const [status, note, vel] = e.data;
        if ((status & 0xf0) === 0x90 && vel > 0 && this.onNote) {
          this.onNote(note, vel / 127);
        }
      };
    }
  }

  bindOutput(id) {
    this.output = this.outputs().find((o) => o.id === id) ?? null;
  }

  send(note, velocity = 0.7, durMs = 220) {
    if (!this.output) return;
    const n = note & 0x7f;
    this.output.send([0x90, n, Math.round(velocity * 127)]);
    setTimeout(() => this.output && this.output.send([0x80, n, 0]), durMs);
  }
}
