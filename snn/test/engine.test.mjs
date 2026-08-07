import test from 'node:test';
import assert from 'node:assert/strict';

import { NeuralGraph } from '../js/neural/graph.js';
import { SpikeEngine } from '../js/neural/engine.js';
import { resetNeuronIds } from '../js/neural/neuron.js';
import { resetSynapseIds } from '../js/neural/synapse.js';

function freshGraph() {
  resetNeuronIds();
  resetSynapseIds();
  return new NeuralGraph();
}

test('LIF neuron crosses threshold and resets', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: n.id, weight: 1.5, delaySteps: 1 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id);
  e.step(); // delivery arrives next step
  const spikes = e.step();
  assert.ok(spikes.includes(n.id), 'neuron should spike after suprathreshold input');
  assert.equal(n.membrane, n.resetPotential, 'membrane resets after spike');
});

test('subthreshold input leaks away without spiking', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: n.id, weight: 0.5, delaySteps: 1 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id);
  for (let i = 0; i < 200; i++) {
    assert.ok(!e.step().includes(n.id));
  }
  assert.ok(n.membrane < 0.01, 'membrane should have leaked to near zero');
});

test('refractory period blocks immediate re-firing', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R', refractoryMs: 10 });
  // hammer the neuron with strong input every step
  g.addSynapse({ source: input.id, target: n.id, weight: 2.0, delaySteps: 1 });
  const e = new SpikeEngine(g);

  const spikeSteps = [];
  for (let i = 0; i < 40; i++) {
    e.forceFire(input.id);
    const s = e.step();
    if (s.includes(n.id)) spikeSteps.push(e.stepCount - 1);
  }
  assert.ok(spikeSteps.length >= 2, 'should spike more than once over 40 steps');
  for (let i = 1; i < spikeSteps.length; i++) {
    assert.ok(spikeSteps[i] - spikeSteps[i - 1] >= 10, 'inter-spike interval respects refractory');
  }
});

test('inhibitory synapse lowers membrane potential', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const inhibitor = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: n.id, weight: 0.9, delaySteps: 1 });
  g.addSynapse({ source: inhibitor.id, target: n.id, weight: -0.9, delaySteps: 1 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id);
  e.forceFire(inhibitor.id);
  e.step();
  e.step();
  assert.ok(Math.abs(n.membrane) < 0.05, 'excitation and inhibition should roughly cancel');
});

test('synaptic delay controls arrival time', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: n.id, weight: 1.5, delaySteps: 25 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id);
  let spikeStep = -1;
  for (let i = 0; i < 60; i++) {
    if (e.step().includes(n.id)) {
      spikeStep = e.stepCount - 1;
      break;
    }
  }
  assert.equal(spikeStep, 25, 'spike should occur exactly when the delayed input arrives');
});

test('recurrent loop A->B->A sustains bounded activity', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const a = g.addNeuron({ role: 'excitatory', region: 'R' });
  const b = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: a.id, weight: 1.5, delaySteps: 1 });
  g.addSynapse({ source: a.id, target: b.id, weight: 1.5, delaySteps: 10 });
  g.addSynapse({ source: b.id, target: a.id, weight: 1.5, delaySteps: 10 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id);
  let aSpikes = 0;
  let bSpikes = 0;
  for (let i = 0; i < 500; i++) {
    const s = e.step();
    if (s.includes(a.id)) aSpikes++;
    if (s.includes(b.id)) bSpikes++;
  }
  assert.ok(aSpikes > 10 && bSpikes > 10, 'loop should reverberate');
  assert.ok(aSpikes < 100, 'refractory + delay bound the loop rate');
});

test('removing a neuron removes its synapses and drops in-flight deliveries safely', () => {
  const g = freshGraph();
  g.addRegion('R', 0, 'leaf');
  const input = g.addNeuron({ role: 'input', region: 'R' });
  const n = g.addNeuron({ role: 'excitatory', region: 'R' });
  g.addSynapse({ source: input.id, target: n.id, weight: 1.5, delaySteps: 30 });
  const e = new SpikeEngine(g);

  e.forceFire(input.id); // delivery in flight
  g.removeNeuron(n.id);
  assert.equal(g.synapses.size, 0);
  for (let i = 0; i < 60; i++) e.step(); // must not throw
});
