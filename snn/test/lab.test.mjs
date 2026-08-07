import test from 'node:test';
import assert from 'node:assert/strict';

import { Lab } from '../js/sim/lab.js';

test('grammar expands a compact rule into a substantially larger network', () => {
  const lab = new Lab({ seed: 7 });
  const c = lab.graph.counts();
  assert.ok(c.neurons > 25, `expected substantial expansion, got ${c.neurons}`);
  assert.ok(c.synapses > 60, `expected substantial connectivity, got ${c.synapses}`);
  assert.ok(c.outputs > 0, 'some neurons must be flagged as musical outputs');
  assert.ok(lab.graph.leafRegions().length >= 2, 'multiple leaf regions expected');
  for (const s of lab.graph.synapses.values()) {
    assert.ok(lab.graph.neurons.has(s.source) && lab.graph.neurons.has(s.target));
  }
});

test('stochastic depth gives leaf regions at different depths → register spread', () => {
  // across a few seeds, at least one organism must have depth (octave) variety
  let spreads = [];
  for (const seed of [1, 2, 3, 4, 5]) {
    const lab = new Lab({ seed });
    const depths = new Set(lab.graph.leafRegions().map((r) => r.depth));
    spreads.push(depths.size);
  }
  assert.ok(Math.max(...spreads) >= 2, `expected varied leaf depths, got spreads ${spreads}`);
});

test('pitch is structural: octave follows region depth, fixed at birth', () => {
  const lab = new Lab({ seed: 7 });
  for (const n of lab.graph.neurons.values()) {
    if (n.role === 'input') continue;
    const region = lab.graph.regions.get(n.region);
    assert.equal(n.octave, Math.max(1, Math.min(5, region.depth)));
    assert.ok(Number.isInteger(n.structDegree));
  }
});

test('same seed reproduces identical topology and identical spike history', () => {
  const run = (seed) => {
    const lab = new Lab({ seed, sim: { developmentEnabled: true } });
    const spikeLog = [];
    for (let i = 0; i < 3 * lab.simParams.epochSteps; i++) {
      const spikes = lab.step();
      if (spikes.length) spikeLog.push(`${i}:${spikes.join(',')}`);
    }
    return { log: spikeLog.join('|'), report: lab.report() };
  };
  const a = run(123);
  const b = run(123);
  assert.equal(a.log, b.log, 'spike-for-spike deterministic');
  assert.deepEqual(a.report, b.report);
  const c = run(124);
  assert.notEqual(a.log, c.log, 'different seed diverges');
});

test('network is active: rhythmic input produces network spiking', () => {
  const lab = new Lab({ seed: 42, sim: { developmentEnabled: false } });
  lab.runEpochs(2);
  assert.ok(lab.activity.networkRateHz > 0.2, `network nearly silent: ${lab.activity.networkRateHz} Hz`);
});

test('development grows and prunes without invalid graph state', () => {
  const lab = new Lab({ seed: 42 });
  lab.runEpochs(20);
  const r = lab.report();
  assert.ok(r.grownTotal > 0, 'expected some growth events');
  for (const s of lab.graph.synapses.values()) {
    assert.ok(lab.graph.neurons.has(s.source) && lab.graph.neurons.has(s.target));
  }
  for (const [id, list] of lab.graph.outgoing) {
    assert.ok(lab.graph.neurons.has(id));
    for (const s of list) assert.equal(s.source, id);
  }
});

test('budgets are hard limits over a long run', () => {
  // shallow genotype so the initial organism starts under the budget —
  // the budget constrains development, not the genotype's first expansion
  const lab = new Lab({
    seed: 9,
    grammar: { depth: 2 },
    dev: { maxNeurons: 80, maxSynapses: 1500 },
  });
  assert.ok(lab.graph.neurons.size <= 80, `initial organism too large for this test: ${lab.graph.neurons.size}`);
  for (let e = 0; e < 40; e++) {
    lab.runEpochs(1);
    assert.ok(lab.graph.neurons.size <= 80, `neuron budget exceeded: ${lab.graph.neurons.size}`);
    assert.ok(lab.graph.synapses.size <= 1500 + 200, `synapse count unbounded: ${lab.graph.synapses.size}`);
  }
});

test('input neurons are never pruned', () => {
  const lab = new Lab({ seed: 11 });
  const inputIds = new Set(lab.inputIds);
  lab.runEpochs(25);
  for (const id of inputIds) {
    assert.ok(lab.graph.neurons.has(id), `input neuron ${id} was pruned`);
  }
  for (const ev of lab.dev.events) {
    if (ev.type === 'NeuronPruned') {
      assert.ok(ev.energy < lab.dev.params.pruneEnergy, 'pruned neurons must have starved');
    }
  }
});

test('leaf regions never shrink below the developmental floor', () => {
  const lab = new Lab({ seed: 5 });
  lab.runEpochs(30);
  for (const region of lab.graph.leafRegions()) {
    assert.ok(
      region.members.size >= lab.dev.params.minPerLeafRegion,
      `region ${region.path} fell to ${region.members.size}`
    );
  }
});

test('development responds to activity: silent network gets excitatory/afferent growth', () => {
  const lab = new Lab({
    seed: 3,
    sim: { pulseFireProb: 0, backgroundHz: 0 },
    walk: { count: 0 },
  });
  lab.runEpochs(6);
  const growthEvents = lab.dev.events.filter(
    (e) => (e.type === 'NeuronBorn' && e.role === 'excitatory') || e.type === 'SynapseGrown'
  );
  const inhibitoryBirths = lab.dev.events.filter(
    (e) => e.type === 'NeuronBorn' && e.role === 'inhibitory'
  );
  assert.ok(growthEvents.length > 0, 'silence should trigger excitatory/afferent growth');
  assert.equal(inhibitoryBirths.length, 0, 'silence should not trigger inhibitory growth');
});

test('region subdivision occurs and preserves members, floors and sector partition', () => {
  // encourage divisions: low threshold
  const lab = new Lab({ seed: 42, dev: { subdivideSize: 10, subdivideProb: 1 } });
  lab.runEpochs(30);
  assert.ok(lab.dev.subdividedTotal > 0, 'expected at least one region division');
  for (const ev of lab.dev.events) {
    if (ev.type !== 'RegionExpanded') continue;
    const parent = lab.graph.regions.get(ev.region);
    assert.equal(parent.kind, 'branch', 'divided region becomes a branch');
    const kids = ev.children.map((p) => lab.graph.regions.get(p));
    for (const k of kids) {
      assert.ok(k, 'child region exists');
      assert.equal(k.depth, parent.depth + 1);
      assert.ok(k.a0 >= parent.a0 - 1e-9 && k.a1 <= parent.a1 + 1e-9, 'children stay in parent sector');
    }
  }
  // every neuron belongs to the region it claims
  for (const n of lab.graph.neurons.values()) {
    const region = lab.graph.regions.get(n.region);
    assert.ok(region && region.members.has(n.id), `neuron ${n.id} region mismatch`);
  }
});

test('walkers are deterministic and deposit survival energy', () => {
  const visits = (seed) => {
    const lab = new Lab({ seed, walk: { count: 2 } });
    const path = [];
    lab.walkers.onNote = (n) => path.push(n.id);
    lab.runEpochs(2);
    return path.join(',');
  };
  assert.equal(visits(50), visits(50), 'same seed → same walker melody');
  assert.notEqual(visits(50), visits(51), 'different seed → different melody');

  // energy feedback: a walked network keeps more energy than an unwalked one
  const walked = new Lab({ seed: 8, walk: { count: 4 }, sim: { developmentEnabled: false } });
  const unwalked = new Lab({ seed: 8, walk: { count: 0 }, sim: { developmentEnabled: false } });
  walked.runEpochs(8);
  unwalked.runEpochs(8);
  const meanEnergy = (lab) => {
    let s = 0;
    let c = 0;
    for (const n of lab.graph.neurons.values()) {
      if (n.role !== 'input') {
        s += n.energy;
        c++;
      }
    }
    return s / c;
  };
  assert.ok(
    meanEnergy(walked) > meanEnergy(unwalked),
    'walker visits should raise survival energy'
  );
});

test('walker-occupied neurons are protected from pruning', () => {
  const lab = new Lab({ seed: 13, walk: { count: 3 } });
  for (let e = 0; e < 20; e++) {
    const occupied = lab.walkers.occupiedIds();
    lab.runEpochs(1);
    for (const id of lab.lastEpochChanges.pruned) {
      assert.ok(!occupied.has(id), `pruned neuron ${id} was walker-occupied`);
    }
  }
});

test('branchOut sprouts new populations recursively and keeps the graph valid', () => {
  const lab = new Lab({ seed: 21 });
  const before = lab.report();
  const ch = lab.branchOut(2);
  const after = lab.report();
  assert.ok(ch.sprouted.length >= 1, 'expected at least one sprouted region');
  assert.ok(ch.born.length > 0, 'sprouting must create fresh neurons');
  assert.ok(after.neurons > before.neurons);
  assert.ok(after.leafRegions > before.leafRegions, 'fan-out adds leaf regions');
  // moved + new neurons all claim regions that claim them back
  for (const n of lab.graph.neurons.values()) {
    const region = lab.graph.regions.get(n.region);
    assert.ok(region && region.members.has(n.id), `neuron ${n.id} region mismatch`);
  }
  for (const s of lab.graph.synapses.values()) {
    assert.ok(lab.graph.neurons.has(s.source) && lab.graph.neurons.has(s.target));
  }
  // budget respected even when hammered
  for (let i = 0; i < 40; i++) lab.branchOut(3);
  assert.ok(lab.graph.neurons.size <= lab.dev.params.maxNeurons, 'branch respects neuron budget');
  // sim still runs after heavy manual branching
  lab.runEpochs(2);
  assert.ok(lab.report().neurons > 0);
});

test('modulator nodes exist but are rare', () => {
  let modulators = 0;
  let excitatory = 0;
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const lab = new Lab({ seed });
    for (const n of lab.graph.neurons.values()) {
      if (n.role !== 'excitatory') continue;
      excitatory++;
      if (n.isModulator) modulators++;
    }
  }
  assert.ok(modulators > 0, 'some organisms should carry modulator nodes');
  assert.ok(modulators / excitatory < 0.15, `modulators should be rare, got ${modulators}/${excitatory}`);
});

test('modulator spikes move the key along the circle of fifths (both rules)', () => {
  const lab = new Lab({
    seed: 4,
    grammar: { modulatorFraction: 0.6 },
    sim: { modProb: 1, modCooldownMs: 50 },
  });
  lab.runEpochs(6);
  assert.ok(lab.keyChangesTotal > 3, `expected many forced key changes, got ${lab.keyChangesTotal}`);
  assert.ok(lab.key.fifths >= 0 && lab.key.fifths < 12);
  assert.equal(lab.key.offset, (7 * lab.key.fifths) % 12, 'offset consistent with circle position');
  const rules = new Set(
    lab.dev.events.filter((e) => e.type === 'KeyChanged').map((e) => e.rule)
  );
  const adjacent = ['up a fifth', 'down a fourth'].some((r) => rules.has(r));
  assert.ok(adjacent, `expected adjacent-key rule to occur, saw ${[...rules]}`);
});

test('key changes are rare and deterministic under default bias', () => {
  const run = (seed) => {
    const lab = new Lab({ seed });
    lab.runEpochs(20);
    return lab;
  };
  const a = run(42);
  const b = run(42);
  assert.equal(a.keyChangesTotal, b.keyChangesTotal, 'key journey reproduces with same seed');
  assert.equal(a.key.fifths, b.key.fifths);
  // 40 simulated seconds with an 8 s cooldown caps changes at 5; bias keeps it lower
  assert.ok(a.keyChangesTotal <= 5, `too many key changes for "biased quite low": ${a.keyChangesTotal}`);
});

test('STDP potentiates correlated pathways and stays within bounds', () => {
  const lab = new Lab({ seed: 42, sim: { stdpEnabled: true, developmentEnabled: false } });
  const before = new Map(
    [...lab.graph.synapses.values()].filter((s) => s.weight > 0).map((s) => [s.id, s.weight])
  );
  lab.runEpochs(6);
  let changed = 0;
  for (const s of lab.graph.synapses.values()) {
    if (s.weight <= 0) continue;
    assert.ok(s.weight >= 0.05 - 1e-9 && s.weight <= 1.1 + 1e-9, `weight out of bounds: ${s.weight}`);
    const b = before.get(s.id);
    if (b !== undefined && Math.abs(s.weight - b) > 1e-6) changed++;
  }
  assert.ok(changed > 10, `expected many weights to move under STDP, got ${changed}`);
  // inhibitory weights untouched
  for (const s of lab.graph.synapses.values()) {
    if (s.weight < 0) assert.ok(s.weight >= -2, 'inhibitory weights remain negative and sane');
  }
});

test('STDP off leaves weights frozen; on/off both deterministic', () => {
  const weightsAfter = (stdpEnabled) => {
    const lab = new Lab({ seed: 8, sim: { stdpEnabled, developmentEnabled: false } });
    lab.runEpochs(3);
    return [...lab.graph.synapses.values()].map((s) => s.weight.toFixed(9)).join(',');
  };
  assert.equal(weightsAfter(false), weightsAfter(false));
  assert.equal(weightsAfter(true), weightsAfter(true));
  assert.notEqual(weightsAfter(false), weightsAfter(true));
});

test('drive patterns change the rhythm deterministically', () => {
  const pulses = (drivePattern) => {
    const lab = new Lab({ seed: 6, sim: { drivePattern, developmentEnabled: false } });
    lab.runEpochs(3);
    return lab.pulseCount;
  };
  const steady = pulses('steady');
  const sparse = pulses('sparse');
  const bursts = pulses('bursts');
  assert.ok(sparse < steady, `sparse (${sparse}) should pulse less than steady (${steady})`);
  assert.ok(bursts > sparse, 'bursts should pulse more than sparse');
  assert.equal(pulses('euclidean'), pulses('euclidean'), 'deterministic');
});

test('attractor biases walker traversal toward a location', () => {
  // stub positions: put region R.0's neurons at (0,0), everything else at (1,1),
  // park the attractor at (0,0) and check visits concentrate there
  const visitShare = (useAttractor) => {
    const lab = new Lab({ seed: 3, walk: { count: 2, variation: 0.9 }, sim: { developmentEnabled: false } });
    const nearIds = new Set(
      [...lab.graph.neurons.values()].filter((n) => n.region.startsWith('R.0')).map((n) => n.id)
    );
    lab.walkers.posOf = (id) => (nearIds.has(id) ? { x: 0, y: 0 } : { x: 1, y: 1 });
    if (useAttractor) lab.walkers.attractor = { x: 0, y: 0, strength: 8 };
    let near = 0;
    let total = 0;
    lab.walkers.onNote = (n) => {
      total++;
      if (nearIds.has(n.id)) near++;
    };
    lab.runEpochs(4);
    return total ? near / total : 0;
  };
  const withAttractor = visitShare(true);
  const without = visitShare(false);
  assert.ok(
    withAttractor > without + 0.1,
    `attractor should concentrate visits: with=${withAttractor.toFixed(2)} without=${without.toFixed(2)}`
  );
});

test('long-running developmental simulation stays bounded and alive', () => {
  const lab = new Lab({ seed: 42 });
  lab.runEpochs(60); // 2 simulated minutes
  const r = lab.report();
  assert.ok(r.neurons <= lab.dev.params.maxNeurons);
  assert.ok(r.neurons >= 20, 'network should not collapse to nothing');
  assert.ok(lab.dev.events.length <= lab.dev.maxEvents, 'event history is bounded');
  assert.ok(lab.activity.rateHistory.length <= lab.activity.maxHistory, 'rate history is bounded');
});
