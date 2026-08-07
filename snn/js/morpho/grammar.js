// "Morpho-lite" recursive developmental grammar.
//
// This is not a parser for the real Morpho language — it is the developmental
// *semantics* the brief asks us to test: a compact recursive rule expands into
// a much larger operational SNN. A real Morpho front-end could later compile
// down to exactly these grow calls (see NOTES in EXPERIMENT.md).
//
// Grammar (recursive in space, stochastic per the SNN brief §23):
//   Region(depth > 0)  -> 2–3 × Region(depth - 1), with a per-branch chance
//                         of terminating early — so leaf regions land at
//                         *different* depths, and depth carries register
//   Region(depth == 0) -> leaf population (variable size): excitatory pool +
//                         inhibitory regulators, locally recurrent
//   siblings           -> sparse long-delay excitatory projections
//   InputPopulation    -> wired into a subset of leaf regions
//
// Pitch is structural and fixed at birth (plugin briefs §12):
//   region depth            → octave / register
//   region sector + birth # → scale degree
// Development can later SUBDIVIDE a leaf (Morpho cell division during
// lifetime): children split the parent's angular sector, so new structure
// grows into new registers.

import { randInt, pick } from '../core/rng.js';

export const DEFAULT_GRAMMAR = {
  depth: 3,
  terminationProb: 0.3, // per-branch chance to stop recursing early
  leafExcitatory: [5, 10], // range
  leafInhibitory: [1, 3],
  localConnectProb: 0.35,
  excitatoryWeight: [0.35, 0.65],
  inhibitoryWeight: [-0.9, -0.5],
  localDelayMs: [1, 6],
  longRangeCount: 3, // per sibling pair
  longRangeWeight: [0.3, 0.55],
  longRangeDelayMs: [12, 45], // long delays = audible echo structure
  inputNeurons: 6,
  inputFanout: 4, // synapses per input neuron
  inputWeight: [0.5, 0.9],
  outputFraction: 0.4,
  modulatorFraction: 0.05, // rare key-changer nodes (non-output excitatory)
  maxOctave: 5,
};

function range(rng, [lo, hi]) {
  return lo + rng() * (hi - lo);
}

// depth → register, sector + structural birth position → degree.
export function assignPitch(region, params) {
  return {
    octave: Math.max(1, Math.min(params.maxOctave, region.depth)),
    structDegree: region.degreeOffset + region.bornCount,
  };
}

export function makeNeuron(graph, region, role, params, rng, epoch) {
  region.bornCount++;
  const isOutput = role === 'excitatory' && rng() < params.outputFraction;
  const isModulator = role === 'excitatory' && !isOutput && rng() < params.modulatorFraction;
  const pitch = assignPitch(region, params);
  return graph.addNeuron({
    role,
    region: region.path,
    isOutput,
    isModulator,
    octave: pitch.octave,
    structDegree: pitch.structDegree,
    bornEpoch: epoch,
  });
}

// Wire one neuron the way the grammar prescribes. Also used by the
// development layer when it grows single neurons into an existing region, so
// grown structure is "the same anatomy" the genotype would have produced.
export function wireNeuronIntoRegion(graph, neuron, region, params, rng) {
  const members = [...region.members]
    .map((id) => graph.neurons.get(id))
    .filter((n) => n && n.id !== neuron.id);
  for (const other of members) {
    if (neuron.role !== 'inhibitory' && rng() < params.localConnectProb) {
      graph.addSynapse({
        source: neuron.id,
        target: other.id,
        weight: range(rng, params.excitatoryWeight),
        delaySteps: randInt(rng, params.localDelayMs[0], params.localDelayMs[1] + 1),
      });
    }
    if (neuron.role === 'inhibitory' && other.role !== 'inhibitory' && rng() < params.localConnectProb * 1.6) {
      graph.addSynapse({
        source: neuron.id,
        target: other.id,
        weight: range(rng, params.inhibitoryWeight),
        delaySteps: randInt(rng, params.localDelayMs[0], params.localDelayMs[1] + 1),
      });
    }
    // reciprocal: existing excitatory members may project onto the newcomer
    if (other.role !== 'inhibitory' && neuron.role !== 'input' && rng() < params.localConnectProb) {
      graph.addSynapse({
        source: other.id,
        target: neuron.id,
        weight: other.role === 'inhibitory' ? range(rng, params.inhibitoryWeight) : range(rng, params.excitatoryWeight),
        delaySteps: randInt(rng, params.localDelayMs[0], params.localDelayMs[1] + 1),
      });
    }
  }
}

function growLeaf(graph, path, parentPath, a0, a1, params, rng, epoch) {
  const depth = path.split('.').length - 1;
  const region = graph.addRegion(path, depth, 'leaf', parentPath, a0, a1);
  const born = [];
  const nE = randInt(rng, params.leafExcitatory[0], params.leafExcitatory[1] + 1);
  const nI = randInt(rng, params.leafInhibitory[0], params.leafInhibitory[1] + 1);
  for (let i = 0; i < nE; i++) born.push(makeNeuron(graph, region, 'excitatory', params, rng, epoch));
  for (let i = 0; i < nI; i++) born.push(makeNeuron(graph, region, 'inhibitory', params, rng, epoch));
  for (const n of born) wireNeuronIntoRegion(graph, n, region, params, rng);
  return region;
}

function growRegion(graph, path, parentPath, a0, a1, depth, params, rng, epoch) {
  const isRoot = path === 'R';
  if (depth === 0 || (!isRoot && rng() < params.terminationProb)) {
    return [growLeaf(graph, path, parentPath, a0, a1, params, rng, epoch)];
  }
  graph.addRegion(path, path.split('.').length - 1, 'branch', parentPath, a0, a1);
  const branching = randInt(rng, 2, 4); // 2 or 3 children
  const leaves = [];
  const childLeafSets = [];
  for (let b = 0; b < branching; b++) {
    const span = (a1 - a0) / branching;
    const childLeaves = growRegion(
      graph,
      `${path}.${b}`,
      path,
      a0 + span * b,
      a0 + span * (b + 1),
      depth - 1,
      params,
      rng,
      epoch
    );
    childLeafSets.push(childLeaves);
    leaves.push(...childLeaves);
  }
  // sparse long-range projections between sibling subtrees (with long delays)
  for (let a = 0; a < childLeafSets.length; a++) {
    for (let b = 0; b < childLeafSets.length; b++) {
      if (a === b) continue;
      for (let k = 0; k < params.longRangeCount; k++) {
        const srcRegion = pick(rng, childLeafSets[a]);
        const dstRegion = pick(rng, childLeafSets[b]);
        const srcCandidates = [...srcRegion.members].filter(
          (id) => graph.neurons.get(id).role === 'excitatory'
        );
        const dstCandidates = [...dstRegion.members];
        if (!srcCandidates.length || !dstCandidates.length) continue;
        graph.addSynapse({
          source: pick(rng, srcCandidates),
          target: pick(rng, dstCandidates),
          weight: range(rng, params.longRangeWeight),
          delaySteps: randInt(rng, params.longRangeDelayMs[0], params.longRangeDelayMs[1] + 1),
        });
      }
    }
  }
  return leaves;
}

// Expand the whole genotype into a phenotype graph.
export function growNetwork(graph, params, rng, epoch = 0) {
  const leaves = growRegion(graph, 'R', null, 0, Math.PI * 2, params.depth, params, rng, epoch);

  // input population, wired into a subset of leaf regions
  const inputRegion = graph.addRegion('IN', 0, 'input', null);
  for (let i = 0; i < params.inputNeurons; i++) {
    const input = graph.addNeuron({ role: 'input', region: 'IN', bornEpoch: epoch });
    for (let f = 0; f < params.inputFanout; f++) {
      const leaf = pick(rng, leaves);
      const targets = [...leaf.members].filter((id) => graph.neurons.get(id).role === 'excitatory');
      if (!targets.length) continue;
      graph.addSynapse({
        source: input.id,
        target: pick(rng, targets),
        weight: range(rng, params.inputWeight),
        delaySteps: randInt(rng, 1, 4),
      });
    }
  }
  return { leaves, inputRegion };
}

// Recursive fan-out on demand: a leaf becomes a branch, its existing
// population moves into child .0, and 1–2 brand-new leaf populations sprout
// beside it (fresh neurons, one register deeper), wired to the old population
// with long-delay projections in both directions. This is the grammar
// recursing again during lifetime — new anatomy, not just re-partitioning.
export function sproutRegion(graph, region, params, rng, epoch) {
  const kids = randInt(rng, 2, 4); // 2–3 children total
  const span = (region.a1 - region.a0) / kids;
  const child0 = graph.addRegion(
    `${region.path}.0`,
    region.depth + 1,
    'leaf',
    region.path,
    region.a0,
    region.a0 + span
  );
  const movedIds = [...region.members].sort((x, y) => x - y);
  for (const id of movedIds) {
    const n = graph.neurons.get(id);
    region.members.delete(id);
    child0.members.add(id);
    child0.bornCount++;
    n.region = child0.path; // birth-fixed pitch retained
  }
  const newLeaves = [];
  for (let b = 1; b < kids; b++) {
    newLeaves.push(
      growLeaf(
        graph,
        `${region.path}.${b}`,
        region.path,
        region.a0 + span * b,
        region.a0 + span * (b + 1),
        params,
        rng,
        epoch
      )
    );
  }
  region.kind = 'branch';
  // long-range wiring between the old population and each new sibling
  for (const leaf of newLeaves) {
    for (const [src, dst] of [
      [child0, leaf],
      [leaf, child0],
    ]) {
      for (let k = 0; k < params.longRangeCount; k++) {
        const sources = [...src.members].filter((id) => graph.neurons.get(id).role === 'excitatory');
        const targets = [...dst.members];
        if (!sources.length || !targets.length) continue;
        graph.addSynapse({
          source: pick(rng, sources),
          target: pick(rng, targets),
          weight: range(rng, params.longRangeWeight),
          delaySteps: randInt(rng, params.longRangeDelayMs[0], params.longRangeDelayMs[1] + 1),
        });
      }
    }
  }
  return {
    children: [child0, ...newLeaves],
    movedIds,
    bornIds: newLeaves.flatMap((l) => [...l.members]),
  };
}

// Morpho cell division during lifetime: a large leaf splits into two child
// leaves, each inheriting half the members and half the angular sector.
// Existing synapses are untouched (they now span the children — the old local
// circuit becomes the new long-range scaffolding). Newborns in the children
// land one register higher: development literally grows upward in pitch.
export function subdivideRegion(graph, region, rng) {
  const mid = (region.a0 + region.a1) / 2;
  const children = [
    graph.addRegion(`${region.path}.0`, region.depth + 1, 'leaf', region.path, region.a0, mid),
    graph.addRegion(`${region.path}.1`, region.depth + 1, 'leaf', region.path, mid, region.a1),
  ];
  const members = [...region.members].sort((x, y) => x - y); // deterministic
  members.forEach((id, i) => {
    const child = children[i % 2];
    const n = graph.neurons.get(id);
    region.members.delete(id);
    child.members.add(id);
    child.bornCount++;
    n.region = child.path;
    // birth-fixed pitch is retained — continuity for existing voices
  });
  region.kind = 'branch';
  return children;
}
