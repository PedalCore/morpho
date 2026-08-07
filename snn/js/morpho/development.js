// Development feedback layer (slow timescale). Reads activity statistics and
// survival energy, returns explicit loggable decisions, then applies them at
// the epoch boundary — never during spike processing.
//
// Rules (simple + inspectable):
//   homeostatic growth   — leaf firing below target band → grow excitatory
//                          neuron OR a long-range afferent from an active
//                          region (repair, not just local churn); above band
//                          → grow inhibitory neuron
//   energy survival      — energy decays each epoch, restored by spiking and
//                          by walker visits (music keeps structure alive);
//                          old low-energy neurons are pruned. Neurons a
//                          walker currently occupies are protected.
//   region subdivision   — a large healthy leaf divides into two child
//                          regions (Morpho cell division during lifetime);
//                          newborns there sound one register higher
//   bounded growth       — maxNeurons / maxSynapses are hard budgets

import { wireNeuronIntoRegion, makeNeuron, subdivideRegion } from './grammar.js';
import { randInt, pick } from '../core/rng.js';

export const DEFAULT_DEV_PARAMS = {
  maxNeurons: 260,
  maxSynapses: 4000,
  minPerLeafRegion: 4,
  lowRateHz: 1.5, // below → excitatory growth pressure
  highRateHz: 12, // above → inhibitory growth pressure
  pruneEnergy: 0.13, // energy below this → prune candidate
  minAgeEpochs: 4, // protect newborns
  growProb: 0.7, // stochastic development
  afferentProb: 0.5, // silent region: chance the fix is a long-range afferent
  maxGrowPerEpoch: 3,
  maxPrunePerEpoch: 3,
  subdivideSize: 12, // leaf member count that triggers possible division
  subdivideProb: 0.35,
  maxDepth: 5,
};

export class DevelopmentController {
  constructor(params = {}) {
    this.params = { ...DEFAULT_DEV_PARAMS, ...params };
    this.events = []; // bounded developmental history
    this.maxEvents = 200;
    this.grownTotal = 0;
    this.prunedTotal = 0;
    this.subdividedTotal = 0;
  }

  log(event) {
    this.events.push(event);
    if (this.events.length > this.maxEvents) this.events.shift();
  }

  evaluate(graph, activity, epoch, rng, protectedIds = new Set()) {
    const p = this.params;
    const decisions = [];

    // --- pruning: energy-based survival (activity + musical use) ---
    const candidates = [];
    for (const n of graph.neurons.values()) {
      if (n.role === 'input') continue;
      if (epoch - n.bornEpoch < p.minAgeEpochs) continue;
      if (n.energy >= p.pruneEnergy) continue;
      if (protectedIds.has(n.id)) continue; // walker is sitting on it
      const region = graph.regions.get(n.region);
      if (region && region.members.size <= p.minPerLeafRegion) continue;
      candidates.push(n);
    }
    candidates.sort((a, b) => a.energy - b.energy);
    for (const n of candidates.slice(0, p.maxPrunePerEpoch)) {
      decisions.push({ type: 'prune', id: n.id, region: n.region, energy: n.energy, rate: n.activityEMA });
    }

    // --- growth: homeostatic morphogenesis + long-range repair ---
    let grown = 0;
    for (const region of graph.leafRegions()) {
      if (grown >= p.maxGrowPerEpoch) break;
      if (graph.neurons.size >= p.maxNeurons) break;
      if (graph.synapses.size >= p.maxSynapses) break;
      const rate = activity.regionMeanRate(graph, region);
      if (rate < p.lowRateHz && rng() < p.growProb) {
        if (rng() < p.afferentProb) {
          decisions.push({ type: 'afferent', region: region.path, rate });
        } else {
          decisions.push({ type: 'grow', region: region.path, role: 'excitatory', rate });
        }
        grown++;
      } else if (rate > p.highRateHz && rng() < p.growProb) {
        decisions.push({ type: 'grow', region: region.path, role: 'inhibitory', rate });
        grown++;
      }
    }

    // --- structural elaboration: one region division per epoch at most ---
    if (graph.neurons.size < p.maxNeurons) {
      for (const region of graph.leafRegions()) {
        if (
          region.members.size >= Math.max(p.subdivideSize, 2 * p.minPerLeafRegion + 2) &&
          region.depth < p.maxDepth &&
          rng() < p.subdivideProb
        ) {
          decisions.push({ type: 'subdivide', region: region.path, size: region.members.size });
          break;
        }
      }
    }
    return decisions;
  }

  apply(graph, decisions, epoch, rng, grammarParams) {
    const p = this.params;
    const born = [];
    const pruned = [];
    const subdivided = [];
    for (const d of decisions) {
      if (d.type === 'prune') {
        const n = graph.neurons.get(d.id);
        if (!n) continue;
        // re-check the floor at apply time — several prunes in one epoch may
        // target the same region, and each passed the check individually
        const region = graph.regions.get(n.region);
        if (region && region.members.size <= p.minPerLeafRegion) continue;
        graph.removeNeuron(d.id);
        this.prunedTotal++;
        pruned.push(d.id);
        this.log({ epoch, type: 'NeuronPruned', id: d.id, region: d.region, rate: d.rate, energy: d.energy });
      } else if (d.type === 'grow') {
        if (graph.neurons.size >= p.maxNeurons) continue;
        const region = graph.regions.get(d.region);
        if (!region || region.kind !== 'leaf') continue;
        const n = makeNeuron(graph, region, d.role, grammarParams, rng, epoch);
        wireNeuronIntoRegion(graph, n, region, grammarParams, rng);
        this.grownTotal++;
        born.push(n.id);
        this.log({
          epoch,
          type: 'NeuronBorn',
          id: n.id,
          region: d.region,
          role: d.role,
          rate: d.rate,
          isOutput: n.isOutput,
        });
      } else if (d.type === 'afferent') {
        // repair a silent region: project into it from an active neuron
        // elsewhere, instead of growing yet another silent local neuron
        if (graph.synapses.size >= p.maxSynapses) continue;
        const region = graph.regions.get(d.region);
        if (!region) continue;
        const sources = [...graph.neurons.values()].filter(
          (n) => n.role === 'excitatory' && n.region !== d.region && n.activityEMA > 1.0
        );
        const targets = [...region.members]
          .map((id) => graph.neurons.get(id))
          .filter((n) => n && n.role === 'excitatory');
        if (!sources.length || !targets.length) continue;
        const src = pick(rng, sources);
        const dst = pick(rng, targets);
        graph.addSynapse({
          source: src.id,
          target: dst.id,
          weight: 0.4 + rng() * 0.3,
          delaySteps: randInt(rng, 10, 40),
        });
        this.grownTotal++;
        this.log({ epoch, type: 'SynapseGrown', from: src.region, region: d.region, rate: d.rate });
      } else if (d.type === 'subdivide') {
        const region = graph.regions.get(d.region);
        if (!region || region.kind !== 'leaf') continue;
        const children = subdivideRegion(graph, region, rng);
        this.subdividedTotal++;
        subdivided.push({ parent: d.region, children: children.map((c) => c.path) });
        this.log({ epoch, type: 'RegionExpanded', region: d.region, children: children.map((c) => c.path), size: d.size });
      }
    }
    return { born, pruned, subdivided };
  }
}
