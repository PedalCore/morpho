// Slow activity statistics, updated once per development epoch — deliberately
// separate from instantaneous membrane state.

export class ActivityTracker {
  constructor({ emaAlpha = 0.6 } = {}) {
    this.emaAlpha = emaAlpha; // weight on previous EMA
    this.epoch = 0;
    this.networkRateHz = 0;
    this.rateHistory = []; // bounded, for the UI sparkline
    this.maxHistory = 240;
  }

  // Called at each epoch boundary. epochMs = wall duration of simulated epoch.
  update(graph, epochMs) {
    this.epoch++;
    let totalSpikes = 0;
    let counted = 0;
    for (const n of graph.neurons.values()) {
      const rateHz = (n.spikeCount * 1000) / epochMs;
      n.activityEMA = this.emaAlpha * n.activityEMA + (1 - this.emaAlpha) * rateHz;
      n.spikeCount = 0;
      // survival energy: decays each epoch, restored by firing and by walker
      // visits (musical use). Silent, unplayed neurons starve in ~6 epochs.
      const rateNorm = Math.min(rateHz / 8, 1);
      const visitNorm = Math.min(n.walkerVisits / 6, 1);
      n.energy = Math.min(1.5, n.energy * 0.7 + 0.3 * rateNorm + 0.35 * visitNorm);
      n.walkerVisits = 0;
      if (n.role !== 'input') {
        totalSpikes += rateHz;
        counted++;
      }
    }
    this.networkRateHz = counted ? totalSpikes / counted : 0;
    this.rateHistory.push(this.networkRateHz);
    if (this.rateHistory.length > this.maxHistory) this.rateHistory.shift();
  }

  regionMeanRate(graph, region) {
    let sum = 0;
    let count = 0;
    for (const id of region.members) {
      const n = graph.neurons.get(id);
      if (n && n.role !== 'input') {
        sum += n.activityEMA;
        count++;
      }
    }
    return count ? sum / count : 0;
  }
}
