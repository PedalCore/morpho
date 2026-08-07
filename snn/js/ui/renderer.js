// Network renderer. Layout mirrors the developmental hierarchy: each leaf
// region occupies an angular sector (recursively subdivided by the region
// path), the input population sits at the center. Purely cosmetic — reads sim
// state, never writes it.

const COLORS = {
  surface: '#12151b',
  excitatory: '#e2a13f',
  inhibitory: '#5b8def',
  input: '#8a8f98',
  outputRing: '#e9e4d6',
  spike: '#ffffff',
  birth: '#7bd8a5',
  prune: '#d96c6c',
  modulator: '#ffd75e',
  synapseE: 'rgba(226,161,63,0.10)',
  synapseI: 'rgba(91,141,239,0.13)',
  regionLabel: 'rgba(160,166,178,0.45)',
};

function hashJitter(id, salt) {
  // deterministic per-id jitter so layout is stable across frames & reloads
  let h = (id * 2654435761 + salt * 40503) >>> 0;
  h = ((h ^ (h >>> 13)) * 0x5bd1e995) >>> 0;
  return ((h & 0xffff) / 0xffff) * 2 - 1;
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.positions = new Map(); // neuronId -> {x, y} in unit space
    this.births = []; // {id, x, y, at}
    this.deaths = []; // {x, y, at}
    this.keyFlashes = []; // {x, y, at} — a modulator changed the key
    this.now = 0; // sim step, set each frame
  }

  regionCenter(graph, path) {
    // regions carry their own angular sector (subdivided as they divide);
    // radius grows with developmental depth, so deep = far out AND high pitch
    if (path === 'IN') return { x: 0.5, y: 0.5, spread: 0.045 };
    const region = graph.regions.get(path);
    if (!region) return { x: 0.5, y: 0.5, spread: 0.05 };
    const angle = (region.a0 + region.a1) / 2 - Math.PI / 2;
    const radius = 0.1 + 0.075 * Math.min(region.depth, 5);
    return {
      x: 0.5 + Math.cos(angle) * radius,
      y: 0.5 + Math.sin(angle) * radius,
      spread: Math.max(0.035, 0.085 - 0.01 * region.depth),
    };
  }

  // region subdivision moved these neurons — re-place them in their new home
  invalidatePositions(ids) {
    for (const id of ids) this.positions.delete(id);
  }

  positionOf(graph, n) {
    let p = this.positions.get(n.id);
    if (!p) {
      const c = this.regionCenter(graph, n.region);
      p = {
        x: c.x + hashJitter(n.id, 1) * c.spread,
        y: c.y + hashJitter(n.id, 2) * c.spread,
      };
      this.positions.set(n.id, p);
    }
    return p;
  }

  markBirth(graph, id) {
    const n = graph.neurons.get(id);
    if (!n) return;
    const p = this.positionOf(graph, n);
    this.births.push({ id, x: p.x, y: p.y, at: this.now });
  }

  markDeath(id) {
    const p = this.positions.get(id);
    if (p) this.deaths.push({ x: p.x, y: p.y, at: this.now });
    this.positions.delete(id);
  }

  markKeyChange(graph, id) {
    const n = graph.neurons.get(id);
    if (!n) return;
    const p = this.positionOf(graph, n);
    this.keyFlashes.push({ x: p.x, y: p.y, at: this.now });
  }

  panOf(graph, n) {
    return this.positionOf(graph, n).x * 2 - 1;
  }

  draw(graph, engine, walkers = null) {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    const S = Math.min(w, h);
    const ox = (w - S) / 2;
    const oy = (h - S) / 2;
    const px = (p) => ({ x: ox + p.x * S, y: oy + p.y * S });
    this.view = { S, ox, oy }; // for canvas→unit inverse mapping (attractor)
    this.now = engine.stepCount;

    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, w, h);

    // synapses
    ctx.lineWidth = 1;
    for (const s of graph.synapses.values()) {
      const a = graph.neurons.get(s.source);
      const b = graph.neurons.get(s.target);
      if (!a || !b) continue;
      const pa = px(this.positionOf(graph, a));
      const pb = px(this.positionOf(graph, b));
      ctx.strokeStyle = s.weight >= 0 ? COLORS.synapseE : COLORS.synapseI;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      const mx = (pa.x + pb.x) / 2 + (pb.y - pa.y) * 0.12;
      const my = (pa.y + pb.y) / 2 - (pb.x - pa.x) * 0.12;
      ctx.quadraticCurveTo(mx, my, pb.x, pb.y);
      ctx.stroke();
    }

    // neurons
    for (const n of graph.neurons.values()) {
      const p = px(this.positionOf(graph, n));
      const base =
        n.role === 'input' ? COLORS.input : n.role === 'inhibitory' ? COLORS.inhibitory : COLORS.excitatory;
      const r = n.isOutput ? 5.5 : n.role === 'input' ? 3.5 : 4.5;

      // spike glow (fast timescale)
      const since = this.now - n.lastSpikeStep;
      if (n.lastSpikeStep >= 0 && since < 140) {
        const glow = 1 - since / 140;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 7 * glow, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${0.28 * glow})`;
        ctx.fill();
      }

      ctx.beginPath();
      if (n.role === 'inhibitory') {
        // square = inhibitory (shape encodes identity, not just color)
        ctx.rect(p.x - r, p.y - r, r * 2, r * 2);
      } else if (n.isModulator) {
        // gold diamond = key-changer node
        ctx.moveTo(p.x, p.y - r - 1.5);
        ctx.lineTo(p.x + r + 1.5, p.y);
        ctx.lineTo(p.x, p.y + r + 1.5);
        ctx.lineTo(p.x - r - 1.5, p.y);
        ctx.closePath();
      } else {
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      }
      ctx.fillStyle =
        since < 25 && n.lastSpikeStep >= 0 ? COLORS.spike : n.isModulator ? COLORS.modulator : base;
      ctx.fill();

      if (n.isOutput) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 2.5, 0, Math.PI * 2);
        ctx.strokeStyle = COLORS.outputRing;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
    }

    // development events (slow timescale): birth rings, death fades
    const DUR = 900;
    this.births = this.births.filter((b) => this.now - b.at < DUR);
    for (const b of this.births) {
      const t = (this.now - b.at) / DUR;
      const p = px(b);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6 + t * 26, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(123,216,165,${0.7 * (1 - t)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    this.deaths = this.deaths.filter((d) => this.now - d.at < DUR);
    for (const d of this.deaths) {
      const t = (this.now - d.at) / DUR;
      const p = px(d);
      ctx.strokeStyle = `rgba(217,108,108,${0.8 * (1 - t)})`;
      ctx.lineWidth = 1.6;
      const r = 4 + t * 6;
      ctx.beginPath();
      ctx.moveTo(p.x - r, p.y - r);
      ctx.lineTo(p.x + r, p.y + r);
      ctx.moveTo(p.x + r, p.y - r);
      ctx.lineTo(p.x - r, p.y + r);
      ctx.stroke();
    }

    // key-change flashes: broad gold rings rippling from the modulator
    const KEYDUR = 1500;
    this.keyFlashes = this.keyFlashes.filter((k) => this.now - k.at < KEYDUR);
    for (const k of this.keyFlashes) {
      const t = (this.now - k.at) / KEYDUR;
      const p = px(k);
      for (const lag of [0, 0.18]) {
        const tt = Math.max(0, t - lag);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8 + tt * 90, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,215,94,${0.55 * (1 - tt)})`;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }

    // steering attractor: crosshair where the performer is pulling walkers
    if (walkers && walkers.attractor) {
      const p = px(walkers.attractor);
      ctx.strokeStyle = 'rgba(190,140,255,0.6)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(p.x - 20, p.y);
      ctx.lineTo(p.x + 20, p.y);
      ctx.moveTo(p.x, p.y - 20);
      ctx.lineTo(p.x, p.y + 20);
      ctx.stroke();
    }

    // walkers: bright roaming melodic agents
    if (walkers) {
      for (const w of walkers.walkers) {
        const n = w.at != null ? graph.neurons.get(w.at) : null;
        if (!n) continue;
        const p = px(this.positionOf(graph, n));
        ctx.beginPath();
        ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(190,140,255,0.9)';
        ctx.lineWidth = 2;
        ctx.stroke();
        // short trail through recent history
        ctx.strokeStyle = 'rgba(190,140,255,0.25)';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        let started = false;
        for (const id of w.history) {
          const hn = graph.neurons.get(id);
          if (!hn) continue;
          const hp = px(this.positionOf(graph, hn));
          if (!started) {
            ctx.moveTo(hp.x, hp.y);
            started = true;
          } else ctx.lineTo(hp.x, hp.y);
        }
        ctx.stroke();
      }
    }

    // region labels
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = COLORS.regionLabel;
    for (const region of graph.leafRegions()) {
      const c = this.regionCenter(graph, region.path);
      const p = px(c);
      ctx.fillText(region.path, p.x - 12, p.y - S * 0.075 - 6);
    }
  }
}

// Single-series sparkline of network mean firing rate, with the homeostatic
// target band shaded and birth/prune epochs tick-marked.
export function drawRateSparkline(canvas, history, band, epochMarks) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!history.length) return;

  const maxRate = Math.max(...history, band[1] * 1.4, 1);
  const x = (i) => (i / Math.max(history.length - 1, 1)) * (w - 8) + 4;
  const y = (v) => h - 4 - (v / maxRate) * (h - 10);

  // homeostatic band (recessive)
  ctx.fillStyle = 'rgba(140,150,165,0.10)';
  ctx.fillRect(0, y(band[1]), w, Math.max(1, y(band[0]) - y(band[1])));

  // birth / prune epoch ticks along the baseline
  for (const m of epochMarks) {
    const mx = x(m.i);
    if (m.born) {
      ctx.strokeStyle = 'rgba(123,216,165,0.8)';
      ctx.beginPath();
      ctx.moveTo(mx, h - 2);
      ctx.lineTo(mx, h - 8);
      ctx.stroke();
    }
    if (m.pruned) {
      ctx.strokeStyle = 'rgba(217,108,108,0.8)';
      ctx.beginPath();
      ctx.moveTo(mx, 2);
      ctx.lineTo(mx, 8);
      ctx.stroke();
    }
  }

  ctx.strokeStyle = '#c9cdd6';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  history.forEach((v, i) => {
    if (i === 0) ctx.moveTo(x(i), y(v));
    else ctx.lineTo(x(i), y(v));
  });
  ctx.stroke();
}
