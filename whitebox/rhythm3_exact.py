"""M10 v3 exact — nine-voice clocked drum process on a musical-tick lattice.

Representation
--------------
x_t in {0,1}^9 on a host/DAW musical-time lattice. Default resolution is
48 steps per quarter (12 legal positions per 16th); BPM does not enter the
learned representation. At inference the host transport maps these musical
ticks to wall time, so one learned process scales to arbitrary tempo.

Preregistered four-rung decomposition
--------------------------------------
1. clock: transport/playhead only
2. own: clock + 10 fixed causal traces of the predicted voice
3. all: clock + all 9 x 10 voice traces
4. coupling: all + causal within-tick autoregressive voice coupling

Free-running protocol
---------------------
Teacher-force 2 bars, generate 16, and measure separately:
* event-rate stability
* metrical/grid stability
* voice-distribution stability

Usage
-----
python3 -m whitebox.rhythm3_exact --prep --data ~/datasets/groove
python3 -m whitebox.rhythm3_exact --sweep --freerun
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

GMD = pathlib.Path.home() / "datasets" / "groove"
OUT = pathlib.Path("whitebox/runs/rhythm")
NV = 9
CHANNELS = (
    "kick", "snare", "closed_hh", "open_hh", "low_tom",
    "mid_tom", "high_tom", "crash", "ride",
)
# Magenta / GMD canonical 9-class reduction.
VOICE = {
    36: 0,
    38: 1, 37: 1, 40: 1,
    42: 2, 22: 2, 44: 2,
    46: 3, 26: 3,
    43: 4, 58: 4,
    47: 5, 45: 5,
    50: 6, 48: 6,
    49: 7, 52: 7, 55: 7, 57: 7,
    51: 8, 53: 8, 59: 8,
}
RUNGS = ("clock", "own", "all", "coupling")


@dataclass(frozen=True)
class Config:
    steps_per_quarter: int = 48
    seed_bars: int = 2
    generate_bars: int = 16
    trace_half_lives_quarters: Tuple[float, ...] = (
        1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0,
        2.0, 4.0, 8.0, 16.0, 32.0,
    )
    seed: int = 20260829

    @property
    def steps_per_sixteenth(self) -> int:
        if self.steps_per_quarter % 4:
            raise ValueError("steps_per_quarter must be divisible by 4")
        return self.steps_per_quarter // 4

    @property
    def steps_per_bar(self) -> int:
        return 4 * self.steps_per_quarter

    @property
    def clock_dim(self) -> int:
        # 16 coarse sixteenth slots across a 4/4 bar + fine phase inside each 16th.
        return 16 + self.steps_per_sixteenth

    @property
    def traces_per_voice(self) -> int:
        return len(self.trace_half_lives_quarters)


def _split_name(v: object) -> str:
    s = str(v).strip().lower()
    if s in {"validation", "valid", "val", "dev"}:
        return "validation"
    if s in {"training", "train"}:
        return "train"
    if s in {"testing", "test"}:
        return "test"
    return s


def _midi_to_tick_raster(path: pathlib.Path, cfg: Config) -> np.ndarray:
    """Read GMD note-ons and quantize in MIDI musical time, not seconds."""
    import mido

    mid = mido.MidiFile(path)
    merged = mido.merge_tracks(mid.tracks)
    abs_tick = 0
    events: Dict[int, set[int]] = {}
    max_step = 0
    for msg in merged:
        abs_tick += int(msg.time)
        step = int(round(abs_tick * cfg.steps_per_quarter / mid.ticks_per_beat))
        max_step = max(max_step, step)
        if msg.type == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
            v = VOICE.get(int(msg.note))
            if v is not None:
                events.setdefault(step, set()).add(v)
    x = np.zeros((max_step + 1, NV), np.uint8)
    for step, voices in events.items():
        for v in voices:
            x[step, v] = 1
    return x


def prep(data: pathlib.Path = GMD, cfg: Config = Config()) -> None:
    """Prepare official GMD splits; v3 exact is deliberately 4/4 only."""
    rows = list(csv.DictReader(open(data / "info.csv", newline="")))
    store: Dict[str, List[dict]] = {"train": [], "validation": [], "test": []}
    for i, r in enumerate(rows, 1):
        sig = str(r.get("time_signature", ""))
        if sig not in {"4-4", "4/4"}:
            continue
        split = _split_name(r.get("split", "train"))
        if split not in store:
            continue
        path = data / r["midi_filename"]
        try:
            x = _midi_to_tick_raster(path, cfg)
        except Exception as exc:
            print(f"warning: skip {path}: {exc}", flush=True)
            continue
        if int(x.sum()) < 8 or len(x) < cfg.steps_per_bar:
            continue
        store[split].append({
            "x": x,
            "bpm": float(r.get("bpm", 120.0)),
            "midi_filename": r["midi_filename"],
        })
        if i % 100 == 0:
            print(f"prepared {i}/{len(rows)} rows", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "gmd9_tick.npy", np.array(store, dtype=object))
    print({k: len(v) for k, v in store.items()}, flush=True)


def load_store() -> Dict[str, List[dict]]:
    return np.load(OUT / "gmd9_tick.npy", allow_pickle=True).item()


def clock_features(n: int, cfg: Config, start_step: int = 0) -> np.ndarray:
    """DAW playhead as read head: exact coarse bar slot + fine grid phase."""
    step = np.arange(start_step, start_step + n, dtype=np.int64)
    q = cfg.steps_per_sixteenth
    bar_slot = (step // q) % 16
    micro = step % q
    C = np.zeros((n, cfg.clock_dim), np.float32)
    C[np.arange(n), bar_slot] = 1.0
    C[np.arange(n), 16 + micro] = 1.0
    return C


def traces9(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Ten fixed causal traces per voice, measured in musical rather than wall time."""
    cols: List[np.ndarray] = []
    for v in range(NV):
        xf = x[:, v].astype(np.float64)
        for hl_q in cfg.trace_half_lives_quarters:
            hl_steps = hl_q * cfg.steps_per_quarter
            lam = 0.5 ** (1.0 / hl_steps)
            c = lfilter([1.0 - lam], [1.0, -lam], xf)
            cols.append(np.concatenate([[0.0], c[:-1]]))
    return np.stack(cols, axis=1).astype(np.float32)


def _trace_lam(cfg: Config) -> np.ndarray:
    return np.asarray([
        0.5 ** (1.0 / (hl_q * cfg.steps_per_quarter))
        for hl_q in cfg.trace_half_lives_quarters
    ], dtype=np.float64)


def _update_trace_state(state: np.ndarray, y: np.ndarray, lam: np.ndarray) -> None:
    state *= lam[None, :]
    state += (1.0 - lam[None, :]) * y[:, None]


def trainable_scalars(rung: str, cfg: Config) -> int:
    if rung == "clock":
        return NV * cfg.clock_dim + NV
    if rung == "own":
        return NV * cfg.clock_dim + NV * cfg.traces_per_voice + NV
    if rung == "all":
        return NV * (cfg.clock_dim + NV * cfg.traces_per_voice) + NV
    if rung == "coupling":
        return trainable_scalars("all", cfg) + NV * (NV - 1) // 2
    raise ValueError(rung)


def dynamic_state_scalars(rung: str, cfg: Config) -> int:
    return 0 if rung == "clock" else NV * cfg.traces_per_voice


class Readout:
    """Minimal linear probabilistic readout for one preregistered rung."""

    def __init__(self, rung: str, cfg: Config):
        import torch

        if rung not in RUNGS:
            raise ValueError(rung)
        self.rung = rung
        self.cfg = cfg
        self.clock_w = torch.nn.Parameter(torch.zeros(NV, cfg.clock_dim))
        self.bias = torch.nn.Parameter(torch.zeros(NV))
        self.own_w = None
        self.all_w = None
        self.coupling_raw = None
        if rung == "own":
            self.own_w = torch.nn.Parameter(torch.zeros(NV, cfg.traces_per_voice))
        elif rung in {"all", "coupling"}:
            self.all_w = torch.nn.Parameter(
                torch.zeros(NV, NV * cfg.traces_per_voice)
            )
        if rung == "coupling":
            self.coupling_raw = torch.nn.Parameter(torch.zeros(NV, NV))

    def parameters(self):
        ps = [self.clock_w, self.bias]
        if self.own_w is not None:
            ps.append(self.own_w)
        if self.all_w is not None:
            ps.append(self.all_w)
        if self.coupling_raw is not None:
            ps.append(self.coupling_raw)
        return ps

    def _coupling_matrix(self):
        import torch
        if self.coupling_raw is None:
            return None
        return torch.tril(self.coupling_raw, diagonal=-1)

    def logits(self, C, T, Y_prefix=None):
        """Teacher-forced logits; coupling is a proper AR factorization."""
        z = C @ self.clock_w.T + self.bias
        if self.rung == "own":
            Tv = T.reshape(-1, NV, self.cfg.traces_per_voice)
            z = z + (Tv * self.own_w[None, :, :]).sum(-1)
        elif self.rung in {"all", "coupling"}:
            z = z + T @ self.all_w.T
        if self.rung == "coupling":
            if Y_prefix is None:
                raise ValueError("coupling rung requires teacher-forced current tick")
            z = z + Y_prefix @ self._coupling_matrix().T
        return z

    def sample_tick(self, c: np.ndarray, tr: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        import torch

        with torch.no_grad():
            C = torch.from_numpy(c.astype(np.float32, copy=False))
            z = self.clock_w @ C + self.bias
            if self.rung == "own":
                Tv = torch.from_numpy(tr.astype(np.float32, copy=False))
                z = z + (Tv * self.own_w).sum(-1)
            elif self.rung in {"all", "coupling"}:
                Tf = torch.from_numpy(tr.reshape(-1).astype(np.float32, copy=False))
                z = z + self.all_w @ Tf
            out = np.zeros(NV, np.uint8)
            if self.rung != "coupling":
                p = torch.sigmoid(z).cpu().numpy()
                out[:] = (rng.random(NV) < p).astype(np.uint8)
                return out
            J = self._coupling_matrix()
            for v in range(NV):
                if v:
                    prefix = torch.from_numpy(out[:v].astype(np.float32, copy=False))
                    zv = z[v] + (J[v, :v] * prefix).sum()
                else:
                    zv = z[v]
                out[v] = np.uint8(rng.random() < float(torch.sigmoid(zv)))
            return out


def _iter_chunks(x: np.ndarray, cfg: Config, chunk: int = 65536):
    C = clock_features(len(x), cfg)
    T = traces9(x, cfg)
    Y = x.astype(np.float32)
    for i in range(0, len(x), chunk):
        j = min(len(x), i + chunk)
        yield C[i:j], T[i:j], Y[i:j]


def fit_model(rung: str, store: Dict[str, List[dict]], cfg: Config,
              epochs: int = 6, lr: float = 3e-3):
    import torch

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = Readout(rung, cfg)
    total = np.zeros(NV, np.float64)
    nt = 0
    for it in store["train"]:
        total += it["x"].sum(0)
        nt += len(it["x"])
    rate = np.clip(total / max(nt, 1), 1e-5, 1 - 1e-5)
    with torch.no_grad():
        model.bias.copy_(torch.from_numpy(
            np.log(rate / (1 - rate)).astype(np.float32)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        order = rng.permutation(len(store["train"]))
        running = 0.0
        nobs = 0
        for idx in order:
            x = store["train"][int(idx)]["x"]
            for Cn, Tn, Yn in _iter_chunks(x, cfg):
                C = torch.from_numpy(Cn)
                T = torch.from_numpy(Tn)
                Y = torch.from_numpy(Yn)
                z = model.logits(C, T, Y if rung == "coupling" else None)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if rung == "coupling":
                    with torch.no_grad():
                        model.coupling_raw.copy_(
                            torch.tril(model.coupling_raw, diagonal=-1))
                running += float(loss.detach()) * len(Yn)
                nobs += len(Yn)
        print(f"  {rung} epoch {ep+1}/{epochs}: BCE "
              f"{running/max(nobs,1):.6f}", flush=True)
    return model, rate


def bpe_gain(model: Readout, store: Dict[str, List[dict]], rate0: np.ndarray,
             cfg: Config, split: str = "test") -> float:
    import torch

    eps = 1e-7
    nll = nll0 = nevents = 0.0
    with torch.no_grad():
        for it in store[split]:
            for Cn, Tn, Yn in _iter_chunks(it["x"], cfg):
                C = torch.from_numpy(Cn)
                T = torch.from_numpy(Tn)
                Y = torch.from_numpy(Yn)
                z = model.logits(C, T, Y if model.rung == "coupling" else None)
                nll += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    z, Y, reduction="sum")) / math.log(2)
                p0 = np.broadcast_to(rate0, Yn.shape)
                nll0 += float((-(Yn * np.log2(p0 + eps) +
                                (1 - Yn) * np.log2(1 - p0 + eps))).sum())
                nevents += float(Yn.sum())
    return (nll0 - nll) / max(nevents, 1.0)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    q = np.asarray(q, np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0 if (p.sum() > 0) != (q.sum() > 0) else 0.0
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    a = p > 0
    b = q > 0
    return float(0.5 * (p[a] * np.log2(p[a] / m[a])).sum() +
                 0.5 * (q[b] * np.log2(q[b] / m[b])).sum())


def _R16(x: np.ndarray, cfg: Config) -> float:
    on = np.where(x.any(1))[0]
    if len(on) == 0:
        return 0.0
    th = 2 * np.pi * (on % cfg.steps_per_sixteenth) / cfg.steps_per_sixteenth
    return float(np.abs(np.exp(1j * th).mean()))


def generate(model: Readout, seed_x: np.ndarray, cfg: Config,
             rng: np.random.Generator, start_step: int = 0) -> np.ndarray:
    lam = _trace_lam(cfg)
    tr = np.zeros((NV, cfg.traces_per_voice), np.float64)
    if model.rung != "clock":
        for y in seed_x:
            _update_trace_state(tr, y, lam)
    n = cfg.generate_bars * cfg.steps_per_bar
    C = clock_features(n, cfg, start_step=start_step + len(seed_x))
    gen = np.zeros((n, NV), np.uint8)
    for t in range(n):
        y = model.sample_tick(C[t], tr, rng)
        gen[t] = y
        if model.rung != "clock":
            _update_trace_state(tr, y, lam)
    return gen


def freerun_metrics(model: Readout, store: Dict[str, List[dict]], cfg: Config,
                    max_files: int = 40) -> Dict[str, object]:
    need = (cfg.seed_bars + cfg.generate_bars) * cfg.steps_per_bar
    candidates = [it for it in store["test"] if len(it["x"]) >= need][:max_files]
    if not candidates:
        raise RuntimeError("no test clips contain seed+generation horizon")
    rng = np.random.default_rng(cfg.seed)
    rate_rows, r_rows, js_rows = [], [], []
    for it in candidates:
        x = it["x"]
        seed_n = cfg.seed_bars * cfg.steps_per_bar
        seed = x[:seed_n]
        true = x[seed_n:need]
        gen = generate(model, seed, cfg, rng)
        gb = gen.reshape(cfg.generate_bars, cfg.steps_per_bar, NV)
        seed_rate = seed.sum() / cfg.seed_bars
        g_rates = gb.sum((1, 2)).astype(np.float64)
        rate_rows.append({
            "bar1_ratio": float(g_rates[0] / max(seed_rate, 1e-9)),
            "bar16_ratio": float(g_rates[-1] / max(seed_rate, 1e-9)),
            "cv": float(g_rates.std() / max(g_rates.mean(), 1e-9)),
            "slope_frac_per_bar": float(np.polyfit(
                np.arange(cfg.generate_bars), g_rates, 1)[0] /
                max(g_rates.mean(), 1e-9)),
        })
        first4 = gb[:4].reshape(-1, NV)
        last4 = gb[-4:].reshape(-1, NV)
        r_rows.append({
            "generated_R16": _R16(gen, cfg),
            "human_R16": _R16(true, cfg),
            "R16_first4": _R16(first4, cfg),
            "R16_last4": _R16(last4, cfg),
            "R16_drift": _R16(last4, cfg) - _R16(first4, cfg),
        })
        js_rows.append({
            "gen_vs_human_bits": _js(gen.sum(0), true.sum(0)),
            "first4_vs_last4_bits": _js(first4.sum(0), last4.sum(0)),
        })

    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    return {
        "n_files": len(candidates),
        "event_rate": {k: med(rate_rows, k) for k in rate_rows[0]},
        "metrical": {k: med(r_rows, k) for k in r_rows[0]},
        "voice_distribution": {k: med(js_rows, k) for k in js_rows[0]},
    }


def run(store: Dict[str, List[dict]], cfg: Config, epochs: int,
        do_freerun: bool) -> Dict[str, object]:
    results: Dict[str, object] = {
        "representation": {
            "voices": list(CHANNELS),
            "steps_per_quarter": cfg.steps_per_quarter,
            "steps_per_sixteenth": cfg.steps_per_sixteenth,
            "steps_per_bar": cfg.steps_per_bar,
            "tempo_conditioning": "host transport; BPM not a learned feature",
        },
        "rungs": {},
    }
    for rung in RUNGS:
        print(f"\n=== {rung.upper()} ===", flush=True)
        model, rate0 = fit_model(rung, store, cfg, epochs=epochs)
        gain = bpe_gain(model, store, rate0, cfg)
        row = {
            "trainable": trainable_scalars(rung, cfg),
            "state": dynamic_state_scalars(rung, cfg),
            "bits_per_event_over_rate": float(gain),
        }
        actual = sum(int(p.numel()) for p in model.parameters())
        if rung == "coupling":
            # coupling_raw stores 81 values; only 36 strict-lower entries are live.
            actual -= NV * (NV + 1) // 2
        assert actual == row["trainable"], (actual, row["trainable"])
        if do_freerun:
            row["freerun"] = freerun_metrics(model, store, cfg)
        results["rungs"][rung] = row
        print(f"params={row['trainable']} state={row['state']} "
              f"gain={gain:+.3f} b/event", flush=True)
        if do_freerun:
            m = row["freerun"]
            print("  free-run | "
                  f"rate {m['event_rate']['bar1_ratio']:.2f}->"
                  f"{m['event_rate']['bar16_ratio']:.2f}x "
                  f"R16 {m['metrical']['generated_R16']:.3f} "
                  f"JS {m['voice_distribution']['gen_vs_human_bits']:.3f}",
                  flush=True)
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--freerun", action="store_true")
    ap.add_argument("--data", type=pathlib.Path, default=GMD)
    ap.add_argument("--steps-per-quarter", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--output", type=pathlib.Path, default=OUT / "v3_exact.json")
    a = ap.parse_args(argv)
    cfg = Config(steps_per_quarter=a.steps_per_quarter, seed=a.seed)
    if a.prep:
        prep(a.data.expanduser(), cfg)
    if a.sweep or a.freerun:
        store = load_store()
        result = run(store, cfg, epochs=a.epochs, do_freerun=a.freerun)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf8")
        print(f"wrote {a.output}", flush=True)
    if not (a.prep or a.sweep or a.freerun):
        ap.error("choose --prep, --sweep, and/or --freerun")


if __name__ == "__main__":
    main()
