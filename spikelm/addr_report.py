"""Combined oracle-vs-learned report: init vs trained, per seed, all metrics.

Reports exactly what the review asked, together:
  * recall by seed and position, with counts
  * target rank, margin, read mass
  * self-sensitivity and cross-contamination
  * training loss + checkpoint step (completed budget != convergence)
  * each trained checkpoint compared with ITS OWN init on the SAME eval
    examples (diagnose() fixes the eval rng), because learned keys need
    not become orthogonal to support recall - the init delta shows what
    training actually moved.

Kept open to partial progress: better addressing without successful
decoding is a distinct, informative outcome (recall low but overlap
down / margin up).

    python addr_report.py --arms oracle learned --seeds 0 1
"""

import argparse
import json

import numpy as np
import torch

from nonce_lm import load_carrier
from oracle_addr import OracleAddrLM
from addr_diag import diagnose


def init_model(arm, N, seed, device):
    torch.manual_seed(seed)                    # exact training init (verified)
    return OracleAddrLM(arm, N).to(device)


def loss_at(logpath, arm, seed):
    """Final training loss + step for (arm, seed) from the run log."""
    try:
        lines = [l for l in open(logpath)
                 if f"{arm} N=" in l and f"seed{seed} step" in l]
        if lines:
            p = lines[-1].split()
            step = p[p.index("step") + 1] if "step" in p else "?"
            loss = p[-1]
            return step, loss
    except FileNotFoundError:
        pass
    return "?", "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["oracle", "learned"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--log", default="oracle.log")
    ap.add_argument("--out", default="addr-report.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()
    report = {}

    for arm in a.arms:
        for s in a.seeds:
            step, loss = loss_at(a.log, arm, s)
            di = diagnose(init_model(arm, a.N, s, dev), a.N, carrier, dev)
            ck = torch.load(f"oracle-{arm}-s{s}.pt", map_location=dev)
            mt = OracleAddrLM(arm, a.N).to(dev); mt.load_state_dict(ck["model"])
            dt = diagnose(mt, a.N, carrier, dev)
            report[f"{arm}-s{s}"] = {"init": di, "trained": dt,
                                     "final_step": step, "final_loss": loss}

            print(f"\n=== {arm}  seed {s}   (final step {step}, loss {loss}) ===")
            print(f"  recall           init {di['recall']:.1%}  ->  "
                  f"trained {dt['recall']:.1%}")
            print(f"  write-overlap    {di['write_overlap']:+.3f} -> "
                  f"{dt['write_overlap']:+.3f}   (oracle target 0)")
            print(f"  query tgt/dist   {di['query_target']:+.2f}/"
                  f"{di['query_distractor']:+.2f} -> {dt['query_target']:+.2f}/"
                  f"{dt['query_distractor']:+.2f}")
            print(f"  target rank      {di['target_rank']:.2f} -> "
                  f"{dt['target_rank']:.2f}  /{a.N-1}   (0=target wins)")
            print(f"  target margin    {di['target_margin']:+.3f} -> "
                  f"{dt['target_margin']:+.3f}   (>0 target beats all)")
            print(f"  read mass t/d    {di['read_mass_target']:.2f}/"
                  f"{di['read_mass_distractor']:.2f} -> "
                  f"{dt['read_mass_target']:.2f}/{dt['read_mass_distractor']:.2f}")
            print(f"  cf self/cross    {di['sens_self']:.2f}/{di['sens_cross']:.2f}"
                  f" -> {dt['sens_self']:.2f}/{dt['sens_cross']:.2f}"
                  f"   (clean: self high, cross 0)")
            print(f"  recall by position (trained, count):")
            print("    " + "  ".join(f"p{i}:{r:.2f}({c})" for i, (r, c) in
                                     enumerate(zip(dt['pos_recall'],
                                                   dt['pos_counts']))))
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
