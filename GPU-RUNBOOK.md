# GPU session runbook — M5 language phase
# (written 2026-08-25 so rented hours are execution, not setup)

Target: any CUDA box with a 24GB-class card (A10/3090/4090/L4). Rough
budget: the full session below is ~6-8 GPU-hours.

## 0. Environment (~15 min)

```
git clone https://github.com/PedalCore/whitebox-lm && cd whitebox-lm
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install flash-linear-attention triton datasets tokenizers
# spikelm data dependency: copy the TinyStories token cache + tokenizer
# from the Mac (whitebox expects spikelm.data; either rsync the spikelm
# repo or vendor its data/ module — paths at top of train.py)
```

## 1. VERIFY BEFORE ANYTHING TRAINS (~30 min)

The sequential implementations in model.py are the correctness
oracles. Nothing trains until the fast kernels match them.

a. FLA gated-deltanet vs our DeltaMem (M5-GDN labeling — this is a
   REPLICATION arm, not M5-implicit; equations differ where they
   differ, document the delta):
```
python -m whitebox.verify_fla   # to write on-site: load
  fla.layers.GatedDeltaNet at d=448 H=8; feed identical (k,v,q,gates);
  compare forward, input grads, param grads, final recurrent state vs
  DeltaMem sequential. Accept < 1e-4 fwd / 1e-3 grad (fp32).
```
b. FLA chunked path vs our LonghornMem (diagonal): same protocol.
   If FLA has no exact diagonal-delta equivalent, port our chunked
   LonghornMem to Triton OR run FLA-GDN as the speed vehicle and keep
   LonghornMem-naive for ablations (it is fast enough on CUDA to
   check).
c. Speed table at the exact LM shape (B16 T256 d448 L12, fwd+bwd):
   qkv+MLP (flash attention), M5-diag (FLA or naive), M5-GDN (FLA),
   slots-v2 (historical 2.33 s/step on MPS for context). GATE 2:
   M5 within 1.5x of qkv wall-clock.

## 2. The LM ladder (order fixed; ~1h per 3k screen, ~2-3h per 20k)

| run | config | judged by |
|---|---|---|
| B-screen | --longhorn --mlp, 3k, d448 | gate 3: within 0.3-0.5 ppl of qkv@3k (9.32) |
| C-screen | FLA GDN + MLP, 3k | same |
| winner-20k | best of B/C, 20k | vs qkv-ref final 6.04 (10k) + the SHELVED matched-20k qkv control — run it here (cheap on GPU) |
| E | winner + --dict 4 --dict-local (feature block swap), 3k then 20k if within 0.3 of winner | white-box story |

Flags already exist in train.py (--longhorn wired via attn='longhorn';
add a --gdn flag wrapping the verified FLA layer — thin adapter,
~20 lines, write on-site after step 1a passes).

## 3. After every finished LM run (~10 min each)

```
python -m whitebox.counterfactual whitebox/runs/<name>/ckpt.pt
```
THE question of the campaign: the M5 operator binds perfectly under
explicit supervision (probe phase, replicated, stress-swept) — does
language finally recruit it? S > 0 at BOTH query positions = first
genuine LM binding of the program. S ~ 0 with position-prior signature
= the task-pressure finding extends to an operator with no capability
excuse, which is itself the strongest form of that result.

## 4. Bring home

runs/*/log.jsonl, ckpt.pt for every arm, the speed table, verify_fla
output. Everything else (docs, analysis, pages) happens back on the
Mac. Do NOT delete the pod before checking ckpts copied.

## Deferred / out of scope for session 1

EAM (closed), arm D (retired unneeded), Triton sparse segmented scan
(only if a future operator needs it), long-context extrapolation
(needs position-encoding work first), MPS/Metal kernels (community
side-quest, separate decision).
