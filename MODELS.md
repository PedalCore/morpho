# Two language models, running in a page

A proof of concept for portability: two different recurrent architectures,
trained in the same harness, exported to the same flat format, and executed
by a page of plain JavaScript — no server, no framework, no libraries.

| page | architecture | params | held-out perplexity |
|---|---|---|---|
| [`rwkv-live.html`](rwkv-live.html) | RWKV-v4 (fixed decay, normalized) | 13.10M | 6.84 |
| [`mamba-live.html`](mamba-live.html) | Mamba/S6 (selective decay, no division) | 13.14M | 7.83 |

Open either file over HTTP (`python3 -m http.server`, then visit the page —
`file://` will not work, the loader uses `fetch`). Click **load model**; the
weights are the `*-export/` directory beside the page.

**Both are verified against their PyTorch originals** — the shipped browser
code, run in node on the same int8 weights, reproduces the reference logits
to four decimal places.

**Caveat on the perplexity comparison:** Mamba trained at batch 8 against
RWKV's 16, so at equal step counts it saw half the tokens. The gap is not a
fair architecture verdict, and the controlled rematch is outstanding.

## Why this sits in a hardware-description-language repository

Both recurrences reduce to the same primitive — `h ← a·h + b·u` with
`a = 2^(negative)` — differing in whether `a` is a per-channel constant
(RWKV) or computed per token (Mamba), and in whether the state is a scalar
or an N-vector. That is a *parameterized cell*, which is what this project
is for: one recursive definition, either architecture, any width.

The circuit treatment of both, including a verified 2^(−x) unit and measured
fixed-point tolerances, is at
[soundlark.studio/wkv-cell.html](https://soundlark.studio/wkv-cell.html) and
[soundlark.studio/mamba-cell.html](https://soundlark.studio/mamba-cell.html).
Training, export and validation code:
[`spikelm/` on the snn-lab branch](https://github.com/PedalCore/morpho/tree/snn-lab/spikelm).

## Getting the weights

The two `*-export/` directories in this branch hold everything: manifest,
tokenizer, and the int8 binary. Clone and serve locally and the pages work
unchanged:

```bash
git clone -b models https://github.com/PedalCore/morpho
cd morpho && python3 -m http.server
# then open http://localhost:8000/rwkv-live.html
```

Opened directly from `file://`, the pages fetch the binary from jsDelivr
instead, which serves this repository with `Access-Control-Allow-Origin: *`.

A note for anyone trying the same trick: **GitHub release assets do not set
CORS headers** — we checked, on both `HEAD` and ranged `GET` — so a page
cannot fetch a model from a release. jsDelivr can, and serves the identical
bytes (verified by SHA-256 against the local file).

The weights live on this branch rather than `main`, which carries only the
sequential MorphoHDL extension.
