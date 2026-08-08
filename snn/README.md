# morpho-snn lab

Lives in `snn/` inside the [PedalCore/morpho](https://github.com/PedalCore/morpho)
fork — a research corner for SNN musical experimentation alongside the original
Morpho web implementation at the repo root.

Experimental browser lab exploring the ideas in
*Coding Agent Brief — Morpho Developmental SNN Research Track*: a Morpho-style
recursive developmental grammar grows a recurrent spiking (LIF) network, whose
activity feeds back into growth and pruning — and whose output spikes are
sonified as notes in a musical scale.

**The anatomy is the music:** pitch is structural and fixed at each neuron's
birth — region depth sets the octave (center = low, rim = high), structural
position sets the scale degree. Growth adds pitches, pruning removes them,
region division opens new registers. Violet **walkers** (the plugin briefs'
LIVE mode) roam the graph as melodic voices, and the structures they play
gain survival energy — the music helps decide what lives.

## Run it

```bash
# from the repo root:
python3 -m http.server 8765   # → http://localhost:8765/snn/  (root serves original Morpho)
# or from snn/:
npm run serve                  # → http://localhost:8765/
```

Enable **sound**, press **run**. Try:

- **development** off/on — frozen organism vs living one
- **seed** + **grow** — a different genotype, a different organism
- **⑂ branch** — recursive fan-out on demand: sprout fresh populations a
  register deeper
- **walkers / variation** — melodic agents and how adventurous they are
- **strum** — fan chords out into arpeggios
- **birth sfx** — toggle the structural sounds (chimes/thuds/arpeggios)
- **reverb / delay / chorus** — fx sends; the delay time follows the pulse
  (dotted), so echoes stay in the organism's tempo
- **scale / pulse / density** — musical exploration

## Duet mode

`duet.html` — the human-in-the-loop experiment: your notes (MIDI keyboard,
on-screen pads, or the computer keys shown on the pads, all locked to the
current scale/key) become spike bursts in a tonotopic sensory layer. There's
no metronome — you are the environment the brain develops in. STDP
strengthens what you play, development grows around it, **✚ reinforce**
rewards responses you like, and MIDI out lets it play your gear.

## Test it

```bash
cd snn && npm test   # 25 headless tests, node >= 20
```

## Layout

```
js/core/rng.js            seeded PRNG (3 independent streams)
js/neural/                LIF neurons, synapses, graph, spike engine, activity stats
js/morpho/                "morpho-lite" recursive grammar + development controller
js/sim/lab.js             headless experiment harness (used by tests AND the UI)
js/ui/                    canvas renderer + WebAudio sonification
index.html, js/main.js    the lab page
EXPERIMENT.md             hypothesis, protocol, findings, next experiments
```

See `EXPERIMENT.md` for what has been learned so far and what to try next.
