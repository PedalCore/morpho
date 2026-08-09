// Deterministic seeded PRNG (mulberry32). Every stochastic subsystem gets its
// own stream so e.g. toggling development does not perturb input noise.
export function mulberry32(seed) {
  let a = seed >>> 0;
  const rng = function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // state access for organism persistence — same stream continues after load
  rng.getState = () => a >>> 0;
  rng.setState = (s) => {
    a = s >>> 0;
  };
  return rng;
}

export function makeStreams(baseSeed) {
  return {
    build: mulberry32(baseSeed ^ 0x9e3779b9),
    sim: mulberry32(baseSeed ^ 0x85ebca6b),
    dev: mulberry32(baseSeed ^ 0xc2b2ae35),
  };
}

export function pick(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

export function randInt(rng, lo, hi) {
  // inclusive lo, exclusive hi
  return lo + Math.floor(rng() * (hi - lo));
}
