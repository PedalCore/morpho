# Division and exponentiation

Two arithmetic extensions built entirely from the article's existing
primitives (ripple adder, Wallace multiplier, mux) plus the sequential
extension's `REG`/`DRIVE`. Run either file from the repository root:

```bash
python3 examples/arithmetic/divider.py
python3 examples/arithmetic/power.py
```

| design | idea | space version | time version |
|---|---|---|---|
| `divider.py` | restoring long division; the ripple adder's carry-out **is** the ≥ comparison | recursion with the same shape as the ripple adder, but the *whole remainder bus* threads through it like a carry (16/8: 527 gates) | a streaming divider: one divisor-wide subtractor + a remainder register; feed any dividend MSB-first and quotient bits emerge in real time |
| `power.py` | binary square-and-multiply | a Morpho recursion over the **exponent bus** — peel a bit, conditionally multiply, square, recurse; fallback returns the accumulator (8-bit, e≤15: 572 gates) | the extreme space→time rotation: state = one accumulator register; `acc' = e_bit ? acc²·a : acc²`, one tick per exponent bit, any exponent length (8 regs + 188 gates) |

Both are verified against integer arithmetic (`divmod`, `pow(a, e, 2^N)`)
on thousands of random cases, in dynamic execution and through the
compiler. Division requires a nonzero divisor, the usual restoring-
division precondition.
