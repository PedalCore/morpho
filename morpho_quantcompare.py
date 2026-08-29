"""Binary/ternary weights vs stochastic vs exact — measured in Morpho."""
import numpy as np, sys
sys.path.insert(0, '/Users/marycarrigan/coding/morpho-snn')
from tiny_morpho import morpho, compile, LUT, LSLICE, CAT, VOID, ZERO, ONE, \
    brent_kung_adder, Not, Xor, And, Or, REPEAT
from morpho_lm import qmul_s

Xnor = LUT(2, 0b1001)
ONEBIT = np.zeros((1,1), np.int32)

# ---- 1. exact: signed multiply (what we measured before)
@morpho
def exact_mac(w, x, acc):
    p = qmul_s(w, x, len(w)//2)
    s, _ = brent_kung_adder(p, acc, ZERO)
    return s

# ---- 2. ternary weight {-1,0,+1} x full activation: no multiplier at all
@morpho
def ternary_mac(sign, nonzero, x, acc):
    """w in {-1,0,+1} as two bits. add x, subtract x, or skip. A multiply
    becomes a conditional negate and an add — BitNet b1.58's arithmetic."""
    gated = And(x, REPEAT(nonzero, x))            # zero weight -> contribute nothing
    inv = Xor(gated, REPEAT(sign, gated))         # negative weight -> two's complement
    s, _ = brent_kung_adder(inv, acc, sign)       # sign as carry-in completes the negate
    return s

# ---- 3. binary weight AND activation: XNOR, then popcount
@morpho
def binary_mul(w, x):
    return Xnor(w, x)

def _pc_done(bits):
    return VOID

@morpho(fallback=0)
def popcount(bits):
    """Sum a bag of 1-bit values into a binary count, by recursive halving."""
    n = len(bits)
    if n == 1:
        return bits
    half = n // 2
    a = popcount(bits[:half])
    b = popcount(bits[half:2*half])
    w = max(len(a), len(b))
    a = CAT(a, np.zeros((w-len(a),)+a.shape[1:], np.int32))
    b = CAT(b, np.zeros((w-len(b),)+b.shape[1:], np.int32))
    s, c = brent_kung_adder(a, b, ZERO)
    out = CAT(s, c)
    if 2*half < n:                                 # odd leftover bit
        z = np.zeros((len(out)-1,)+out.shape[1:], np.int32)
        out, _ = brent_kung_adder(out, CAT(bits[-1:], z), ZERO)
    return out

@morpho
def binary_dot(w, x):
    return popcount(binary_mul(w, x))

def gates(f, sizes):
    return len(compile(f, sizes).ops)

B = 8
print("cost of one multiply-accumulate, 8-bit activations, measured:\n")
g_exact = gates(exact_mac, [B, B, B])
g_tern  = gates(ternary_mac, [1, 1, B, B])
print(f"  exact   signed multiply + add          {g_exact:6,} gates   1 pass")
print(f"  ternary weight in {{-1,0,+1}}, add/sub   {g_tern:6,} gates   1 pass"
      f"   ({g_exact/g_tern:.1f}x cheaper)")
for n in (64, 256):
    g_bin = gates(binary_dot, [n, n])
    print(f"  binary  both 1-bit, XNOR+popcount     {g_bin/n:6.1f} gates   1 pass"
          f"   ({g_exact/(g_bin/n):.0f}x cheaper, n={n})")
print(f"  stochastic (from morpho_stochastic)      4.2 gates   L passes")
print()
gb = gates(binary_dot, [256, 256])/256
print("size x time per multiply:")
print(f"  exact       {g_exact:8,.0f}")
print(f"  ternary     {g_tern:8,.0f}")
print(f"  binary      {gb:8,.1f}")
for L in (64, 1024, 16384):
    print(f"  stochastic L={L:<6} {4.2*L:8,.0f}   ({4.2*L/gb:.0f}x binary)")
