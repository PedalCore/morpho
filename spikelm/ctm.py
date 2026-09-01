"""A Continuous Thought Machine, implemented from the paper's mechanism.

Darlow et al., arXiv:2505.05522. Our earlier attempt at this took CTM's
synchronisation READOUT and bolted it onto an RWKV, which measured worse
than its own baseline. This implements the part we left out: the control
loop. Synchronisation generates a query, the query cross-attends to the
data, the result drives the next internal tick, and synchronisation
updates. Without that loop the sync representation is just an odd feature
extractor, which is very likely why ours did nothing.

One internal tick, in the paper's order:

    a^t     = f_syn(concat(z^t, o^t))            pre-activations, R^D
    A^t     = [a^(t-M+1) ... a^t]                per-neuron FIFO history
    z^(t+1) = g_d(A_d^t) for each neuron d       NEURON-LEVEL MODELS
    S^t     from the history of post-activations SYNCHRONISATION
    q^t     = W_in  . S_action^t                 the query
    o^t     = Attention(q^t, KV=data)            cross-attention
    y^t     = W_out . S_out^t                    the prediction

Two things make this different from an ordinary recurrent net, and both
are implemented here rather than approximated:

  * every neuron has PRIVATE weights over its own history of incoming
    signal, so units are not interchangeable the way channels usually are;
  * the representation handed to the output is not activations but the
    decayed inner product between PAIRS of neurons over time - what fires
    together, weighted by how persistently.

Synchronisation is computed recursively, which the paper notes is possible
and which is what keeps a tick O(pairs) instead of O(pairs x t):

    num_ij^t = exp(-r_ij) num_ij^(t-1) + z_i^t z_j^t
    den_ij^t = exp(-r_ij) den_ij^(t-1) + 1
    S_ij^t   = num_ij^t / sqrt(den_ij^t)

Deviation from the paper, stated rather than hidden: the synapse model is
a 2-layer residual MLP, not the 16-layer U-NET the paper uses. That is a
capacity choice for a small experiment, and it is the first thing to
change if results are weak.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronLevelModels(nn.Module):
    """One private MLP per neuron, over that neuron's own history.

    Batched as a grouped matmul: parameters are (D, M, H) and (D, H), so
    neuron d never sees neuron e's history. This is the piece that makes
    units non-interchangeable.
    """

    def __init__(self, D, M, hidden=16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(D, M, hidden) / math.sqrt(M))
        self.b1 = nn.Parameter(torch.zeros(D, hidden))
        self.w2 = nn.Parameter(torch.randn(D, hidden) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(D))

    def forward(self, A):                       # A: (B, D, M)
        h = torch.einsum("bdm,dmh->bdh", A, self.w1) + self.b1
        h = F.gelu(h)
        return torch.einsum("bdh,dh->bd", h, self.w2) + self.b2


class Synchronisation(nn.Module):
    """Decayed inner products between sampled pairs of neurons.

    r >= 0 is learnable per pair, so each pair chooses its own timescale:
    some report whether two neurons fired together on this tick, others
    whether they have been agreeing for many ticks.
    """

    def __init__(self, D, n_pairs, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("ia", torch.randint(0, D, (n_pairs,), generator=g))
        self.register_buffer("ib", torch.randint(0, D, (n_pairs,), generator=g))
        self.r_raw = nn.Parameter(torch.zeros(n_pairs))   # decay via softplus
        self.n_pairs = n_pairs

    def reset(self, B, device, dtype):
        z = torch.zeros(B, self.n_pairs, device=device, dtype=dtype)
        return z, z.clone()

    def step(self, z, num, den):
        """One recursive update. z: (B, D) post-activations."""
        decay = torch.exp(-F.softplus(self.r_raw))        # in (0, 1]
        prod = z[:, self.ia] * z[:, self.ib]
        num = decay * num + prod
        den = decay * den + 1.0
        return num / den.sqrt().clamp_min(1e-6), num, den


class CrossAttention(nn.Module):
    """Single-head cross-attention from the query the model built itself."""

    def __init__(self, d_q, d_kv, d_head):
        super().__init__()
        self.q = nn.Linear(d_q, d_head, bias=False)
        self.k = nn.Linear(d_kv, d_head, bias=False)
        self.v = nn.Linear(d_kv, d_head, bias=False)
        self.scale = d_head ** -0.5

    def forward(self, q_in, kv, mask=None):     # q_in (B,dq), kv (B,N,dkv)
        q = self.q(q_in).unsqueeze(1)                        # (B,1,H)
        att = (q @ self.k(kv).transpose(1, 2)) * self.scale  # (B,1,N)
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        return (att.softmax(-1) @ self.v(kv)).squeeze(1)     # (B,H)


class CTM(nn.Module):
    def __init__(self, d_input, n_out, D=128, M=16, ticks=8,
                 pairs_out=256, pairs_action=256, d_head=64, nlm_hidden=16):
        super().__init__()
        self.D, self.M, self.ticks = D, M, ticks
        self.attn = CrossAttention(d_head, d_input, d_head)
        self.synapse = nn.Sequential(
            nn.Linear(D + d_head, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.ln = nn.LayerNorm(D)
        self.nlm = NeuronLevelModels(D, M, nlm_hidden)
        self.sync_out = Synchronisation(D, pairs_out, seed=1)
        self.sync_act = Synchronisation(D, pairs_action, seed=2)
        self.w_in = nn.Linear(pairs_action, d_head, bias=False)
        self.w_out = nn.Linear(pairs_out, n_out, bias=False)
        self.z0 = nn.Parameter(torch.zeros(D))
        self.a0 = nn.Parameter(torch.zeros(M, D))

    def forward(self, kv, mask=None):
        """Returns per-tick predictions, (B, ticks, n_out)."""
        B = kv.shape[0]
        dev, dt = kv.device, kv.dtype
        z = self.z0.expand(B, self.D).contiguous()
        A = self.a0.t().expand(B, self.D, self.M).contiguous()
        no, do = self.sync_out.reset(B, dev, dt)
        na, da = self.sync_act.reset(B, dev, dt)
        o = torch.zeros(B, self.attn.q.in_features, device=dev, dtype=dt)

        ys = []
        for _ in range(self.ticks):
            a = self.synapse(torch.cat([z, o], dim=-1))       # pre-activations
            A = torch.cat([A[:, :, 1:], a.unsqueeze(-1)], dim=-1)   # FIFO
            z = self.ln(self.nlm(A))                          # per-neuron MLPs

            s_out, no, do = self.sync_out.step(z, no, do)
            s_act, na, da = self.sync_act.step(z, na, da)

            o = self.attn(self.w_in(s_act), kv, mask)          # query -> data
            ys.append(self.w_out(s_out))
        return torch.stack(ys, dim=1)


def ctm_loss(logits, target):
    """The paper's two-tick loss: train on the tick that was most correct
    AND the tick the model was most certain about, so it learns both to be
    right and to know when.

    DEGENERACY, measured here, when ticks are comparable to classes.
    t1 = argmin_t L^t lets the model emit a DIFFERENT CONSTANT at each
    tick. With C classes and T >= C ticks, one tick is then correct for any
    target, argmin selects it, and the loss goes to ~0 having learned
    nothing. We measured exactly this: on a trivial 2-class task the
    selected loss reached 0.0001 while accuracy stayed at 48%, where the
    mean-over-ticks loss on the identical model reached 100%.

    This does not affect the paper's own results - with 1000 ImageNet
    classes no number of ticks can cover the label set - but it makes this
    loss unusable for small-C tasks without a guard. Use mean_tick_loss
    below when C is small, and say which one was used.

    logits: (B, T, C). Returns loss, plus the two chosen tick indices.
    """
    B, T, C = logits.shape
    flat = logits.reshape(B * T, C)
    tgt = target.unsqueeze(1).expand(B, T).reshape(-1)
    per = F.cross_entropy(flat, tgt, reduction="none").view(B, T)

    p = logits.softmax(-1)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(C)
    certainty = 1.0 - ent                                     # (B, T)

    t1 = per.argmin(dim=1)
    t2 = certainty.argmax(dim=1)
    idx = torch.arange(B, device=logits.device)
    return 0.5 * (per[idx, t1] + per[idx, t2]).mean(), t1, t2


def mean_tick_loss(logits, target):
    """Every tick supervised equally.

    Deviates from the paper deliberately: with a small label set the
    two-tick loss above is exploitable, so a fair test of whether the TICK
    LOOP works needs a loss that cannot be gamed by tick-shopping. The cost
    is that this supervises the early ticks that cannot yet have seen the
    data, which slightly understates what the loop can do.
    """
    B, T, C = logits.shape
    tgt = target.unsqueeze(1).expand(B, T).reshape(-1)
    return F.cross_entropy(logits.reshape(B * T, C), tgt), None, None


def last_tick_loss(logits, target):
    """Supervise only the final tick.

    Needed because neither loss above can measure whether the tick loop
    helps. The paper's two-tick loss is degenerate when ticks are
    comparable to classes; the mean loss penalises large T, because ticks
    the model cannot yet have answered (tick 0 sees no data at all, by the
    paper's ordering) are supervised as if it could. Measured: at 4-bit
    parity the mean loss gives 100% at T=4 and 55% at T=8 — an artifact of
    the loss, not of the loop.

    Supervising the last tick asks exactly the ablation's question: after T
    ticks of thinking, how good is the answer? It is silent about WHEN the
    model became right, which is what the paper's loss is for, so use that
    one for adaptive-computation claims and this one for tick ablations.
    """
    return F.cross_entropy(logits[:, -1], target), None, None


def tail_mean_loss(logits, target, frac=0.5):
    """Mean over the final fraction of ticks — the only one of these four
    that can actually run a tick ablation.

    The other three each fail in a different way, all measured here:
      two_tick   degenerate when ticks ~ classes (loss -> 0 at chance)
      mean       penalises large T by supervising un-answerable early ticks
      last_tick  gradient too sparse to train: 55.8% at 4-bit parity where
                 the mean loss reaches 100% on the identical model

    Supervising the tail keeps dense gradients while excluding the ticks
    that cannot have seen the data yet, so raising T adds capacity instead
    of adding penalty.
    """
    B, T, C = logits.shape
    start = max(0, T - max(1, int(round(T * frac))))
    tail = logits[:, start:]
    n = tail.shape[1]
    tgt = target.unsqueeze(1).expand(B, n).reshape(-1)
    return F.cross_entropy(tail.reshape(B * n, C), tgt), None, None
