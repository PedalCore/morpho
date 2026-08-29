import numpy as np

from rhythm3_exact import (
    CHANNELS,
    Config,
    Readout,
    VOICE,
    _R16,
    _js,
    clock_features,
    dynamic_state_scalars,
    traceable_scalars if False else trainable_scalars,
    traces9,
)


def test_mapping_and_counts():
    cfg = Config()
    assert len(CHANNELS) == 9
    assert VOICE[36] == 0 and VOICE[40] == 1 and VOICE[59] == 8
    assert cfg.clock_dim == 28
    assert trainable_scalars("clock", cfg) == 261
    assert trainable_scalars("own", cfg) == 351
    assert trainable_scalars("all", cfg) == 1071
    assert trainable_scalars("coupling", cfg) == 1107
    assert dynamic_state_scalars("clock", cfg) == 0
    assert dynamic_state_scalars("all", cfg) == 90


def test_clock_and_traces_are_causal():
    cfg = Config()
    C = clock_features(cfg.steps_per_sixteenth + 4, cfg)
    assert C.shape == (cfg.steps_per_sixteenth + 4, cfg.clock_dim)
    assert np.allclose(C.sum(1), 2)
    x = np.zeros((20, 9), np.uint8)
    x[0, 0] = 1
    T = traces9(x, cfg).reshape(20, 9, 10)
    assert np.all(T[0, 0] == 0)
    assert np.all(T[1, 0] > 0)


def test_coupling_is_strictly_autoregressive():
    import torch

    cfg = Config()
    model = Readout("coupling", cfg)
    with torch.no_grad():
        model.coupling_raw.fill_(1.0)
    J = model._coupling_matrix().detach().numpy()
    assert np.allclose(np.triu(J), 0)
    assert np.allclose(np.tril(J, -1), np.tril(np.ones((9, 9)), -1))


def test_stability_primitives():
    cfg = Config(generate_bars=16)
    gen = np.zeros((cfg.generate_bars * cfg.steps_per_bar, 9), np.uint8)
    for b in range(16):
        for q in range(4):
            t = b * cfg.steps_per_bar + q * cfg.steps_per_quarter
            gen[t, 0] = 1
            gen[t, 2] = 1
    assert _R16(gen, cfg) > 0.999
    assert _js(gen[:4 * cfg.steps_per_bar].sum(0),
               gen[-4 * cfg.steps_per_bar:].sum(0)) < 1e-12
