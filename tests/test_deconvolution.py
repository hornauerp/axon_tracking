import numpy as np

from axon_tracking import deconvolution as dc


def _traces(footprints, trains, n_samples, pre):
    """Superposition of footprints (unit -> (window, electrodes), lag a = time a - pre) at the spike times."""
    traces = np.zeros((n_samples, next(iter(footprints.values())).shape[1]))
    for u, fp in footprints.items():
        for t in trains[u]:
            traces[t - pre : t - pre + len(fp)] += fp
    return traces


def test_deconvolution_recovers_footprints_of_locked_units():
    rng = np.random.default_rng(0)
    pre, post, n = 5, 15, 20000
    fps = {u: rng.normal(0, 1, (pre + post, 4)) for u in range(3)}
    t0 = np.sort(rng.choice(np.arange(100, n - 100), 150, replace=False))
    locked = t0[rng.random(len(t0)) < 0.8] + 3  # unit 1 fires 3 samples after unit 0 in 80% of its spikes
    t1 = np.unique(np.concatenate([locked, rng.choice(np.arange(100, n - 100), 30)]))
    t2 = np.sort(rng.choice(np.arange(100, n - 100), 200, replace=False))
    trains = dc.valid_spikes([t0, t1, t2], n, pre, post)
    traces = _traces(fps, dict(enumerate(trains)), n, pre)
    S = dc.spike_triggered_sums(traces, trains, pre, post)
    G = dc.gram_matrix(trains, pre + post)
    F = dc.deconvolve(S, G)
    for u in range(3):
        np.testing.assert_allclose(F[u], fps[u], atol=1e-8)
    sta0 = S[0] / len(trains[0])
    assert np.abs(sta0 - fps[0]).max() > 0.5  # the average contains unit 1


def test_chunked_sums_and_solver_match_direct_computation():
    rng = np.random.default_rng(1)
    pre, post, n = 5, 15, 30000
    fps = {u: rng.normal(0, 1, (pre + post, 3)) for u in range(2)}
    t0 = np.sort(rng.choice(np.arange(100, n - 100), 200, replace=False))
    t1 = np.unique(np.concatenate([t0[:150] + 4, rng.choice(np.arange(100, n - 100), 50)]))
    trains = dc.valid_spikes([t0, t1], n, pre, post)
    traces = _traces(fps, dict(enumerate(trains)), n, pre) + rng.normal(0, 0.1, (n, 3))
    S = dc.spike_triggered_sums(traces, trains, pre, post)
    S_chunked = dc.spike_triggered_sums_chunked(lambda a, b: traces[a:b], n, trains, pre, post, chunk_samples=997)
    np.testing.assert_allclose(S_chunked, S)
    F, info = dc.solve_footprints(trains, S, pre + post)
    np.testing.assert_allclose(F, dc.deconvolve(S, dc.gram_matrix(trains, pre + post)), atol=1e-9)
    assert info["ridge"] == 0.0
    # the same neuron sorted twice at a fixed lag: singular, solved with a ridge
    F2, info2 = dc.solve_footprints([t0, t0 + 3], dc.spike_triggered_sums(traces, [t0, t0 + 3], pre, post), pre + post)
    assert info2["ridge"] > 0 and np.all(np.isfinite(F2))


def test_noise_of_isolated_units_equals_sta_noise():
    trains = [np.arange(100, 10000, 200), np.arange(150, 10000, 300)]  # never within 20 samples of each other
    G = dc.gram_matrix(trains, 20)
    sd = dc.deconvolve_noise_sd(G, 20, noise_sd=6.0)
    np.testing.assert_allclose(sd[0], 6.0 / np.sqrt(len(trains[0])))
    np.testing.assert_allclose(sd[1], 6.0 / np.sqrt(len(trains[1])))


def test_simulated_deconvolution_is_exact_without_noise():
    from axon_tracking.simulation import scenarios

    culture = scenarios.build_culture(n_neurons=4, seed=5, neuron_kwargs=dict(axon_length=(800, 800)))
    sc = scenarios.Scenario(network=scenarios.BURSTS, n_configs=4, config_duration_s=60.0, noise_uv=0.0)
    res = scenarios.run_deconvolution(sc, culture, targets=[0], seed=3)[0]
    fp = culture.neurons[0].footprint
    si = fp.spike_index
    true = np.zeros((220 * 120, 80))
    lags = np.arange(-30, 50) + si
    ok = (lags >= 0) & (lags < fp.data_uv.shape[1])
    true[np.ix_(fp.electrodes, np.flatnonzero(ok))] = fp.data_uv[:, lags[ok]]
    true = true.reshape(120, 220, 80).transpose(1, 0, 2)
    recorded = np.abs(res["template_deconv"]).max(axis=2) > 0
    err_deconv = np.abs(res["template_deconv"] - true)[recorded].max()
    err_sta = np.abs(res["template_sta"] - true)[recorded].max()
    assert err_deconv < 1e-3 * np.abs(true).max()
    assert err_sta > 10 * err_deconv


def test_locked_duplicates_are_detected_and_shifted():
    rng = np.random.default_rng(7)
    pre, post, n = 5, 15, 40000
    W = pre + post
    fp = rng.normal(0, 1, (W, 3))
    t0 = np.sort(rng.choice(np.arange(100, n - 100, 30), 400, replace=False))
    t_dup = t0 + 4  # the same neuron, detected 4 samples later
    t_other = np.sort(rng.choice(np.arange(100, n - 100), 300, replace=False))
    pairs = dc.locked_duplicates([t_dup[:-20], t0, t_other], W - 1)
    assert pairs == [(0, 1, 4)]
    # the same timing, but the troughs far apart: a locked partner neuron, not a duplicate
    sites = np.array([[0.0, 0.0], [200.0, 0.0], [0.0, 500.0]])
    t_sub = t_dup[rng.random(len(t_dup)) < 0.85]  # fires with 85% of unit 1's spikes: not mutually locked
    assert dc.locked_duplicates([t_sub, t0, t_other], W - 1, sites=sites) == []
    # mutually locked (every spike of both): one neuron, wherever its troughs are
    assert dc.locked_duplicates([t_dup[:-20], t0, t_other], W - 1, sites=sites) == [(0, 1, 4)]
    assert dc.locked_duplicates([t_dup[:-20], t0, t_other], W - 1, sites=sites * 0.1) == [(0, 1, 4)]
    shifted = dc.shift_footprint(fp, 4)
    np.testing.assert_allclose(shifted[: W - 4], fp[4:])
    assert np.all(shifted[W - 4 :] == 0)
    counts = dc.ccg_counts([t0, t_dup], 10)
    assert counts[0, 1, 4 + 10] == len(t0)
