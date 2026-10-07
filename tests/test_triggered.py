import numpy as np

from axon_tracking import deconvolution as dc
from axon_tracking import triggered as tg


def _traces(footprints, trains, n_samples, pre):
    traces = np.zeros((n_samples, next(iter(footprints.values())).shape[1]))
    for u, fp in footprints.items():
        for t in trains[u]:
            traces[t - pre : t - pre + len(fp)] += fp
    return traces


def _snippets(trigger_times, n_samples, pre, post):
    """Stored frames of a triggered acquisition: [t - pre, t + post) around every trigger (merged if overlapping)."""
    keep = np.zeros(n_samples, bool)
    for t in trigger_times:
        keep[max(t - pre, 0) : t + post] = True
    return np.flatnonzero(keep)


def test_masked_gram_matches_brute_force_and_continuous_case():
    rng = np.random.default_rng(0)
    W, pre, n = 12, 4, 3000
    trains = [np.sort(rng.choice(np.arange(20, n - 20), k, replace=False)) for k in (60, 80, 40)]
    G_cont = dc.gram_matrix(trains, W)
    np.testing.assert_allclose(dc.gram_matrix(trains, W, recorded=np.ones(n, bool), pre=pre), G_cont)
    recorded = rng.random(n) < 0.6
    G = dc.gram_matrix(trains, W, recorded=recorded, pre=pre)
    X = np.zeros((n, 3 * W))  # design matrix restricted to recorded samples
    for u, t in enumerate(trains):
        for s in t:
            X[s - pre + np.arange(W), u * W + np.arange(W)] += 1
    X = X[recorded]
    np.testing.assert_allclose(G, X.T @ X)


def test_gap_aware_deconvolution_is_exact_on_snippets_and_concatenation_is_not():
    rng = np.random.default_rng(1)
    pre, post, n = 5, 25, 60000
    W = pre + post
    fps = {u: rng.normal(0, 1, (W, 4)) for u in range(3)}
    t0 = np.sort(rng.choice(np.arange(100, n - 100, 7), 300, replace=False))
    t1 = np.unique(np.r_[t0[rng.random(len(t0)) < 0.7] + 6, rng.choice(np.arange(100, n - 100), 100)])
    t2 = np.sort(rng.choice(np.arange(100, n - 100), 300, replace=False))
    traces = _traces(fps, {0: t0, 1: t1, 2: t2}, n, pre)
    # triggers: spikes of unit 0 and 2 (their somata), windows shorter than the footprints
    frames = _snippets(np.r_[t0, t2], n, 8, 12)
    tr = tg.TriggeredTraces(lambda a, b: traces[frames[a:b]], frames + 1000)  # frame numbers start at 1000
    assert tr.n_snippets > 100
    solve = lambda trains: dc.solve_footprints(
        trains, dc.spike_triggered_sums_chunked(tr.get_traces, tr.n_samples, trains, pre, post, chunk_samples=5000),
        W, recorded=tr.recorded, pre=pre)[0]
    # all spikes that reach a stored sample as regressors (on the real axis): exact
    oracle = [t - frames[0] for t in (t0, t1, t2)]
    oracle = [t[dc.window_masks(t, W, pre, tr.recorded).any(axis=1)] for t in oracle]
    F = solve(oracle)
    for u in range(3):
        covered = dc.window_masks(oracle[u], W, pre, tr.recorded).any(axis=0)  # lags recorded for some spike
        np.testing.assert_allclose(F[u][covered], fps[u][covered], atol=1e-6)
    # as from a sorting of the stored samples: spikes whose peak was not stored are missing, but cut off near a gap
    stored = lambda t: np.flatnonzero(np.isin(frames, t))  # indices into the stored samples
    real = dc.valid_spikes([tr.to_real(stored(t)) for t in (t0, t1, t2)], tr.n_samples, pre, post)
    err = np.abs(solve(real)[0] - fps[0]).max()
    # treating the concatenated snippets as continuous glues unrelated times together
    concat = traces[frames]
    naive = dc.valid_spikes([stored(t) for t in (t0, t1, t2)], len(frames), pre, post)
    Fn, _ = dc.solve_footprints(naive, dc.spike_triggered_sums(concat, naive, pre, post), W)
    err_naive = np.abs(Fn[0] - fps[0]).max()
    assert err < 0.1 and err_naive > 5 * err


def test_triggered_traces_zero_fill_and_per_snippet_filter():
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(2)
    stored = rng.normal(0, 1, (300, 2))
    frames = np.r_[np.arange(10, 110), np.arange(500, 700)]
    sos = butter(2, 300, "highpass", fs=10000, output="sos")
    tr = tg.TriggeredTraces(lambda a, b: stored[a:b], frames, sos=sos)
    out = tr.get_traces(0, tr.n_samples)
    assert tr.n_snippets == 2 and tr.n_samples == 690 and tr.recorded.sum() == 300
    np.testing.assert_allclose(out[100:490], 0)
    pad = 3 * (2 * len(sos) + 1)
    centred = lambda x: x - np.median(x, axis=0)
    np.testing.assert_allclose(out[:100], sosfiltfilt(sos, centred(stored[:100]), axis=0, padlen=pad))
    np.testing.assert_allclose(out[490:], sosfiltfilt(sos, centred(stored[100:]), axis=0, padlen=pad))
    np.testing.assert_allclose(tr.get_traces(50, 520)[:50], out[50:100])


def test_silent_decoys_avoid_the_unit_and_gaps():
    rng = np.random.default_rng(3)
    recorded = rng.random(20000) < 0.5
    train = np.sort(rng.choice(np.flatnonzero(recorded), 50, replace=False))
    d = tg.silent_decoys(train, recorded, 10, 20, rng)
    assert len(d) == 50 and recorded[d].all()
    assert np.min(np.abs(d[:, None] - train[None, :])) >= 30
    assert np.min(np.diff(d)) >= 30


def test_null_footprint_has_no_target_signal():
    rng = np.random.default_rng(4)
    pre, post, n = 5, 25, 200000
    W = pre + post
    fps = {u: rng.normal(0, 5, (W, 3)) for u in range(2)}
    t0 = np.sort(rng.choice(np.arange(100, n - 100), 400, replace=False))
    t1 = np.unique(np.r_[t0 + 4, rng.choice(np.arange(100, n - 100), 200)])  # partner locked to the target
    traces = _traces(fps, {0: t0, 1: t1}, n, pre) + rng.normal(0, 1.0, (n, 3))
    trains = dc.valid_spikes([t0, t1], n, pre, post)
    S = dc.spike_triggered_sums(traces, trains, pre, post)
    F, _ = dc.solve_footprints(trains, S, W)
    decoy = tg.silent_decoys(trains[0], np.ones(n, bool), pre, post, rng)
    S_d = dc.spike_triggered_sums(traces, [decoy], pre, post)[0]
    null, null_avg = dc.null_footprint(decoy, S_d, trains, F, pre)
    sd = 1.0 / np.sqrt(len(decoy))
    assert np.abs(null).max() < 6 * sd  # noise only
    assert np.std(null) < 2 * sd
    assert np.abs(F[0] - fps[0]).max() < 8 * sd  # whereas the target's footprint is recovered


def test_pipeline_deconvolution_on_a_triggered_maxwell_file(tmp_path):
    """Unit A and C trigger the stored snippets, unit B fires inside A's snippets (25 samples after A in 80% of A's
    spikes) and has no snippets of its own. Gap-aware templates match those from the fully stored recording."""
    import sys

    import spikeinterface.full as si

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import synthetic as syn
    from axon_tracking import template_extraction as te

    N = 120000
    rng = np.random.default_rng(0)
    soma_a, axon_a, soma_b, axon_b, soma_c = (syn.electrode_id(c, 10) for c in (20, 30, 40, 50, 60))
    electrodes = np.array([soma_a, axon_a, soma_b, axon_b, soma_c])
    t_a = np.sort(rng.choice(np.arange(1000, N - 1000, 50), 250, replace=False))
    t_b = np.unique(np.r_[t_a[rng.random(len(t_a)) < 0.8] + 25, rng.choice(np.arange(1000, N - 1000), 30)])
    t_c = np.sort(rng.choice(np.arange(1000, N - 1000, 50), 250, replace=False))
    units = [{"spike_times": {("well000", 0): t_a}, "footprint": {soma_a: (-60, 0), axon_a: (-20, 35)}},
             {"spike_times": {("well000", 0): t_b}, "footprint": {soma_b: (-40, 0), axon_b: (-20, 30)}},
             {"spike_times": {("well000", 0): t_c}, "footprint": {soma_c: (-50, 0)}}]
    pre, post = syn.TRIGGER
    keep = np.zeros(N, bool)
    for t in np.r_[t_a, t_c]:
        keep[t - pre : t + post] = True
    stored = np.flatnonzero(keep)
    params = dict(te.DEFAULT_TE_PARAMS, deconvolve=True, null_draws=2)
    cutout = (pre / syn.FS * 1000, post / syn.FS * 1000)
    out = {}
    for name, st in (("continuous", None), ("triggered", stored)):
        path = tmp_path / f"{name}.raw.h5"
        syn.write_maxwell_file(path, {"well000": [electrodes]}, N, units=units, noise_bits=1.0,
                               stored=None if st is None else {("well000", 0): st})
        rec = si.read_maxwell(str(path), stream_id="well000", rec_name="rec0000")
        idx = (lambda t: t) if st is None else (lambda t: np.searchsorted(st, t[np.isin(t, st)]))
        sorting = si.NumpySorting.from_unit_dict([{0: idx(t_a), 1: idx(t_b), 2: idx(t_c)}], syn.FS)
        for mode in (("auto",) if st is None else ("auto", False)):
            out[name, mode] = te.extract_templates_deconvolved(rec, sorting, sorting, dict(params, triggered=mode),
                                                               cutout, return_null=(name == "triggered" and mode))
    reference = out["continuous", "auto"]
    templates, nulls = out["triggered", "auto"]
    err = np.abs(templates - reference).max(axis=(1, 2))
    err_naive = np.abs(out["triggered", False] - reference).max(axis=(1, 2))
    scale = np.abs(reference[0]).max()
    assert np.all(err < 0.06 * scale)
    assert err[1] < 0.3 * err_naive[1]  # the unit without snippets of its own
    # null templates (two independent draws): no signal of the unit itself
    assert nulls.shape == (2,) + templates.shape
    assert np.abs(nulls[:, 0, :, 0]).max() < 0.05 * np.abs(templates[0][:, 0]).max()
    assert not np.allclose(nulls[0], nulls[1])


def test_stored_subset_and_snippet_frames():
    import spikeinterface.full as si

    traces = np.arange(2000 * 3, dtype="float32").reshape(2000, 3)
    rec = si.NumpyRecording([traces], sampling_frequency=10000.0)
    frames = tg.snippet_frames([100, 105, 900, 1995], 10, 20, 2000)
    assert np.array_equal(frames, np.r_[np.arange(90, 125), np.arange(890, 920)])  # 1995 does not fit
    sub = tg.stored_subset(rec, frames)
    assert sub.get_num_samples() == len(frames)
    np.testing.assert_array_equal(sub.get_traces(start_frame=30, end_frame=40), traces[frames[30:40]])
    np.testing.assert_array_equal(sub.get_traces(), traces[frames])


def test_unconstrained_lags_are_left_out_of_the_solve():
    rng = np.random.default_rng(5)
    pre, post, n = 5, 15, 5000
    W = pre + post
    t = np.sort(rng.choice(np.arange(50, n - 50, 40), 60, replace=False))
    recorded = np.zeros(n, bool)
    for s in t:
        recorded[s - pre + 2 : s + post] = True  # the first two lags of every window are never stored
    fp = rng.normal(0, 1, (W, 2))
    traces = np.zeros((n, 2))
    for s in t:
        traces[s - pre : s + post] += fp
    traces[~recorded] = 0
    F, info = dc.solve_footprints([t], dc.spike_triggered_sums(traces, [t], pre, post), W, recorded=recorded, pre=pre)
    assert info["ridge"] == 0 and info["n_unconstrained"] == 2
    np.testing.assert_allclose(F[0, 2:], fp[2:], atol=1e-9)
    np.testing.assert_allclose(F[0, :2], 0)



def test_noise_table_flags_noise_dominated_templates():
    from axon_tracking import template_extraction as te

    rng = np.random.default_rng(8)
    nulls = rng.normal(0, 1.0, (3, 40, 500))
    templates = nulls * np.array([1.05, 1.1, 2.0])[:, None, None]
    table = te.noise_table(templates, nulls, ["a", "b", "c"])
    assert list(table.unit) == ["a", "b", "c"]
    np.testing.assert_allclose(table.noise_inflation, [1.05, 1.1, 2.0], rtol=1e-6)


def test_activity_matched_decoys_follow_the_units_network_state():
    rng = np.random.default_rng(9)
    n = 400000
    recorded = np.ones(n, bool)
    bursts = np.sort(rng.choice(np.arange(5000, n - 5000), 40, replace=False))
    population = np.sort(np.concatenate([b + rng.integers(0, 2000, 300) for b in bursts] + [rng.integers(0, n, 2000)]))
    train = np.sort(np.concatenate([b + rng.integers(0, 2000, 10) for b in bursts]))  # fires in bursts only
    uniform = tg.silent_decoys(train, recorded, 10, 20, rng)
    matched = tg.silent_decoys(train, recorded, 10, 20, rng, population=population)
    act = lambda t: np.median(tg.activity(t, population, 50))
    assert act(matched) > 5 * max(act(uniform), 1)
    assert np.min(np.abs(matched[:, None] - train[None, :])) >= 30
