import numpy as np

from axon_tracking import deconvolution as dc
from axon_tracking.simulation import raw, recording, scenarios


def _small_scan():
    culture = scenarios.build_culture(n_neurons=4, seed=5, neuron_kwargs=dict(axon_length=(800, 800)))
    rng = np.random.default_rng(1)
    trains = {i: np.unique(np.rint(np.sort(rng.uniform(0, 8, 40)) * 1e4)) / 1e4 for i in range(4)}
    trains[1] = np.unique(np.concatenate([trains[1], trains[0] + 0.002]))  # partly locked to neuron 0
    configs = recording.axon_scan_configurations(2, 4.0, culture.common_electrodes, rng=rng)
    return culture, trains, configs


def test_traces_do_not_depend_on_chunking():
    culture, trains, configs = _small_scan()
    rec = raw.axon_scan_recordings(culture, trains, configs, noise_uv=6.0)[0][0]
    whole = rec.get_traces(start_frame=0, end_frame=25000)
    parts = np.concatenate([rec.get_traces(start_frame=0, end_frame=12345),
                            rec.get_traces(start_frame=12345, end_frame=25000)])
    np.testing.assert_array_equal(whole, parts)
    ids = rec.get_channel_ids()
    np.testing.assert_array_equal(rec.get_traces(start_frame=0, end_frame=10, channel_ids=[ids[3], ids[1]]),
                                  whole[:10, [3, 1]])


def test_spike_triggered_average_of_simulated_traces_matches_the_analytic_one():
    culture, trains, configs = _small_scan()
    recs, samples = raw.axon_scan_recordings(culture, trains, configs, noise_uv=0.0)
    pre, post = 30, 50
    footprints = {i: n.footprint for i, n in enumerate(culture.neurons)}
    analytic, n_per_config = recording.sta_template(0, trains, footprints, configs[:1], noise_uv=0.0)
    rec = recs[0]
    t = dc.valid_spikes([samples[0][0]], rec.get_num_samples(), pre, post)[0]
    S = dc.spike_triggered_sums_chunked(lambda a, b: rec.get_traces(start_frame=a, end_frame=b),
                                        rec.get_num_samples(), [t], pre, post, chunk_samples=7000)
    sta = S[0] / len(t)
    e = configs[0].electrodes
    grid = analytic.transpose(1, 0, 2).reshape(-1, pre + post)[e]  # (electrodes, samples)
    assert len(t) == n_per_config[0]
    np.testing.assert_allclose(sta.T, grid, atol=1e-3 * np.abs(grid).max())


def test_burst_attenuation_and_correlated_noise():
    t = np.array([0, 100, 150, 200, 5000, 5100])  # bursts at 10 kHz: intervals < 20 ms
    f = raw.spike_amplitudes(t, np.random.default_rng(0), burst_attenuation=0.2)
    np.testing.assert_allclose(f, [1, 0.8, 0.64, 0.512, 1, 0.8], rtol=1e-6)
    np.testing.assert_allclose(raw.spike_amplitudes(np.arange(0, 2000, 150), np.random.default_rng(0), burst_attenuation=0.5),
                               np.r_[1, np.full(13, 0.5)])  # floor at min_factor
    electrodes = np.arange(100, 140)  # one row, 17.5 um apart
    rec = raw.make_recording(electrodes, {}, {}, 30000, noise_uv=1.0, noise_correlation_um=35.0)
    x = rec.get_traces(start_frame=0, end_frame=30000)
    c = np.corrcoef(x.T)
    assert abs(np.mean(np.diag(c, 1)) - np.exp(-17.5 / 35.0)) < 0.03
    assert abs(np.std(x) - 1.0) < 0.03


def test_observed_sorting_misses_overlapping_spikes():
    culture, trains, configs = _small_scan()
    _, samples = raw.axon_scan_recordings(culture, trains, configs, noise_uv=0.0)
    obs = raw.observed_sorting(samples[0], [0, 1], culture, np.random.default_rng(0), p_miss_overlap=1.0, p_miss=0.0,
                               overlap_radius_um=1e6, overlap_ms=3.0)
    assert set(obs) == {0, 1}
    assert len(obs[1]) < len(samples[0][1])  # locked spikes overlap with neuron 0 and are missed
