import numpy as np
import pytest

from axon_tracking.simulation import (activity, background, currents, evaluation, forward, morphology, recording,
                                      scenarios)


@pytest.fixture(scope="module")
def neuron():
    rng = np.random.default_rng(1)
    m = morphology.random_neuron(rng, soma_xy=(1900, 1000), axon_length=(1500, 1500)).subdivided(10)
    cur = currents.prescribed_currents(m, velocity_m_s=0.5)
    return m, cur, forward.compute_footprint(m, cur)


def test_random_neuron_structure(neuron):
    m, _, _ = neuron
    assert set(m.kind) == {"soma", "axon", "dend"}
    assert np.all(m.parent < np.arange(len(m)))
    assert m.length.max() <= 10 + 1e-6
    d = m.path_distance("axon")
    axon = np.flatnonzero(m.kind == "axon")
    child = [i for i in axon if m.parent[i] >= 0 and m.kind[m.parent[i]] == "axon"]
    assert np.all(d[child] > d[m.parent[child]])


def test_subdivided_preserves_length_and_tree():
    rng = np.random.default_rng(3)
    m = morphology.random_neuron(rng, soma_xy=(500, 500), axon_length=(800, 800))
    s = m.subdivided(4.0)
    assert np.isclose(s.length.sum(), m.length.sum())
    assert np.all(s.parent < np.arange(len(s)))
    assert len(s.branch_points()) == len(m.branch_points())


def test_coarsened_keeps_tree_structure():
    rng = np.random.default_rng(5)
    m = morphology.random_neuron(rng, soma_xy=(800, 800), axon_length=(1000, 1000)).subdivided(1.0)  # dense
    c = m.coarsened(10.0)
    assert len(c) < len(m) / 5
    assert np.all(c.parent < np.arange(len(c)))
    assert len(c.branch_points("axon")) == len(m.branch_points("axon"))
    for kind in ("axon", "dend"):
        assert np.isclose(c.length[c.kind == kind].sum(), m.length[m.kind == kind].sum(), rtol=0.02)


def test_diameters_flip_and_type_map(tmp_path):
    swc = tmp_path / "cell.swc"
    swc.write_text("1 1 0 0 5 6 -1\n2 4 10 0 5 0.5 1\n3 2 50 0 5 0.5 2\n4 2 90 20 5 0.5 3\n5 3 -30 0 5 1 1\n")
    m = morphology.load_swc(swc, type_map={4: "axon"})
    assert (m.kind == "axon").sum() == 3
    d = morphology.with_diameters(m, axon_um=0.5, ais_um=1.0, ais_length_um=20, dend_um=2.0)
    axon = np.flatnonzero(m.kind == "axon")
    assert d.diam[axon[0]] == 1.0 and d.diam[axon[-1]] == 0.5 and d.diam[m.kind == "dend"][0] == 2.0
    flipped = m.transformed(flip=True)
    soma = m.mid[m.kind == "soma"].mean(axis=0)
    np.testing.assert_allclose(flipped.end[:, 0] - soma[0], -(m.end[:, 0] - soma[0]))


def test_ground_truth_density_and_window():
    # one straight 100 um axon as 1000 tiny segments: the ground truth is sampled every ~2 um, not per segment
    n = 1000
    x = np.linspace(0, 100, n + 1)
    start = np.column_stack([x[:-1], np.zeros(n), np.full(n, 5.0)])
    end = np.column_stack([x[1:], np.zeros(n), np.full(n, 5.0)])
    soma = np.array([[-10, 0, 5.0]]), np.array([[0, 0, 5.0]])
    m = morphology.Morphology(np.vstack([soma[0], start]), np.vstack([soma[1], end]), np.full(n + 1, 1.0),
                              np.array(["soma"] + ["axon"] * n), np.arange(n + 1) - 1)
    latency = np.r_[np.nan, x[1:] / 100.0]  # 1 ms at the end
    gt = evaluation.ground_truth(m, latency)
    assert 45 <= len(gt.axon_points) <= 55
    gt_window = evaluation.ground_truth(m, latency, max_latency_ms=0.5)
    assert gt_window.axon_points[:, 0].max() <= 50.5


def test_asymmetric_ap_waveform():
    t = np.arange(-2, 4, 0.005)
    f_sym, df_sym, ddf_sym = currents.ap_waveform(t)
    f_num, df_num, ddf_num = currents.ap_waveform(t, decay_width_ms=0.15 + 1e-9)  # numerical path, ~symmetric
    np.testing.assert_allclose(df_num, df_sym, atol=1e-3 * np.abs(df_sym).max())
    np.testing.assert_allclose(ddf_num, ddf_sym, atol=1e-2 * np.abs(ddf_sym).max())
    f, df, _ = currents.ap_waveform(t, width_ms=0.15, decay_width_ms=0.3)
    assert abs(t[np.argmax(f)]) < 0.01  # peak at 0
    assert df.max() > 1.5 * -df.min()  # faster rise than repolarisation


def test_dendritic_bap_conserves_current(neuron):
    m, _, _ = neuron
    cur = currents.prescribed_currents(m, velocity_m_s=0.5, dendrite_velocity_m_s=0.3,
                                       dendrite_length_constant_um=50, return_length_constant_um=50)
    assert np.abs(cur.imem_na.sum(axis=0)).max() < 1e-9 * np.abs(cur.imem_na).max()
    dend = m.kind == "dend"
    assert np.abs(cur.imem_na[dend]).max() > 0


def test_load_swc(tmp_path):
    swc = tmp_path / "cell.swc"
    swc.write_text("1 1 0 0 5 6 -1\n2 2 10 0 5 0.5 1\n3 2 50 0 5 0.5 2\n4 2 90 20 5 0.5 3\n"
                   "5 2 90 -20 5 0.5 3\n6 3 -30 0 5 1 1\n")
    m = morphology.load_swc(swc)
    assert m.kind[0] == "soma" and (m.kind == "axon").sum() == 4 and (m.kind == "dend").sum() == 1
    assert len(m.branch_points("axon")) == 1
    np.testing.assert_allclose(m.branch_points("axon")[0, :2], [50, 0])


def test_prescribed_currents_are_conserved_and_propagate(neuron):
    m, cur, _ = neuron
    assert np.abs(cur.imem_na.sum(axis=0)).max() < 1e-9
    d = m.path_distance("axon")
    axon = m.kind == "axon"
    np.testing.assert_allclose(np.diff(cur.latency_ms[axon][np.argsort(d[axon])]) >= -1e-12, True)


def test_line_source_matches_lfpykit(neuron):
    lfpykit = pytest.importorskip("lfpykit")
    m, _, _ = neuron
    pos = forward.electrode_positions()
    e = np.flatnonzero(np.linalg.norm(pos - [1900, 1000], axis=1) < 200)[:200]
    epos = np.column_stack([pos[e], np.zeros(len(e))])
    cell = lfpykit.CellGeometry(x=np.column_stack([m.start[:, 0], m.end[:, 0]]),
                                y=np.column_stack([m.start[:, 1], m.end[:, 1]]),
                                z=np.column_stack([m.start[:, 2], m.end[:, 2]]), d=m.diam)
    ref = lfpykit.RecExtElectrode(cell, sigma=1.5, x=epos[:, 0], y=epos[:, 1], z=epos[:, 2],
                                  method="linesource").get_transformation_matrix()
    np.testing.assert_allclose(forward.line_source_matrix(epos, m.start, m.end, m.diam, 1.5), ref, rtol=1e-9, atol=1e-15)


def test_footprint_waveforms(neuron):
    m, _, fp = neuron
    pos = forward.electrode_positions()[fp.electrodes]
    d_soma = np.linalg.norm(pos - [1900, 1000], axis=1)
    amp = np.ptp(fp.data_uv, axis=1)
    soma_w = fp.data_uv[np.argmax(amp)]
    assert np.argmin(soma_w) < np.argmax(soma_w)  # somatic: negative first
    far = np.flatnonzero(d_soma > 300)
    w = fp.data_uv[far[np.argmax(amp[far])]]
    trough = np.argmin(w)
    assert w[:trough].max() > 0 and w[trough:].max() > 0  # axonal: triphasic (+, -, +)
    assert 1 < amp[far].max() < 100  # small axonal signals (uV)


def test_spike_trains_bursts_and_refractoriness():
    rng = np.random.default_rng(0)
    p = activity.NetworkParams(duration_s=200, burst_rate_hz=0.5, cell_types={
        "always": activity.CellTypeParams(tonic_rate_hz=0.0, burst_participation=1.0, spikes_per_burst=3),
        "never": activity.CellTypeParams(tonic_rate_hz=1.0, burst_participation=0.0)})
    trains = activity.simulate_spike_trains(["always", "never"], p, rng)
    assert all(np.all(np.diff(t) >= p.refractory_ms / 1000 - 1e-12) for t in trains)
    assert len(trains[0]) > 2 * 0.5 * 200  # ~3 spikes in each of ~100 bursts
    assert abs(len(trains[1]) - 200) < 60  # tonic only


def test_given_burst_onsets_reproduce_the_trains_and_are_shared():
    p = activity.NetworkParams(duration_s=200, burst_rate_hz=0.5)
    types = ["excitatory"] * 3
    rng = np.random.default_rng(1)
    onsets = activity.burst_onsets(p, rng)
    same = activity.simulate_spike_trains(types, p, rng, onsets=onsets)
    default = activity.simulate_spike_trains(types, p, np.random.default_rng(1))
    assert all(np.array_equal(a, b) for a, b in zip(same, default))
    other = activity.simulate_spike_trains(types, p, np.random.default_rng(5), onsets=onsets)
    # neurons added later burst with the same network (spikes within 100 ms after the shared onsets)
    near = lambda t: np.mean(np.min(np.abs(t[:, None] - onsets[None, :] - 0.05), axis=1) < 0.1)
    assert near(other[0]) > 0.6


def test_background_footprints_decay_and_propagate():
    fps = background.background_footprints(20, np.random.default_rng(0), axon_length_um=(500.0, 1000.0),
                                           velocity_m_s=(0.5, 0.5))
    pos = forward.electrode_positions()
    for fp in fps:
        assert fp.data_uv.shape == (len(fp.electrodes), 80) and np.all(np.diff(fp.electrodes) > 0)
        peak_el = fp.electrodes[np.argmin(fp.data_uv.min(axis=1))]
        d = np.linalg.norm(pos[fp.electrodes] - pos[peak_el], axis=1)
        amp = -fp.data_uv.min(axis=1)
        assert amp[d > 60].max(initial=0) < amp.max()  # largest at the soma
    # latency along the axon: electrodes far from the soma peak later
    fp = max(fps, key=lambda f: len(f.electrodes))
    lat = np.argmin(fp.data_uv, axis=1)
    peak_el = fp.electrodes[np.argmin(fp.data_uv.min(axis=1))]
    d = np.linalg.norm(pos[fp.electrodes] - pos[peak_el], axis=1)
    far = d > 300
    assert far.any() and np.median(lat[far]) > np.median(lat[d < 40]) + 5


def test_axon_scan_configurations_cover_the_array():
    common = np.arange(100)
    configs = recording.axon_scan_configurations(40, 30.0, common=common)
    assert all(len(c.electrodes) <= 1020 and np.all(np.isin(common, c.electrodes)) for c in configs)
    assert len(np.unique(np.concatenate([c.electrodes for c in configs]))) == 26400
    sparse = recording.axon_scan_configurations(40, 30.0, sparse_fraction=0.5)
    assert len(np.unique(np.concatenate([c.electrodes for c in sparse]))) < 0.6 * 26400


def _toy_footprint(electrodes, shape):
    data = np.zeros((len(electrodes), 40), np.float32)
    data[:, 10:10 + len(shape)] = shape
    return forward.Footprint(data_uv=data, electrodes=np.asarray(electrodes), fs=10000.0, spike_index=10)


def test_sta_template_superposition_is_exact():
    rng = np.random.default_rng(0)
    configs = [recording.ScanConfig(np.array([0, 1]), 0.0, 100.0)]
    trig = np.arange(1.0, 99.0, 0.5)
    locked = trig + 0.0005  # always 5 samples after the trigger
    random = np.sort(rng.uniform(0, 100, len(trig)))
    fps = {0: _toy_footprint([0], [-10, 5]), 1: _toy_footprint([1], [-4]), 2: _toy_footprint([1], [-100])}
    grid, n = recording.sta_template(0, {0: trig, 1: locked, 2: random}, fps, configs, pre=30, post=50, noise_uv=0.0)
    assert n == [len(trig)]
    np.testing.assert_allclose(grid[0, 0, 30:32], [-10, 5])  # own spike at the trigger
    assert abs(grid[1, 0, 35] + 4) < 0.5  # locked partner at +5 samples, full weight
    assert np.abs(grid[1, 0, :]).max() < 10  # the random partner's -100 uV is averaged out


def test_evaluation_perfect_and_shifted():
    rng = np.random.default_rng(2)
    m = morphology.random_neuron(rng, soma_xy=(1000, 800), axon_length=(1000, 1000)).subdivided(10)
    lat = currents.prescribed_currents(m).latency_ms
    gt = evaluation.ground_truth(m, lat)
    axon = m.kind == "axon"
    paths = [np.vstack([m.start[i, :2], m.end[i, :2]]) for i in np.flatnonzero(axon)]
    perfect = evaluation.evaluate(paths, gt)
    assert perfect["precision"] > 0.99 and perfect["recall"] > 0.99
    shifted = evaluation.evaluate([p + [300, 0] for p in paths], gt)
    assert shifted["precision"] < 0.5


def test_part3_reconstructs_an_isolated_simulated_neuron():
    """Realistic template noise (6 uV raw, ~30 spikes per configuration). With much lower noise the current
    thresholds (relative to each electrode's template MAD) over-detect the far field and precision drops."""
    culture = scenarios.build_culture(n_neurons=3, seed=4, neuron_kwargs=dict(axon_length=(2000, 2000)))
    res = scenarios.run_scenario(scenarios.Scenario(network=activity.NetworkParams(burst_rate_hz=0.0,
                                 cell_types={"excitatory": activity.CellTypeParams(tonic_rate_hz=1.0),
                                             "inhibitory": activity.CellTypeParams(tonic_rate_hz=1.0)}),
                                 isolated=True, noise_uv=6.0), culture, targets=[0])[0]
    paths, _ = scenarios.reconstruct_part3(res["template"])
    m = evaluation.evaluate(paths, res["ground_truth"])
    assert m["precision"] > 0.7 and m["recall_detectable"] > 0.6


def test_scenarios_share_trains_and_contamination_is_the_others_part():
    """Same seed: the bursting template is exactly the isolated template plus the other neurons' contamination."""
    culture = scenarios.build_culture(n_neurons=4, seed=5, neuron_kwargs=dict(axon_length=(800, 800)))
    kw = dict(network=scenarios.BURSTS, n_configs=4, config_duration_s=60.0)
    iso = scenarios.run_scenario(scenarios.Scenario(isolated=True, **kw), culture, targets=[0], seed=3)[0]
    burst = scenarios.run_scenario(scenarios.Scenario(**kw), culture, targets=[0], seed=3)[0]
    assert iso["n_per_config"] == burst["n_per_config"]
    assert np.all(iso["ground_truth"].contamination_p2p == 0)
    contamination = burst["template"] - iso["template"]
    np.testing.assert_allclose(np.ptp(contamination, axis=2), burst["ground_truth"].contamination_p2p, atol=1e-3)
    assert burst["ground_truth"].contamination_p2p.max() > 1.0  # co-active neurons do contaminate
    few = scenarios.run_scenario(scenarios.Scenario(max_triggers=3, **kw), culture, targets=[0], seed=3)[0]
    assert max(few["n_per_config"]) <= 3


def test_neuron_simulation_runs_if_installed(tmp_path):
    """Needs NEURON and the axon_velocity mechanisms compiled with nrnivmodl in the folder given by the
    environment variable AXON_TRACKING_NRN_MECHANISMS."""
    import os

    pytest.importorskip("neuron")
    mechanisms = os.environ.get("AXON_TRACKING_NRN_MECHANISMS")
    if not mechanisms:
        pytest.skip("set AXON_TRACKING_NRN_MECHANISMS to the compiled axon_velocity mechanisms")
    swc = tmp_path / "ball_and_stick.swc"
    lines = ["1 1 0 0 0 8 -1", "2 3 -8 0 0 1 1", "3 3 -150 0 0 1 2"]  # soma, dendrite
    lines += [f"{4 + k} 2 {8 + 50 * k} 0 0 0.5 {1 if k == 0 else 3 + k}" for k in range(20)]  # 950 um axon
    swc.write_text("\n".join(lines) + "\n")
    m, cur = currents.simulate_neuron(swc, mechanisms, plane_z=0.0)
    axon = m.kind == "axon"
    assert np.all(np.isfinite(cur.latency_ms[axon]))  # the AP reaches the end of the axon
    dist = m.path_distance("axon")[axon]
    velocity = np.polyfit(cur.latency_ms[axon], dist, 1)[0] / 1000
    assert 0.1 < velocity < 2.0
    assert np.abs(cur.imem_na.sum(axis=0)).max() < 1e-6 * np.abs(cur.imem_na).max()
