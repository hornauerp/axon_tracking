"""Simulated cultures and axon scan scenarios with ground truth.

    culture = build_culture(n_neurons=80, seed=0, cache_dir="sim_cache")
    results = run_scenario("bursting", culture, n_targets=10)
    metrics = [evaluate(reconstruct_part3(r["template"])[0], r["ground_truth"]) for r in results]
"""

import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from axon_tracking.simulation.activity import NetworkParams, burst_onsets, simulate_spike_trains
from axon_tracking.simulation.background import background_footprints
from axon_tracking.simulation.currents import Currents, prescribed_currents
from axon_tracking.simulation.evaluation import GroundTruth, ground_truth
from axon_tracking.simulation.forward import N_COLS, N_ROWS, PITCH, Footprint, compute_footprint, electrode_positions
from axon_tracking.simulation.morphology import (CHIP_UM, Morphology, load_swc, random_neuron, with_diameters,
                                                  with_random_dendrites)
from axon_tracking.simulation.recording import axon_scan_configurations, config_sums, sta_template


@dataclass
class SimNeuron:
    morphology: Morphology
    currents: Currents
    footprint: Footprint
    cell_type: str
    velocity_m_s: float


@dataclass
class Culture:
    neurons: list
    common_electrodes: np.ndarray
    seed: int


@dataclass
class Scenario:
    network: NetworkParams = field(default_factory=NetworkParams)
    n_configs: int = 40
    config_duration_s: float = 30.0
    sparse_fraction: float = 1.0
    noise_uv: float = 6.0
    combine: str = "last"
    isolated: bool = False  # only the target's own footprint contributes (no other neurons)
    shuffle_others: bool = False  # shift the other neurons' trains in time: same rates/bursts, no co-activity
    max_triggers: int = None  # use at most this many trigger spikes per configuration
    n_background: int = 0  # cheap background neurons in the same network (simulation.background)
    background_kwargs: dict = field(default_factory=dict)  # for background_footprints


BURSTS = NetworkParams(burst_rate_hz=0.1, jitter_ms=10.0, latency_sd_ms=20.0)
PRECISE_BURSTS = NetworkParams(burst_rate_hz=0.1, jitter_ms=1.0, latency_sd_ms=5.0)

# Scenarios with the same network share identical spike trains (same seed), so they differ only in what is
# changed: isolated (no other neurons), shuffled (others active but not co-active), triggers, sparseness, ...
SCENARIOS = {
    "isolated": Scenario(network=BURSTS, isolated=True),
    "shuffled": Scenario(network=BURSTS, shuffle_others=True),
    "bursting": Scenario(network=BURSTS),
    "bursting_mean": Scenario(network=BURSTS, combine="mean"),
    "few_spikes": Scenario(network=BURSTS, max_triggers=5),
    "sparse": Scenario(network=BURSTS, sparse_fraction=0.5),
    "precise_isolated": Scenario(network=PRECISE_BURSTS, isolated=True),
    "precise_bursting": Scenario(network=PRECISE_BURSTS),
}


def build_culture(n_neurons=80, seed=0, frac_excitatory=0.8, velocity_range=(0.3, 0.8), plane_z=5.0,
                  margin_um=200.0, max_common=360, cache_dir=None, neuron_kwargs=None, current_kwargs=None,
                  morphologies=None, diameter_kwargs=None, cache_tag=None, verbose=False, random_dendrites=None):
    """Neurons with prescribed-propagation currents and footprints (cached in cache_dir if given).

    morphologies: {cell_type: [SWC paths]} to draw reconstructed morphologies from (randomly rotated and
    mirrored, diameters from with_diameters(**diameter_kwargs)); otherwise random_neuron(**neuron_kwargs).
    random_dendrites: {cell_type: with_random_dendrites kwargs, total_length_um as (min, max)} replaces the
    traced dendrites of those cell types (for low-magnification reconstructions that miss most dendrites).
    cache_tag distinguishes cached cultures built with different morphologies or parameters."""
    name = f"culture_n{n_neurons}_seed{seed}" + (f"_{cache_tag}" if cache_tag else "")
    cache = Path(cache_dir) / f"{name}.pkl" if cache_dir else None
    if cache is not None and cache.exists():
        return pickle.load(open(cache, "rb"))
    rng = np.random.default_rng(seed)
    loaded = {}
    neurons = []
    for _ in range(n_neurons):
        soma = rng.uniform([margin_um, margin_um], [CHIP_UM[0] - margin_um, CHIP_UM[1] - margin_um])
        if morphologies:
            cell_type = "excitatory" if rng.random() < frac_excitatory else "inhibitory"
            v = rng.uniform(*velocity_range)
            path = morphologies[cell_type][rng.integers(len(morphologies[cell_type]))]
            if path not in loaded:
                loaded[path] = load_swc(path).coarsened(10.0)  # densely traced cells: ~10 um segments
            dk = dict(diameter_kwargs or {})
            if "axon_um" not in dk:
                # unmyelinated axons: v ~ sqrt(d); the reference (0.7 um at 0.4 m/s) gives axonal amplitudes
                # as recorded from patched neurons on MaxOne (validation.dandi_000294)
                dk["axon_um"] = 0.7 * (v / 0.4) ** 2
            m = loaded[path]
            if random_dendrites and cell_type in random_dendrites:
                rd = dict(random_dendrites[cell_type])
                lo, hi = rd.pop("total_length_um")
                m = with_random_dendrites(m, rng, total_length_um=rng.uniform(lo, hi), **rd)
                dk["dend_um"] = None  # keep the tapering diameters
            m = with_diameters(m, **dk)
            centre = m.mid[m.kind == "soma"].mean(axis=0)[:2]
            morph = m.transformed(rng.uniform(0, 360), np.r_[soma - centre, 0.0], plane_z=plane_z,
                                  flip=bool(rng.random() < 0.5)).subdivided(10.0)
        else:
            morph = random_neuron(rng, soma, plane_z=plane_z, **(neuron_kwargs or {})).subdivided(10.0)
            v = rng.uniform(*velocity_range)
        cur = prescribed_currents(morph, velocity_m_s=v, **(current_kwargs or {}))
        fp = compute_footprint(morph, cur)
        if not morphologies:  # drawn last, as in earlier versions (same cultures for the same seed)
            cell_type = "excitatory" if rng.random() < frac_excitatory else "inhibitory"
        # keep the latencies, not the membrane currents (hundreds of MB for long reconstructed axons)
        cur = replace(cur, imem_na=np.zeros((0, 0), np.float32))
        neurons.append(SimNeuron(morph, cur, fp, cell_type, v))
        if verbose:
            print(f"neuron {len(neurons)}/{n_neurons}: {cell_type}, {len(morph)} segments, "
                  f"{len(fp.electrodes)} electrodes", flush=True)
    # common electrodes: 3x3 blocks around each soma (as the sorted electrodes of an axon scan)
    pos = electrode_positions()
    common = []
    for n in neurons:
        soma = n.morphology.mid[n.morphology.kind == "soma"].mean(axis=0)[:2]
        common += list(np.flatnonzero(np.max(np.abs(pos - soma), axis=1) <= PITCH * 1.01))
    common = np.unique(common)[:max_common]
    culture = Culture(neurons, common, seed)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(culture, open(cache, "wb"))
    return culture


def run_scenario(scenario, culture: Culture, n_targets=10, seed=0, targets=None, k_detect=3.0, window_ms=5.0):
    """Templates and ground truth of target neurons for one scenario (name in SCENARIOS or a Scenario).
    window_ms: end of the template window; axon reached later is left out of the ground truth."""
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    neurons = culture.neurons
    duration = sc.n_configs * sc.config_duration_s
    network = replace(sc.network, duration_s=duration)
    onsets = burst_onsets(network, rng)
    trains = dict(enumerate(simulate_spike_trains([n.cell_type for n in neurons], network, rng, onsets=onsets)))
    background = {}
    if sc.n_background:
        brng = np.random.default_rng([seed, 2])  # separate stream: the culture's trains and configurations stay
        fps = background_footprints(sc.n_background, brng, **sc.background_kwargs)
        types = ["excitatory" if brng.random() < 0.8 else "inhibitory" for _ in fps]
        for k, (fp, tr) in enumerate(zip(fps, simulate_spike_trains(types, network, brng, onsets=onsets))):
            background[len(neurons) + k], trains[len(neurons) + k] = fp, tr
    configs = axon_scan_configurations(sc.n_configs, sc.config_duration_s, culture.common_electrodes,
                                       sc.sparse_fraction, rng=rng)
    if targets is None:
        targets = rng.choice(len(neurons), size=min(n_targets, len(neurons)), replace=False)
    results = []
    for t in targets:
        trains_t = dict(trains)
        if sc.shuffle_others:
            for j in trains_t:
                if j != t:
                    trains_t[j] = np.sort((trains[j] + rng.uniform(0, duration)) % duration)
        others = {} if sc.isolated else {**{i: n.footprint for i, n in enumerate(neurons) if i != t}, **background}
        kwargs = dict(configs=configs, combine=sc.combine, max_triggers=sc.max_triggers)
        template, n_per_config = sta_template(t, trains_t, {t: neurons[t].footprint, **others}, noise_uv=sc.noise_uv,
                                              rng=np.random.default_rng([seed, int(t)]), **kwargs)
        # contamination: noise-free template of the other neurons only (same triggers)
        contamination = sta_template(t, trains_t, others, noise_uv=0.0, rng=np.random.default_rng([seed, int(t)]),
                                     **kwargs)[0] if others else np.zeros_like(template)
        n_med = np.median([n for n in n_per_config if n > 0]) if any(n_per_config) else 1
        gt = ground_truth(neurons[t].morphology, neurons[t].currents.latency_ms, neurons[t].footprint,
                          template_noise_uv=sc.noise_uv / np.sqrt(n_med),
                          others=[n.morphology for i, n in enumerate(neurons) if i != t], k_detect=k_detect,
                          contamination_p2p=np.ptp(contamination, axis=2), max_latency_ms=window_ms - 0.3)
        results.append(dict(unit=int(t), cell_type=neurons[t].cell_type, velocity_m_s=neurons[t].velocity_m_s,
                            template=template, ground_truth=gt, n_per_config=n_per_config,
                            n_spikes=len(trains[t])))
    return results


def run_deconvolution(scenario, culture: Culture, targets, seed=0, sorted_fraction=1.0, k_detect=3.0, pre=30,
                      post=80, out_pre=30, out_post=50, ridge=0.0):
    """Spike-triggered averages and least-squares footprints (axon_tracking.deconvolution) of the targets from
    the same simulated recordings.

    Per configuration, the spike-triggered sums of all sorted units get noise with the covariance of white
    recording noise (sigma^2 G, G the Gram matrix of the sorted spike trains). sorted_fraction < 1: only a random
    subset of the other neurons is sorted (the rest still contributes to the recording). The deconvolution window
    [-pre, post) must contain the footprints; templates are cropped to [-out_pre, out_post).
    Returns one dict per target with template_sta, template_deconv and ground_truth (as run_scenario)."""
    import scipy.linalg

    from axon_tracking.deconvolution import gram_matrix

    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    neurons = culture.neurons
    fs = 10000.0
    duration = sc.n_configs * sc.config_duration_s
    trains = simulate_spike_trains([n.cell_type for n in neurons], replace(sc.network, duration_s=duration), rng)
    trains = {i: np.unique(np.rint(t * fs)) / fs for i, t in enumerate(trains)}  # on the sample grid
    configs = axon_scan_configurations(sc.n_configs, sc.config_duration_s, culture.common_electrodes,
                                       sc.sparse_fraction, rng=rng)
    targets = [int(t) for t in targets]
    others = [i for i in range(len(neurons)) if i not in targets]
    n_sorted_others = int(round(sorted_fraction * len(others)))
    sorted_units = sorted(targets + list(rng.choice(others, n_sorted_others, replace=False)))
    footprints = {i: n.footprint for i, n in enumerate(neurons)}
    W, crop = pre + post, slice(pre - out_pre, pre + out_post)
    grids = {t: {k: np.zeros((N_COLS * N_ROWS, out_pre + out_post)) for k in ("sta", "deconv", "contamination")}
             for t in targets}
    n_per_config = {t: [] for t in targets}
    noise_rng = np.random.default_rng([seed, 1])
    for cfg in configs:
        S, own, regressors = config_sums(sorted_units, trains, footprints, cfg, fs, pre, post)
        n_spikes = np.array([len(r) for r in regressors])
        active = np.flatnonzero(n_spikes > 0)
        G = gram_matrix([regressors[k] for k in active], W)
        chol = scipy.linalg.cholesky(G, lower=True)
        noise = sc.noise_uv * (chol @ noise_rng.standard_normal((len(G), S.shape[2])))
        S_noisy = S[active] + noise.reshape(len(active), W, -1)
        F = scipy.linalg.cho_solve((chol, True), S_noisy.reshape(len(G), -1)) if ridge == 0 else \
            scipy.linalg.solve(G + ridge * np.eye(len(G)), S_noisy.reshape(len(G), -1), assume_a="pos")
        F = F.reshape(len(active), W, -1)
        for t in targets:
            k = sorted_units.index(t)
            n_per_config[t].append(int(n_spikes[k]))
            if n_spikes[k] == 0:
                continue
            a = int(np.flatnonzero(active == k)[0])
            grids[t]["sta"][cfg.electrodes] = (S_noisy[a] / n_spikes[k])[crop].T
            grids[t]["deconv"][cfg.electrodes] = F[a][crop].T
            grids[t]["contamination"][cfg.electrodes] = ((S[k] - own[k]) / n_spikes[k])[crop].T
    results = []
    for t in targets:
        to_grid = lambda x: x.reshape(N_ROWS, N_COLS, -1).transpose(1, 0, 2).astype(np.float32)
        n_med = np.median([n for n in n_per_config[t] if n > 0])
        gt = ground_truth(neurons[t].morphology, neurons[t].currents.latency_ms, neurons[t].footprint,
                          template_noise_uv=sc.noise_uv / np.sqrt(n_med),
                          others=[n.morphology for i, n in enumerate(neurons) if i != t], k_detect=k_detect,
                          contamination_p2p=np.ptp(to_grid(grids[t]["contamination"]), axis=2),
                          max_latency_ms=out_post / fs * 1000 - 0.3)
        results.append(dict(unit=t, cell_type=neurons[t].cell_type, velocity_m_s=neurons[t].velocity_m_s,
                            template_sta=to_grid(grids[t]["sta"]), template_deconv=to_grid(grids[t]["deconv"]),
                            ground_truth=gt, n_per_config=n_per_config[t], n_sorted=len(sorted_units)))
    return results


def reconstruct_part3(template, params=None, skel_params=None, qc_params=None):
    """Reconstruction as in notebooks/part3_skeletonization.ipynb (without fragment linking).

    Returns (paths in um (list of (n, 3) arrays: x, y, t), branch points in um (m, 2))."""
    import kimimaro
    from axon_tracking import skeletonization as skel

    params = params or dict(noise_threshold=-0.75, abs_threshold=-0.1, upsample=[0.5, 0.5, 0.25],
                            sampling_rate=10000 / 0.25, ms_cutout=[3.0, 5.0], max_velocity=0.8,
                            el_spacing=17.5 * 0.5, ais_detection="dev", buffer_frames=10)
    skel_params = skel_params or dict(scale=2, const=10, dust_threshold=10, anisotropy=(1, 1, 1), n_jobs=1)
    qc_params = qc_params or dict(window_size=5, max_duplicate_ratio=0.3, min_r2=0.9, vel_range=[0.3, 0.8],
                                  min_length=1)
    try:
        interp, noise, ais = skel.preprocess_template(template.astype("float64"), params)
    except ValueError:
        return [], np.zeros((0, 2))
    th = skel.threshold_template(interp, noise, ais, params)
    skels = skel.skeletonize(th.astype(bool), **skel_params)
    if 1 not in skels:
        return [], np.zeros((0, 2))
    skeleton = kimimaro.postprocess(skels[1], tick_threshold=5, dust_threshold=10)
    paths, _, _, _ = skel.perform_path_qc(skel.branches_from_paths(skeleton), params, **qc_params)
    branch_points = skeleton.vertices[skeleton.branches(), :2] * params["el_spacing"]
    return paths, branch_points
