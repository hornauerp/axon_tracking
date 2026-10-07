"""Axon scan recordings: electrode configurations and spike-triggered average (STA) templates.

The STA of a target unit in one configuration is computed exactly by linear superposition: for every neuron j,
the number of its spikes at each sample offset from the target's trigger spikes (cross-correlogram counts,
including the target's own spikes) is convolved with its footprint; white noise averaged over the N triggers
is added (sd noise_uv / sqrt(N)). Templates have the layout of the pipeline: (220, 120, pre + post) at 10 kHz,
zeros for electrodes that were never recorded.
"""

from dataclasses import dataclass

import numpy as np

from axon_tracking.simulation.forward import N_COLS, N_ROWS, Footprint


@dataclass
class ScanConfig:
    electrodes: np.ndarray  # electrode ids routed in this configuration
    start_s: float
    duration_s: float


def axon_scan_configurations(n_configs=40, duration_s=30.0, common=(), sparse_fraction=1.0, max_channels=1020,
                             rng=None):
    """Tile the array into blocks (one per configuration), each recorded together with the common electrodes.
    sparse_fraction < 1 records only a random subset of every block (sparse configurations)."""
    rng = rng or np.random.default_rng(0)
    ny = max(int(round(np.sqrt(n_configs * N_ROWS / N_COLS))), 1)
    nx = int(np.ceil(n_configs / ny))
    cols = np.linspace(0, N_COLS, nx + 1).round().astype(int)
    rows = np.linspace(0, N_ROWS, ny + 1).round().astype(int)
    common = np.unique(np.asarray(common, int))
    configs = []
    for j in range(ny):
        for i in range(nx):
            c, r = np.meshgrid(np.arange(cols[i], cols[i + 1]), np.arange(rows[j], rows[j + 1]), indexing="ij")
            block = (r * N_COLS + c).ravel()
            block = np.setdiff1d(block, common)
            n_keep = min(int(round(sparse_fraction * len(block))), max_channels - len(common))
            block = np.sort(rng.choice(block, n_keep, replace=False)) if n_keep < len(block) else block
            configs.append(ScanConfig(np.union1d(block, common), len(configs) * duration_s, duration_s))
    return configs


def _offset_counts(triggers, spikes, fs, o_min, o_max):
    """Counts of (trigger, spike) pairs per sample offset in [o_min, o_max]."""
    lo = np.searchsorted(spikes, triggers + (o_min - 0.5) / fs)
    hi = np.searchsorted(spikes, triggers + (o_max + 0.5) / fs)
    n = hi - lo
    counts = np.zeros(o_max - o_min + 1)
    if n.sum() == 0:
        return counts
    idx = np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)  # all of lo[k]..hi[k]-1
    offsets = np.rint((spikes[idx] - np.repeat(triggers, n)) * fs).astype(int) - o_min
    ok = (offsets >= 0) & (offsets < len(counts))
    return counts + np.bincount(offsets[ok], minlength=len(counts))


def config_sums(units, trains, footprints, cfg, fs=10000.0, pre=30, post=80):
    """Noise-free spike-triggered sums of `units` in one configuration (as deconvolution.spike_triggered_sums on
    the recorded traces): S[i, a, e] = sum over spikes of units[i] of the signal of all footprints at lag a - pre.

    Spikes are regressors if their window lies within the configuration. Returns (S (n, W, electrodes),
    own (n, W, electrodes): the part of S from each unit's own footprint, spike trains in samples relative to
    the configuration start (for deconvolution.gram_matrix))."""
    from scipy.signal import fftconvolve

    W = pre + post
    t0, t1 = cfg.start_s, cfg.start_s + cfg.duration_s
    e = cfg.electrodes
    row_of = {}
    for unit, fp in footprints.items():
        lookup = np.full(N_COLS * N_ROWS, -1)
        lookup[fp.electrodes] = np.arange(len(fp.electrodes))
        rows = lookup[e]
        row_of[unit] = (np.flatnonzero(rows >= 0), rows[rows >= 0])
    S = np.zeros((len(units), W, len(e)))
    own = np.zeros_like(S)
    regressors = []
    for k, i in enumerate(units):
        trig = trains[i][(trains[i] >= t0 + pre / fs) & (trains[i] < t1 - post / fs)]
        regressors.append(np.rint((trig - t0) * fs).astype(np.int64))
        if not len(trig):
            continue
        for unit, fp in footprints.items():
            present, rows = row_of[unit]
            if not len(present):
                continue
            L, si = fp.data_uv.shape[1], fp.spike_index
            o_min, o_max = -pre + si - (L - 1), W - 1 - pre + si
            s = trains[unit]
            s = s[(s >= t0 + o_min / fs - 1e-3) & (s < t1 + o_max / fs + 1e-3)]
            counts = _offset_counts(trig, s, fs, o_min, o_max)
            if not counts.any():
                continue
            conv = fftconvolve(fp.data_uv[rows].astype(float), counts[None, :], axes=1)[:, L - 1 : L - 1 + W]
            S[k][:, present] += conv.T
            if unit == i:
                own[k][:, present] += conv.T
    return S, own, regressors


def sta_template(target, trains, footprints, configs, fs=10000.0, pre=30, post=50, noise_uv=6.0, combine="last",
                 rng=None, max_triggers=None):
    """Spike-triggered average template of `target` over an axon scan.

    trains: {unit: spike times (s)}; footprints: {unit: Footprint} (the target's own footprint may be left out
    to obtain the contamination by other neurons only). combine: 'last' (a later configuration overwrites
    electrodes recorded before, as in the current pipeline) or 'mean' (spike-count weighted mean).
    max_triggers: use at most this many (random) trigger spikes per configuration.
    Returns (template (220, 120, pre+post) float32, triggers per configuration).
    """
    from scipy.signal import fftconvolve

    rng = rng or np.random.default_rng(0)
    W = pre + post
    order = {unit: np.argsort(fp.electrodes) for unit, fp in footprints.items()}

    def footprint_rows(unit, e):
        """Indices into e of the electrodes in the unit's footprint, and their footprint rows."""
        el = footprints[unit].electrodes[order[unit]]
        k = np.minimum(np.searchsorted(el, e), len(el) - 1)
        present = np.flatnonzero(el[k] == e)
        return present, order[unit][k[present]]

    total = np.zeros((N_COLS * N_ROWS, W))
    weight = np.zeros(N_COLS * N_ROWS)
    n_per_config = []
    for cfg in configs:
        t0, t1 = cfg.start_s, cfg.start_s + cfg.duration_s
        trig = trains[target]
        trig = trig[(trig >= t0 + pre / fs) & (trig < t1 - post / fs)]
        if max_triggers is not None and len(trig) > max_triggers:
            trig = np.sort(rng.choice(trig, max_triggers, replace=False))
        n_per_config.append(len(trig))
        if len(trig) == 0:
            continue
        e = cfg.electrodes
        acc = np.zeros((len(e), W))
        for unit, fp in footprints.items():
            present, rows = footprint_rows(unit, e)
            if not len(present):
                continue
            L, si = fp.data_uv.shape[1], fp.spike_index
            o_min, o_max = -pre + si - (L - 1), W - 1 - pre + si
            s = trains[unit]
            s = s[(s >= t0 + o_min / fs - 1e-3) & (s < t1 + o_max / fs + 1e-3)]
            counts = _offset_counts(trig, s, fs, o_min, o_max)
            if not counts.any():
                continue
            conv = fftconvolve(fp.data_uv[rows].astype(float), counts[None, :], axes=1)
            acc[present] += conv[:, L - 1 : L - 1 + W]
        acc = acc / len(trig) + rng.normal(0, noise_uv / np.sqrt(len(trig)), acc.shape)
        if combine == "last":
            total[e] = acc
            weight[e] = 1.0
        elif combine == "mean":
            total[e] += acc * len(trig)
            weight[e] += len(trig)
        else:
            raise ValueError("combine must be 'last' or 'mean'")
    recorded = weight > 0
    total[recorded] /= weight[recorded, None]
    grid = total.reshape(N_ROWS, N_COLS, W).transpose(1, 0, 2)  # (col, row, t) = (x, y, t)
    return grid.astype(np.float32), n_per_config
