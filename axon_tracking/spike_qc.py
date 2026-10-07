"""Data-driven removal of false-positive spikes from a sorting.

Lower detection thresholds give more complete spike trains (more spikes of the same neurons, less template noise),
but also more false detections, which dilute the templates (and act like decoys in the deconvolution). Every spike
is scored by the amplitude of its projection onto its unit's template on the unit's main channels; the same score
at decoy windows (random times at which the unit did not fire) gives the null distribution. Spikes whose score is
consistent with the null are flagged.
"""

import numpy as np


def unit_channels(templates, locations, radius_um=25.0):
    """Main channels of every unit: channels within radius_um of the channel with its largest peak-to-peak
    amplitude. templates: (units, samples, channels)."""
    peak = np.argmax(np.ptp(templates, axis=1), axis=1)
    return [np.flatnonzero(np.linalg.norm(locations - locations[p], axis=1) <= radius_um) for p in peak]


def _windows(get_traces, n_samples, times, pre, post, chunk):
    """Yield (index into times, traces[t - pre : t + post]) for all times (sorted), reading in chunks."""
    order = np.argsort(times)
    times = np.asarray(times)[order]
    for c0 in range(0, n_samples, chunk):
        sel = np.flatnonzero((times >= c0) & (times < c0 + chunk))
        if not len(sel):
            continue
        lo, hi = max(times[sel].min() - pre, 0), min(times[sel].max() + post, n_samples)
        traces = np.asarray(get_traces(lo, hi), np.float64)
        for k in sel:
            a = times[k] - pre - lo
            if a >= 0 and a + pre + post <= len(traces):
                yield order[k], traces[a : a + pre + post]


def projection_scores(get_traces, n_samples, trains, templates, channels, pre, post, chunk=100000):
    """Projection amplitude of every spike onto its unit's template on its main channels (1: a typical spike,
    0: no trace of the template). trains: list of spike times (samples); templates: (units, pre + post, channels)."""
    times = np.concatenate([np.asarray(t, np.int64) for t in trains]) if trains else np.zeros(0, np.int64)
    units = np.concatenate([np.full(len(t), k) for k, t in enumerate(trains)]) if trains else np.zeros(0, int)
    T = [templates[k][:, channels[k]] for k in range(len(trains))]
    norm = [max(float((t * t).sum()), 1e-12) for t in T]
    scores = np.full(len(times), np.nan)
    for i, w in _windows(get_traces, n_samples, times, pre, post, chunk):
        k = units[i]
        scores[i] = float((w[:, channels[k]] * T[k]).sum()) / norm[k]
    offsets = np.r_[0, np.cumsum([len(t) for t in trains])]
    return [scores[offsets[k] : offsets[k + 1]] for k in range(len(trains))]


def decoy_times(train, n_samples, n, pre, post, rng, min_distance=None):
    """n random times at least min_distance (default pre + post) from every spike of the unit."""
    train = np.sort(np.asarray(train, np.int64))
    gap = pre + post if min_distance is None else min_distance
    candidates = rng.integers(pre, n_samples - post, size=4 * n + 100)
    if len(train):
        k = np.searchsorted(train, candidates)
        near = np.minimum(np.abs(candidates - train[np.clip(k - 1, 0, len(train) - 1)]),
                          np.abs(train[np.clip(k, 0, len(train) - 1)] - candidates))
        candidates = candidates[near >= gap]
    return np.sort(candidates[:n])


def false_positive_mask(scores, null_scores, fpr=0.01):
    """True for spikes whose projection score does not exceed the (1 - fpr) quantile of the null scores: a spike
    of this size or smaller is produced by noise at rate fpr."""
    null_scores = np.asarray(null_scores, float)
    null_scores = null_scores[np.isfinite(null_scores)]
    if not len(null_scores):
        return np.zeros(len(scores), bool)
    threshold = np.quantile(null_scores, 1 - fpr)
    return ~(np.asarray(scores) > threshold)
