"""Footprints of all sorted units by least squares instead of spike-triggered averaging.

The recording is modelled as the superposition of the footprints of all sorted units at their spike times plus
noise: y_e(t) = sum_j sum_b X[t, (j, b)] f_j,e(b) + n_e(t), with X[t, (j, b)] = 1 if unit j spiked at t - b + pre.
The spike-triggered sums are S = X^T y (S_i(a) = N_i * STA_i(a)) and the Gram matrix G = X^T X is block Toeplitz
with the cross-correlogram counts of the units: G[(i, a), (j, b)] = #{t_j - t_i = a - b}. Solving G f = S gives
the least-squares footprints, which no longer contain the (realized) activity of the other sorted units that
fired around the target's spikes (e.g. during network bursts), nor the unit's own other spikes within the window.
All spikes are used, so no cell type is disadvantaged. Neurons that were not sorted still contaminate.

The noise of the estimate is sigma^2 G^-1: for units that rarely overlap it equals that of the STA
(sigma^2 / N_i); it grows where the spike trains are strongly locked to each other.

Triggered acquisition (axon tracking assay: only snippets around trigger events are stored): spike times and
samples are placed on the real time axis (frame numbers) and only recorded samples enter the fit. Then
G[(i, a), (j, b)] counts the spike pairs with t_i + a = t_j + b whose shared sample was recorded (gram_matrix with
`recorded`); it is no longer Toeplitz, and footprints of spikes near a gap are fitted from their recorded part.
Treating the concatenated snippets as continuous instead glues unrelated times together.
"""

import numpy as np
import scipy.linalg


def ccg_counts(spike_trains, max_lag):
    """Cross-correlogram counts of all unit pairs.

    spike_trains: list of sorted integer spike times (samples). Returns (n, n, 2 * max_lag + 1) with
    counts[i, j, d + max_lag] = #{(t_i, t_j): t_j - t_i = d}.
    """
    n = len(spike_trains)
    counts = np.zeros((n, n, 2 * max_lag + 1))
    for i, ti in enumerate(spike_trains):
        ti = np.asarray(ti, np.int64)
        if not len(ti):
            continue
        for j, tj in enumerate(spike_trains):
            tj = np.asarray(tj, np.int64)
            if not len(tj):
                continue
            lo = np.searchsorted(tj, ti - max_lag)
            hi = np.searchsorted(tj, ti + max_lag, side="right")
            m = hi - lo
            if not m.sum():
                continue
            idx = np.repeat(lo, m) + np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)  # all of lo[k]..hi[k]-1
            d = tj[idx] - np.repeat(ti, m) + max_lag
            counts[i, j] += np.bincount(d, minlength=2 * max_lag + 1)
    return counts


def window_masks(train, window, pre, recorded):
    """(spikes, window) bool: which samples of each spike's window [t - pre, t - pre + window) were recorded
    (samples outside the recorded axis count as not recorded)."""
    idx = np.asarray(train, np.int64)[:, None] - pre + np.arange(window)[None, :]
    inside = (idx >= 0) & (idx < len(recorded))
    mask = np.zeros(idx.shape, bool)
    mask[inside] = recorded[idx[inside]]
    return mask


def _masked_block(ti, tj, window, mask_i):
    """(window, window) block of G for trains ti, tj (sorted) when only the samples in mask_i (ti's windows) count:
    block[a, b] = #{(k, l): ti[k] + a = tj[l] + b and mask_i[k, a]}."""
    W = window
    lo = np.searchsorted(tj, ti - (W - 1))
    hi = np.searchsorted(tj, ti + (W - 1), side="right")
    m = hi - lo
    block = np.zeros((W, W))
    if not m.sum():
        return block
    k = np.repeat(np.arange(len(ti)), m)
    l = np.repeat(lo, m) + np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
    d = tj[l] - ti[k]  # b = a - d
    a = np.arange(W)
    counts = np.zeros((2 * W - 1) * W)
    for c in range(0, len(k), 200000):
        dd = d[c : c + 200000]
        valid = mask_i[k[c : c + 200000]] & (a[None, :] - dd[:, None] >= 0) & (a[None, :] - dd[:, None] < W)
        flat = (dd[:, None] + W - 1) * W + a[None, :]
        counts += np.bincount(flat[valid], minlength=len(counts))
    counts = counts.reshape(2 * W - 1, W)
    lag = a[:, None] - a[None, :] + W - 1  # a - b
    return counts[lag, np.broadcast_to(a[:, None], (W, W))]


def locked_duplicates(spike_trains, max_lag, share=0.95, tolerance=1, sites=None, max_site_distance=35.0,
                      min_spikes=20):
    """Pairs of trains that are one neuron sorted twice: at least `share` of the smaller train's spikes have a
    partner in the other train at one fixed lag (+- tolerance samples) and either the same holds for the larger
    train (mutually locked: not identifiable wherever the units are) or, if sites ((n, 2) positions of each unit's
    largest trough at its own spike times) are given, both troughs lie within max_site_distance (a partner neuron
    elsewhere that fires only with the unit is kept). The least-squares footprints of such a pair are not (or barely)
    identifiable and their noise blows up. max_lag: largest lag between the copies (samples; ~1 ms: they differ
    only in the alignment of the trough); trains with fewer than min_spikes spikes are not tested (chance
    coincidences). Returns [(drop, keep, lag)]: drop the smaller train, whose spikes are those of `keep` shifted by
    `lag` (t_drop ~ t_keep + lag)."""
    n = len(spike_trains)
    sizes = np.array([len(t) for t in spike_trains])
    counts = ccg_counts(spike_trains, max_lag)  # counts[i, j, d + max_lag] = #{t_j - t_i = d}
    k = 2 * tolerance + 1
    smoothed = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k), "same"), 2, counts)
    pairs, dropped = [], set()
    for i in np.argsort(sizes):  # smaller trains first
        if sizes[i] < min_spikes or i in dropped:
            continue
        for j in range(n):
            if j == i or j in dropped or sizes[j] < sizes[i]:
                continue
            d = int(np.argmax(smoothed[i, j]))
            mutual = smoothed[i, j, d] >= share * sizes[j]
            far = sites is not None and np.linalg.norm(np.asarray(sites[i]) - np.asarray(sites[j])) > max_site_distance
            if far and not mutual:
                continue
            if smoothed[i, j, d] >= share * sizes[i]:
                lo = max(d - tolerance, 0)
                d = lo + int(np.argmax(counts[i, j, lo : d + tolerance + 1]))  # exact lag within the window
                lag = -(d - max_lag)  # counts are t_j - t_i = d, so t_i = t_j - d
                pairs.append((int(i), int(j), lag))
                dropped.add(int(i))
                break
    return pairs


def shift_footprint(F, lag):
    """Footprint (window, ...) of a train shifted by `lag` samples relative to the train it was estimated for
    (t_new = t_old + lag): F_new[a] = F_old[a + lag], zero outside the window."""
    out = np.zeros_like(F)
    W = len(F)
    if lag >= 0:
        out[: W - lag] = F[lag:]
    else:
        out[-lag:] = F[: W + lag]
    return out


def gram_matrix(spike_trains, window, recorded=None, pre=0):
    """G = X^T X for footprints of `window` samples: (n * window, n * window), ordered (unit, lag).

    recorded: optional bool array over the time axis of the spike trains (triggered acquisition); then only
    recorded samples count, with windows [t - pre, t - pre + window)."""
    if recorded is not None:
        recorded = np.asarray(recorded, bool)
        n, W = len(spike_trains), window
        trains = [np.sort(np.asarray(t, np.int64)) for t in spike_trains]
        masks = [window_masks(t, W, pre, recorded) for t in trains]
        G = np.zeros((n * W, n * W))
        for i in range(n):
            for j in range(n):
                if len(trains[i]) and len(trains[j]):
                    G[i * W : (i + 1) * W, j * W : (j + 1) * W] = _masked_block(trains[i], trains[j], W, masks[i])
        return G
    n = len(spike_trains)
    counts = ccg_counts(spike_trains, window - 1)
    a = np.arange(window)
    lag = a[:, None] - a[None, :] + window - 1  # a - b
    G = np.zeros((n * window, n * window))
    for i in range(n):
        for j in range(n):
            if counts[i, j].any():
                G[i * window : (i + 1) * window, j * window : (j + 1) * window] = counts[i, j][lag]
    return G


def spike_triggered_sums_chunked(get_traces, n_samples, spike_trains, pre, post, chunk_samples=40000):
    """spike_triggered_sums reading the traces in chunks: get_traces(start, end) -> (end - start, channels).
    Spikes whose window leaves [0, n_samples) must have been removed (valid_spikes)."""
    W = pre + post
    units = np.concatenate([np.full(len(t), k) for k, t in enumerate(spike_trains)]).astype(int)
    times = np.concatenate([np.asarray(t, np.int64) for t in spike_trains])
    order = np.argsort(times, kind="stable")
    units, times = units[order], times[order]
    S = None
    for c0 in range(0, n_samples, chunk_samples):
        sel = (times >= c0) & (times < c0 + chunk_samples)
        if not sel.any():
            continue
        t, u = times[sel], units[sel]
        lo, hi = t.min() - pre, t.max() + post
        traces = np.asarray(get_traces(lo, hi), dtype=np.float64)
        if S is None:
            S = np.zeros((len(spike_trains), W, traces.shape[1]))
        windows = traces[(t - lo - pre)[:, None] + np.arange(W)[None, :]]  # (spikes, W, channels)
        for k in np.unique(u):
            S[k] += windows[u == k].sum(axis=0)
    if S is None:
        S = np.zeros((len(spike_trains), W, np.asarray(get_traces(0, 1)).shape[1]))
    return S


def solve_footprints(spike_trains, S, window, ridge=None, recorded=None, pre=0):
    """Least-squares footprints from spike-triggered sums S (n, window, channels) and the spike trains (samples).

    Units without spikes get zeros. If G is singular (e.g. the same neuron sorted twice at a fixed lag), a small
    ridge (1e-3 of the mean spike count) is added, unless ridge is given. recorded/pre: triggered acquisition
    (see gram_matrix; S must then be summed over traces that are zero where nothing was recorded).
    Returns (F, info)."""
    import scipy.linalg

    n, W, E = S.shape
    active = [k for k, t in enumerate(spike_trains) if len(t)]
    F = np.zeros_like(S, dtype=float)
    if not active:
        return F, dict(ridge=0.0, n_active=0)
    G = gram_matrix([spike_trains[k] for k in active], W, recorded, pre)
    # unknowns that no recorded sample constrains (triggered acquisition: lags that are never stored) stay zero
    free = np.diag(G) > 0

    def gram():
        G = gram_matrix([spike_trains[k] for k in active], W, recorded, pre)
        return G if free.all() else G[np.ix_(free, free)]

    if not free.all():
        G = G[np.ix_(free, free)]
    rhs = S[active].reshape(len(active) * W, E)[free]
    used = 0.0 if ridge is None else float(ridge)
    for attempt in range(2):
        if used:
            G[np.diag_indices_from(G)] += used
        try:
            c = scipy.linalg.cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
            break
        except np.linalg.LinAlgError:
            if attempt or ridge is not None:
                raise
            G = gram()
            used = 1e-3 * np.mean([len(spike_trains[k]) for k in active])
    solution = np.zeros((len(active) * W, E))
    solution[free] = scipy.linalg.cho_solve(c, rhs, check_finite=False)
    F[active] = solution.reshape(len(active), W, E)
    return F, dict(ridge=used, n_active=len(active), n_unconstrained=int((~free).sum()))


def null_footprint(decoy, S_decoy, spike_trains, F, pre, recorded=None):
    """Footprint of a decoy train (spike times where the target unit was silent) given the footprints F of the
    regressor units: one block Gauss-Seidel step, f = (S_decoy - sum_j G_dj f_j) / diag(G_dd). It is processed like
    the target's own footprint (same spike count, same regressors) and contains its noise and the residual
    background, but no signal of the target: a null template for thresholds. Also returns the plain average of the
    decoy windows (null of the averaged template). recorded: as in gram_matrix (None: continuous)."""
    W = S_decoy.shape[0]
    decoy = np.sort(np.asarray(decoy, np.int64))
    rec = np.ones(decoy.max() + W + 1, bool) if recorded is None else np.asarray(recorded, bool)
    mask = window_masks(decoy, W, pre, rec)
    coverage = np.maximum(mask.sum(axis=0), 1)[:, None]  # recorded windows per lag
    explained = np.zeros_like(S_decoy, dtype=float)
    for t, f in zip(spike_trains, F):
        t = np.sort(np.asarray(t, np.int64))
        if len(t) and np.any(f):
            explained += _masked_block(decoy, t, W, mask) @ f
    return (S_decoy - explained) / coverage, S_decoy / coverage


def spike_triggered_sums(traces, spike_trains, pre, post):
    """S_i(a) = sum over spikes of unit i of traces[t - pre + a] (traces: (samples, electrodes)).
    Spikes whose window leaves the traces are dropped (and must be dropped from spike_trains as well)."""
    W = pre + post
    S = np.zeros((len(spike_trains), W, traces.shape[1]))
    for i, t in enumerate(spike_trains):
        for s in np.asarray(t, np.int64):
            S[i] += traces[s - pre : s + post]
    return S


def valid_spikes(spike_trains, n_samples, pre, post):
    """Spikes whose window [t - pre, t + post) lies within the recording."""
    return [np.asarray(t, np.int64)[(np.asarray(t) >= pre) & (np.asarray(t) <= n_samples - post)] for t in spike_trains]


def deconvolve(S, G, ridge=0.0):
    """Least-squares footprints F (n, window, electrodes) from spike-triggered sums S (n, window, electrodes)
    and the Gram matrix G. Units without spikes get zeros. ridge: Tikhonov regularisation (in units of spike
    counts) for strongly locked units; 0 gives the unbiased least-squares estimate."""
    n, W, E = S.shape
    active = np.flatnonzero(np.diag(G).reshape(n, W)[:, 0] > 0)
    rows = (active[:, None] * W + np.arange(W)[None, :]).ravel()
    Ga = G[np.ix_(rows, rows)] + ridge * np.eye(len(rows))
    F = np.zeros_like(S, dtype=float)
    F[active] = scipy.linalg.solve(Ga, S[active].reshape(len(rows), E), assume_a="pos").reshape(len(active), W, E)
    return F


def deconvolve_noise_sd(G, window, noise_sd, ridge=0.0):
    """Noise sd of every (unit, lag) of the least-squares footprints for white noise of sd noise_sd:
    sqrt(diag(sigma^2 G^-1)), (n, window), NaN for units without spikes; compare with noise_sd / sqrt(N_i)."""
    n = len(G) // window
    active = np.flatnonzero(np.diag(G).reshape(n, window)[:, 0] > 0)
    rows = (active[:, None] * window + np.arange(window)[None, :]).ravel()
    sd = np.full((n, window), np.nan)
    Ginv = np.linalg.inv(G[np.ix_(rows, rows)] + ridge * np.eye(len(rows)))
    sd[active] = noise_sd * np.sqrt(np.clip(np.diag(Ginv), 0, None)).reshape(len(active), window)
    return sd
