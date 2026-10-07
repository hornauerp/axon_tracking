"""Evidence for reconstructed paths from the template itself (instead of the skeleton's geometry).

For every path, the electrodes it passes are looked up in the (raw, not upsampled) template:
- z: negative peak amplitude / noise sd of the electrode (noise from the median absolute successive difference,
  which is insensitive to the few samples of an action potential)
- latency of the negative peak, regressed on the distance along the path: velocity, R2 and the t statistic of the
  slope (a propagating AP gives latencies that increase steadily along the path)
- ridge: share of path electrodes whose amplitude is a local maximum across the path direction (paths running
  parallel to an axon through its weaker field fail this)

Status: experimental, not part of the pipeline yet (a classifier on these features trained on simulated ground
truth needs validation on real templates first).
"""

import numpy as np
from scipy import stats

PITCH_UM = 17.5


def template_noise_sd(template):
    """Robust noise sd per electrode (x, y) from successive differences of the template (x, y, t)."""
    d = np.diff(template, axis=2)
    return np.median(np.abs(d - np.median(d, axis=2, keepdims=True)), axis=2) / (0.6745 * np.sqrt(2))


def peak_maps(template, fs, t0_index, min_lag_ms=-0.5):
    """Negative peak amplitude (positive values) and its latency (ms after the trigger at t0_index) per electrode,
    searched from min_lag_ms on."""
    start = max(int(t0_index + min_lag_ms * fs / 1000), 0)
    seg = template[:, :, start:]
    idx = np.argmin(seg, axis=2)
    amp = -np.take_along_axis(seg, idx[:, :, None], axis=2)[:, :, 0]
    latency = (idx + start - t0_index) / fs * 1000
    return amp, latency


def waveform_maps(template, fs, t0_index, min_lag_ms=-0.5):
    """Shape of the largest negative peak per electrode: half-width (ms) and the ratio of the preceding positive
    peak to the negative peak. Axonal electrodes show narrow, negative-dominant waveforms, dendritic ones broad,
    positive-first waveforms."""
    start = max(int(t0_index + min_lag_ms * fs / 1000), 0)
    seg = template[:, :, start:]
    i = np.argmin(seg, axis=2)
    neg = -np.take_along_axis(seg, i[:, :, None], axis=2)[:, :, 0]
    below = seg <= -neg[:, :, None] / 2
    # contiguous half-maximum run around the peak
    idx = np.arange(seg.shape[2])[None, None, :]
    left = np.where(~below & (idx < i[:, :, None]), idx, -1).max(axis=2) + 1
    right = np.where(~below & (idx > i[:, :, None]), idx, seg.shape[2]).min(axis=2) - 1
    width = (right - left + 1) / fs * 1000
    # largest positive value before the negative peak (running maximum over time)
    running_max = np.maximum.accumulate(template, axis=2)
    peak = start + i
    pos_before = np.where(peak > 0, np.take_along_axis(running_max, np.maximum(peak - 1, 0)[:, :, None], axis=2)[:, :, 0],
                          0.0)
    ratio = np.divide(np.maximum(pos_before, 0), neg, out=np.full(neg.shape, np.nan), where=neg > 0)
    return width, ratio


def path_electrodes(path_um):
    """Electrodes (col, row) along a path in um (x, y, ...), in order, without repeats."""
    cr = np.rint(np.asarray(path_um)[:, :2] / PITCH_UM).astype(int)
    keep = np.r_[True, np.any(np.diff(cr, axis=0) != 0, axis=1)]
    return cr[keep]


def path_features(path_um, amp, latency, noise_sd, ridge_offset=2, shape=None):
    """Features of one path (see module docstring). amp, latency, noise_sd: (220, 120) maps; shape: optional
    (half-width, positive/negative ratio) maps from waveform_maps."""
    nx, ny = amp.shape
    cr = path_electrodes(path_um)
    cr = cr[(cr[:, 0] >= 0) & (cr[:, 0] < nx) & (cr[:, 1] >= 0) & (cr[:, 1] < ny)]
    out = dict(n_electrodes=len(cr), length_um=np.nan, z_median=np.nan, z_mean=np.nan, velocity_m_s=np.nan,
               latency_r2=np.nan, slope_t=np.nan, ridge=np.nan, halfwidth_ms=np.nan, pos_neg_ratio=np.nan)
    if len(cr) < 2:
        return out
    if shape is not None:
        out.update(halfwidth_ms=float(np.nanmedian(shape[0][cr[:, 0], cr[:, 1]])),
                   pos_neg_ratio=float(np.nanmedian(shape[1][cr[:, 0], cr[:, 1]])))
    a = amp[cr[:, 0], cr[:, 1]]
    sd = np.maximum(noise_sd[cr[:, 0], cr[:, 1]], 1e-9)
    z = a / sd
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cr, axis=0), axis=1))] * PITCH_UM
    lat = latency[cr[:, 0], cr[:, 1]]
    out.update(length_um=dist[-1], z_median=float(np.median(z)), z_mean=float(np.mean(z)))
    if len(cr) >= 4 and np.ptp(dist) > 0:
        fit = stats.linregress(dist, lat)
        slope = abs(fit.slope)  # ms per um; the AP may run in either direction along the path
        out.update(velocity_m_s=float(1e-3 / slope) if slope > 0 else np.inf, latency_r2=float(fit.rvalue ** 2),
                   slope_t=float(abs(fit.slope) / fit.stderr) if fit.stderr > 0 else np.inf)
    # ridge: compare with the electrodes `ridge_offset` pitches away, perpendicular to the local direction
    direction = np.gradient(cr.astype(float), axis=0)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(direction, norm, out=np.zeros_like(direction), where=norm > 0)
    perp = np.column_stack([-direction[:, 1], direction[:, 0]])
    on_ridge = []
    for k in range(len(cr)):
        side = [np.rint(cr[k] + s * ridge_offset * perp[k]).astype(int) for s in (-1, 1)]
        side = [s for s in side if 0 <= s[0] < nx and 0 <= s[1] < ny]
        on_ridge.append(all(a[k] >= amp[s[0], s[1]] for s in side))
    out["ridge"] = float(np.mean(on_ridge))
    return out
