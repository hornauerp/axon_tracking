"""Continuous simulated recordings (spikeinterface) for testing the parts of the pipeline that need raw traces.

SimulatedRecording generates the traces of one configuration on demand: white noise (deterministic per block, so
re-reading a chunk gives the same samples) plus the footprints of all neurons at their spike times. Together with
an imperfect sorting (observed_sorting: only some neurons sorted, spikes missed during overlaps) and templates of a
sparse network recording (network_templates), the deconvolution with network units and spike recovery can be run
on ground truth exactly as on real data.
"""

import numpy as np

from axon_tracking.simulation.forward import N_COLS, PITCH, Footprint

BLOCK = 10000  # samples per noise block


def _probe(electrodes):
    from probeinterface import Probe

    electrodes = np.asarray(electrodes)
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.column_stack([electrodes % N_COLS, electrodes // N_COLS]) * PITCH,
                       shapes="square", shape_params={"width": 5.0})
    probe.set_device_channel_indices(np.arange(len(electrodes)))
    probe.set_contact_ids([str(e) for e in electrodes])
    probe.annotate_contacts(electrode=electrodes)
    return probe


def spike_amplitudes(samples, rng, fs=10000.0, burst_attenuation=0.0, burst_isi_ms=20.0, min_factor=0.5,
                     amplitude_jitter=0.0):
    """Amplitude factor of every spike (as MEArec's bursting modulation): the k-th spike of a burst (preceding
    interval < burst_isi_ms) is scaled by (1 - burst_attenuation)^k (at least min_factor), times a random factor
    1 + N(0, amplitude_jitter)."""
    t = np.asarray(samples)
    k = np.zeros(len(t))
    isi = np.diff(t) < burst_isi_ms * fs / 1000
    for i in np.flatnonzero(isi):
        k[i + 1] = k[i] + 1
    factor = np.maximum((1 - burst_attenuation) ** k, min_factor)
    if amplitude_jitter:
        factor = factor * (1 + amplitude_jitter * rng.standard_normal(len(t)))
    return factor.astype(np.float32)


def make_recording(electrodes, footprints, spike_samples, n_samples, fs=10000.0, noise_uv=6.0, seed=0,
                   burst_attenuation=0.0, amplitude_jitter=0.0, noise_correlation_um=None):
    """Simulated recording of `electrodes` (MaxWell ids) with footprints {unit: Footprint} firing at
    spike_samples {unit: sample indices}; float32 in uV (gain 1).

    burst_attenuation, amplitude_jitter: spike amplitude modulation (spike_amplitudes), which violates the fixed
    footprints assumed by averaging and deconvolution as in real bursts. noise_correlation_um: spatially
    correlated noise with covariance exp(-distance / noise_correlation_um) between channels (white if None)."""
    from spikeinterface.core import BaseRecording, BaseRecordingSegment

    electrodes = np.asarray(electrodes)
    chol = None
    if noise_correlation_um:
        xy = np.column_stack([electrodes % N_COLS, electrodes // N_COLS]) * PITCH
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        chol = np.linalg.cholesky(np.exp(-d / noise_correlation_um) + 1e-6 * np.eye(len(electrodes))).astype(np.float32)
    amp_rng = np.random.default_rng([seed, 1])

    class _Segment(BaseRecordingSegment):
        def __init__(self):
            BaseRecordingSegment.__init__(self, sampling_frequency=fs, t_start=None)
            lookup = np.full(N_COLS * 120, -1)
            lookup[electrodes] = np.arange(len(electrodes))
            self.units = []
            for u, fp in footprints.items():
                channels = lookup[fp.electrodes]
                rows = np.flatnonzero(channels >= 0)
                t = np.sort(np.asarray(spike_samples.get(u, []), np.int64))
                if len(rows) and len(t):
                    a = spike_amplitudes(t, amp_rng, fs, burst_attenuation, amplitude_jitter=amplitude_jitter)
                    self.units.append((t, a, channels[rows], fp.data_uv[rows].T.astype(np.float32), fp.spike_index))

        def get_num_samples(self):
            return n_samples

        def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
            start = 0 if start_frame is None else int(start_frame)
            end = n_samples if end_frame is None else int(end_frame)
            out = np.empty((end - start, len(electrodes)), np.float32)
            for b in range(start // BLOCK, (end - 1) // BLOCK + 1):
                lo, hi = max(start, b * BLOCK), min(end, (b + 1) * BLOCK)
                noise = np.random.default_rng([seed, b]).standard_normal((BLOCK, len(electrodes)), np.float32)
                if chol is not None:
                    noise = noise @ chol.T
                out[lo - start : hi - start] = noise_uv * noise[lo - b * BLOCK : hi - b * BLOCK]
            for t, amp, channels, waveform, si in self.units:  # waveform (L, n_channels of the unit)
                L = len(waveform)
                i0, i1 = np.searchsorted(t, start - L + si), np.searchsorted(t, end + si)
                for s, f in zip(t[i0:i1], amp[i0:i1]):
                    a = s - si - start
                    w0, w1 = max(0, -a), min(L, end - start - a)
                    if w1 > w0:
                        out[a + w0 : a + w1, channels] += f * waveform[w0:w1]
            return out if channel_indices is None else out[:, channel_indices]

    class SimulatedRecording(BaseRecording):
        def __init__(self):
            BaseRecording.__init__(self, sampling_frequency=fs, channel_ids=[str(e) for e in electrodes],
                                   dtype="float32")
            self.add_recording_segment(_Segment())
            self.set_channel_gains(1.0)
            self.set_channel_offsets(0.0)
            self.set_probe(_probe(electrodes), in_place=True)

    return SimulatedRecording()


def axon_scan_recordings(culture, trains, configs, fs=10000.0, noise_uv=6.0, seed=0, **kwargs):
    """One SimulatedRecording per configuration (spike trains in s on the scan's clock; kwargs: see
    make_recording); returns (recordings, spike samples per configuration {unit: samples relative to the
    configuration start})."""
    footprints = {i: n.footprint for i, n in enumerate(culture.neurons)}
    recordings, samples = [], []
    for k, cfg in enumerate(configs):
        n = int(round(cfg.duration_s * fs))
        s = {u: np.rint((t[(t >= cfg.start_s) & (t < cfg.start_s + cfg.duration_s)] - cfg.start_s) * fs).astype(np.int64)
             for u, t in trains.items()}
        recordings.append(make_recording(cfg.electrodes, footprints, s, n, fs, noise_uv, seed=hash((seed, k)) % 2**31,
                                         **kwargs))
        samples.append(s)
    return recordings, samples


def observed_sorting(samples, sorted_units, culture, rng, overlap_ms=1.0, p_miss_overlap=0.5, p_miss=0.02,
                     overlap_radius_um=150.0, fs=10000.0):
    """Spike trains as a sorter would report them: only `sorted_units`, missing a share of the spikes that overlap
    (within overlap_ms) with a spike of another neuron whose soma lies within overlap_radius_um, and p_miss of the
    rest. samples: {unit: spike samples} of one configuration."""
    soma = {i: n.morphology.mid[n.morphology.kind == "soma", :2].mean(axis=0) for i, n in enumerate(culture.neurons)}
    window = int(round(overlap_ms * fs / 1000))
    out = {}
    for u in sorted_units:
        t = samples[u]
        near = [v for v in samples if v != u and np.linalg.norm(soma[v] - soma[u]) <= overlap_radius_um]
        others = np.sort(np.concatenate([samples[v] for v in near])) if near else np.zeros(0, np.int64)
        overlapping = np.zeros(len(t), bool)
        if len(others):
            i = np.searchsorted(others, t)
            d = np.minimum(np.abs(t - others[np.maximum(i - 1, 0)]), np.abs(others[np.minimum(i, len(others) - 1)] - t))
            overlapping = d <= window
        miss = rng.random(len(t)) < np.where(overlapping, p_miss_overlap, p_miss)
        out[u] = t[~miss]
    return out


def network_templates(culture, electrodes, fs=10000.0, nbefore=10, n_samples=30, amplitude_change=0.0, rng=None):
    """Templates of all neurons on the electrodes of a (sparse) network recording, as
    network_units.NetworkTemplates; amplitude_change > 0 scales each neuron by a random factor in
    [1 - amplitude_change, 1 + amplitude_change] (a recording on another day)."""
    from axon_tracking.network_units import NetworkTemplates

    rng = rng or np.random.default_rng(0)
    electrodes = np.asarray(electrodes)
    lookup = {e: k for k, e in enumerate(electrodes)}
    T = np.zeros((len(culture.neurons), n_samples, len(electrodes)), np.float32)
    for i, n in enumerate(culture.neurons):
        fp = n.footprint
        lags = np.arange(n_samples) - nbefore + fp.spike_index
        ok = (lags >= 0) & (lags < fp.data_uv.shape[1])
        scale = 1 + rng.uniform(-amplitude_change, amplitude_change)
        for row, e in enumerate(fp.electrodes):
            if e in lookup:
                T[i, ok, lookup[e]] = scale * fp.data_uv[row, lags[ok]]
    keep = np.abs(T).max(axis=(1, 2)) > 0
    return NetworkTemplates(T[keep], electrodes, np.flatnonzero(keep), fs, nbefore)


def true_template(footprint: Footprint, pre, post, highpass_hz=150.0, fs=10000.0):
    """Footprint (electrodes, pre + post samples around the spike) after the pipeline's highpass filter
    (5th order Butterworth, forward-backward), for comparing estimated templates with the truth."""
    import scipy.signal

    sos = scipy.signal.butter(5, highpass_hz, btype="highpass", fs=fs, output="sos")
    pad = 200
    x = np.pad(footprint.data_uv.astype(np.float64), ((0, 0), (pad, pad)))
    y = scipy.signal.sosfiltfilt(sos, x, axis=1)
    lags = np.arange(-pre, post) + footprint.spike_index + pad
    return y[:, lags]
