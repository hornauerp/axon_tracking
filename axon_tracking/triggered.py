"""Recordings that store only snippets around trigger events (MaxWell axon tracking assay).

The raw file holds the stored samples back to back, with the frame number of every sample; readers (neo /
spikeinterface) present them as one continuous recording, and spike sorting and templates are computed on that
concatenation. For least squares and for null templates, the real time axis matters: spikes in neighbouring
snippets did not overlap in time, a footprint is cut where its snippet ends, and the high-pass filter must not ring
across a gap. TriggeredTraces maps stored samples to real time (frame numbers), reads zeros where nothing was
stored and filters each snippet on its own.

Snippets triggered by other units while the target unit was silent hold the background of the target's template
without its signal: silent_decoys draws windows from them for null templates (deconvolution.null_footprint).
"""

import warnings

import numpy as np


def maxwell_frame_numbers(recording):
    """Frame number of every stored sample of a MaxWell recording (walking down wrapped spikeinterface recordings to
    the MaxwellRecordingExtractor), or None if the file has none. The file must be readable (not archived)."""
    import h5py

    rec = recording
    while rec is not None:
        kwargs = getattr(rec, "_kwargs", {})
        if "file_path" in kwargs and "stream_id" in kwargs:
            break
        rec = kwargs.get("recording", kwargs.get("parent_recording"))
    if rec is None:
        return None
    kwargs = rec._kwargs
    rec_name = kwargs.get("rec_name") or "rec0000"
    with h5py.File(kwargs["file_path"], "r") as f:
        if "wells" not in f:  # MaxOne legacy format: continuous
            return None
        groups = f["wells"][kwargs["stream_id"]][rec_name]["groups"]
        group = groups["routed"] if "routed" in groups else groups[list(groups.keys())[0]]
        for name in ("frame_nos", "frame_numbers", "frameno"):
            if name in group:
                frames = np.asarray(group[name][:], np.int64).ravel()
                if len(frames) != recording.get_num_samples(segment_index=0):
                    warnings.warn(f"{name}: {len(frames)} frame numbers for {recording.get_num_samples()} samples")
                    return None
                return frames
    return None


class TriggeredTraces:
    """Real-time access to concatenated trigger snippets.

    get_stored(start, end) -> (end - start, channels) reads stored samples [start, end); frame_nos are their frame
    numbers (increasing, jumps between snippets). Times are on the real axis t = frame - frame_nos[0], of length
    n_samples; `recorded` marks the stored samples. sos: optional filter (scipy second-order sections), applied
    forward-backward to each snippet separately, after subtracting the snippet's median per channel (raw offsets
    are far larger than what a high-pass removes within a few ms)."""

    def __init__(self, get_stored, frame_nos, sos=None):
        self.get_stored = get_stored
        frames = np.asarray(frame_nos, np.int64)
        if np.any(np.diff(frames) <= 0):
            raise ValueError("frame numbers must increase")
        self.t = frames - frames[0]
        self.n_samples = int(self.t[-1]) + 1
        self.recorded = np.zeros(self.n_samples, bool)
        self.recorded[self.t] = True
        breaks = np.flatnonzero(np.diff(self.t) != 1) + 1
        self.block_start = np.r_[0, breaks]  # stored index of the first sample of every snippet
        self.block_end = np.r_[breaks, len(self.t)]
        self.sos = sos

    @property
    def n_snippets(self):
        return len(self.block_start)

    def to_real(self, stored_index):
        """Real time of stored samples (e.g. spike times from a sorting of the concatenated snippets)."""
        return self.t[np.asarray(stored_index, np.int64)]

    def get_traces(self, start, end):
        """Real-time traces [start, end): stored samples (filtered per snippet), zeros elsewhere."""
        s0, s1 = np.searchsorted(self.t, start), np.searchsorted(self.t, end)
        out = None
        if s0 < s1:
            if self.sos is None:
                data = np.asarray(self.get_stored(s0, s1), np.float64)
                out = np.zeros((end - start, data.shape[1]))
                out[self.t[s0:s1] - start] = data
            else:
                from scipy.signal import sosfiltfilt

                b0 = np.searchsorted(self.block_end, s0, side="right")
                b1 = np.searchsorted(self.block_start, s1)
                for b in range(b0, b1):
                    lo, hi = self.block_start[b], self.block_end[b]
                    data = np.asarray(self.get_stored(lo, hi), np.float64)
                    data = data - np.median(data, axis=0, keepdims=True)
                    if hi - lo > 1:
                        pad = min(3 * (2 * len(self.sos) + 1), hi - lo - 1)
                        data = sosfiltfilt(self.sos, data, axis=0, padlen=pad)
                    keep = (self.t[lo:hi] >= start) & (self.t[lo:hi] < end)
                    if out is None:
                        out = np.zeros((end - start, data.shape[1]))
                    out[self.t[lo:hi][keep] - start] = data[keep]
        if out is None:
            out = np.zeros((end - start, np.asarray(self.get_stored(0, 1)).shape[1]))
        return out


def activity(times, population, half_window):
    """Number of spikes of the population (sorted spike times of all units) within +- half_window of each time: a
    proxy of the network state (e.g. inside or outside a burst)."""
    times = np.asarray(times, np.int64)
    return np.searchsorted(population, times + half_window, side="right") - np.searchsorted(population, times - half_window)


def silent_decoys(train, recorded, pre, post, rng, n=None, min_spacing=None, population=None, half_window=50,
                  max_candidates=200000):
    """Decoy spike times for a null template of a unit: n (default: as many as the unit has spikes) recorded
    samples whose window [t - pre, t + post) is recorded at its trigger sample, holds none of the unit's spikes
    (its window plus one window on either side stays clear) and, if possible, does not overlap other decoys
    (min_spacing, default pre + post).

    population: sorted spike times of all sorted units; then the decoys are drawn to match the network activity
    around the unit's own spikes (activity: spikes within +- half_window samples), so that the null contains the
    burst-related background of unsorted neurons, as the unit's template does (uniform decoys underrepresent bursts)."""
    W = pre + post
    train = np.sort(np.asarray(train, np.int64))
    n = len(train) if n is None else n
    candidates = np.flatnonzero(recorded)
    candidates = candidates[(candidates >= pre) & (candidates <= len(recorded) - post)]
    if len(train):
        k = np.searchsorted(train, candidates)
        dist = np.minimum(np.abs(candidates - train[np.clip(k - 1, 0, len(train) - 1)]),
                          np.abs(train[np.clip(k, 0, len(train) - 1)] - candidates))
        candidates = candidates[dist >= W]
    if not len(candidates):
        return np.array([], np.int64)
    spacing = W if min_spacing is None else min_spacing
    if population is not None and len(train):
        if len(candidates) > max_candidates:
            candidates = np.sort(rng.choice(candidates, max_candidates, replace=False))
        population = np.sort(np.asarray(population, np.int64))
        target = activity(train, population, half_window) - 1  # without the unit's own spike
        cand_act = activity(candidates, population, half_window)
        bins = np.unique(np.quantile(np.r_[target, cand_act], np.linspace(0, 1, 21)))
        t_hist = np.histogram(target, bins)[0] / max(len(target), 1)
        c_bin = np.clip(np.searchsorted(bins, cand_act, side="right") - 1, 0, len(bins) - 2)
        c_hist = np.bincount(c_bin, minlength=len(bins) - 1) / len(candidates)
        weight = t_hist[c_bin] / np.maximum(c_hist[c_bin], 1e-12)
        if weight.sum() > 0:
            order = rng.choice(len(candidates), int(np.count_nonzero(weight)), replace=False, p=weight / weight.sum())
            candidates = candidates[order]
        else:
            candidates = rng.permutation(candidates)
    else:
        candidates = rng.permutation(candidates)
    picked = []
    for c in candidates:
        if len(picked) >= n:
            break
        if not picked or np.min(np.abs(np.asarray(picked) - c)) >= spacing:
            picked.append(c)
    if len(picked) < n:  # not enough room for non-overlapping decoys: fill up with overlapping ones
        rest = np.setdiff1d(candidates, picked)
        picked += list(rng.choice(rest, min(n - len(picked), len(rest)), replace=False))
    return np.sort(np.asarray(picked, np.int64))


def stored_subset(recording, frames):
    """Lazy recording of the samples `frames` (sorted indices) of a one-segment recording: emulates a triggered
    acquisition on continuous data (e.g. snippets around trigger events, see snippet_frames)."""
    from spikeinterface.core import BaseRecording, BaseRecordingSegment

    class _Segment(BaseRecordingSegment):
        def __init__(self, parent_segment, frames, sampling_frequency):
            super().__init__(sampling_frequency=sampling_frequency)
            self.parent_segment, self.frames = parent_segment, frames

        def get_num_samples(self):
            return len(self.frames)

        def get_traces(self, start_frame, end_frame, channel_indices):
            start_frame = 0 if start_frame is None else start_frame
            end_frame = len(self.frames) if end_frame is None else end_frame
            fr = self.frames[start_frame:end_frame]
            if not len(fr):
                return self.parent_segment.get_traces(0, 0, channel_indices)
            traces = self.parent_segment.get_traces(int(fr[0]), int(fr[-1]) + 1, channel_indices)
            return traces[fr - fr[0]]

    class StoredSubsetRecording(BaseRecording):
        def __init__(self, recording, frames):
            frames = np.asarray(frames, np.int64)
            BaseRecording.__init__(self, recording.get_sampling_frequency(), recording.get_channel_ids(),
                                   recording.get_dtype())
            recording.copy_metadata(self)
            self.add_recording_segment(_Segment(recording._recording_segments[0], frames,
                                                recording.get_sampling_frequency()))
            self._kwargs = dict(recording=recording, frames=frames)

    return StoredSubsetRecording(recording, frames)


def snippet_frames(trigger_samples, pre, post, n_samples):
    """Samples stored by a triggered acquisition: [t - pre, t + post) around every trigger (overlaps merged)."""
    t = np.asarray(trigger_samples, np.int64)
    t = t[(t >= pre) & (t <= n_samples - post)]
    starts, ends = t - pre, t + post
    diff = np.zeros(n_samples + 1, int)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    keep = np.cumsum(diff[:-1]) > 0
    return np.flatnonzero(keep)
