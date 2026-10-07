"""Units of a separate network recording of the same well as extra regressors for the deconvolution.

An axon scan is sorted on its common electrodes only, so neurons whose somata lie elsewhere are not sorted, yet
they fire during network bursts and appear in the templates of sorted units. A network recording (e.g. a few
hundred electrodes spread over the array, recorded on another day) provides their somatic templates. In every
configuration of the axon scan, the network units whose somata were recorded are detected by template matching
(spikeinterface tdc-peeler: threshold crossings explained by a known template within amplitude limits; spikes of
other neurons remain unassigned) and used as additional regressors. Network units that are the same neurons as
sorted units (soma close to a sorted unit, or spikes coinciding with a sorted unit's) are left out.
"""

import ast
import os
from dataclasses import dataclass

import numpy as np

PITCH = 17.5


@dataclass
class NetworkTemplates:
    templates: np.ndarray  # (units, samples, channels), uV
    electrodes: np.ndarray  # MaxWell electrode id of every channel
    unit_ids: np.ndarray
    fs: float
    nbefore: int

    @property
    def soma_electrodes(self):
        """Electrode with the largest peak-to-peak amplitude of every unit."""
        return self.electrodes[np.argmax(np.ptp(self.templates, axis=1), axis=1)]


def electrode_ids(positions_um):
    """MaxWell electrode ids from channel positions (um, x/y on the 17.5 um grid)."""
    p = np.rint(np.asarray(positions_um, float) / PITCH).astype(int)
    return p[:, 1] * 220 + p[:, 0]


def load_network_templates(sorter_output, subfolder="qc_output", max_trough_offset=3):
    """Templates of the units that passed the QC of a network sorting (sorter_output/qc_output: raw_templates.npy
    (units, samples, channels) in uV, phy_ids.npy, channel_positions.npy; sample rate from params.py).
    Units whose trough (at their largest channel) lies more than max_trough_offset samples from the common
    alignment are left out (they cannot be matched with the common cutout)."""
    folder = os.path.join(sorter_output, subfolder)
    templates = np.load(os.path.join(folder, "raw_templates.npy"))
    params = {}
    for name in (os.path.join(folder, "params.py"), os.path.join(sorter_output, "params.py")):
        if os.path.exists(name):
            for line in open(name):
                if "=" in line:
                    key, value = line.split("=", 1)
                    try:
                        params[key.strip()] = ast.literal_eval(value.strip())
                    except (ValueError, SyntaxError):
                        pass
            break
    peak = np.argmax(np.ptp(templates, axis=1), axis=1)
    troughs = np.argmin(templates[np.arange(len(templates)), :, peak], axis=1)
    nbefore = int(np.median(troughs))
    aligned = np.abs(troughs - nbefore) <= max_trough_offset
    return NetworkTemplates(templates=templates[aligned],
                            electrodes=electrode_ids(np.load(os.path.join(folder, "channel_positions.npy"))),
                            unit_ids=np.load(os.path.join(folder, "phy_ids.npy")).ravel()[aligned],
                            fs=float(params.get("sample_rate", 10000.0)), nbefore=nbefore)


def network_templates_from_sorting(sorter_output, ms_before=2.0, ms_after=3.0, max_spikes_per_unit=500,
                                   labels=("good",), filter_band=150, cache=True):
    """Templates of a Kilosort sorting of a network recording (e.g. Kilosort4 run by scripts/sort_axon_scan_ks4.py),
    averaged on the recording it was sorted on (spikeinterface_recording.json next to sorter_output, high-pass
    filtered as for template extraction). labels: Kilosort labels to keep (None: all units). Cached in
    sorter_output/network_templates.npz."""
    cache_file = os.path.join(sorter_output, "network_templates.npz")
    if cache and os.path.exists(cache_file):
        c = np.load(cache_file)
        return NetworkTemplates(templates=c["templates"], electrodes=c["electrodes"], unit_ids=c["unit_ids"],
                                fs=float(c["fs"]), nbefore=int(c["nbefore"]))
    import pandas as pd
    import spikeinterface.full as si

    from axon_tracking.template_extraction import filter_recording

    sorting = si.read_kilosort(sorter_output)
    if labels is not None:
        label_file = os.path.join(sorter_output, "cluster_KSLabel.tsv")
        lab = pd.read_csv(label_file, sep="\t")
        keep = set(lab.cluster_id[lab.KSLabel.isin(labels)])
        sorting = sorting.select_units([u for u in sorting.get_unit_ids() if u in keep])
    if "main_channel_id" in sorting.get_property_keys():
        sorting.delete_property("main_channel_id")
    recording = si.load(os.path.join(os.path.dirname(os.path.normpath(sorter_output)), "spikeinterface_recording.json"))
    recording = filter_recording(recording, filter_band)
    sorting = si.remove_excess_spikes(sorting, recording)
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=False)
    analyzer.compute(["random_spikes", "templates"],
                     extension_params={"random_spikes": {"max_spikes_per_unit": max_spikes_per_unit},
                                       "templates": {"ms_before": ms_before, "ms_after": ms_after}})
    templates = analyzer.get_extension("templates").get_data()
    fs = recording.get_sampling_frequency()
    net = NetworkTemplates(templates=np.asarray(templates, np.float32),
                           electrodes=electrode_ids(recording.get_channel_locations()),
                           unit_ids=np.asarray(sorting.get_unit_ids()), fs=fs, nbefore=int(ms_before * fs / 1000))
    if cache:
        np.savez(cache_file, templates=net.templates, electrodes=net.electrodes, unit_ids=net.unit_ids, fs=net.fs,
                 nbefore=net.nbefore)
    return net


def sorter_unit_positions(sorter_output):
    """Soma position (um) of every unit of a Kilosort output: position of the channel with the largest
    peak-to-peak amplitude of its template (templates.npy, channel_positions.npy); None if not available."""
    try:
        templates = np.load(os.path.join(sorter_output, "templates.npy"), mmap_mode="r")
        positions = np.load(os.path.join(sorter_output, "channel_positions.npy"))
    except FileNotFoundError:
        return None
    return positions[np.argmax(np.ptp(np.asarray(templates), axis=1), axis=1)]


def exclude_sorted_neurons(net: NetworkTemplates, sorted_positions_um, radius_um=35.0):
    """Network units whose soma is farther than radius_um from every sorted unit's soma."""
    if sorted_positions_um is None or not len(sorted_positions_um):
        return net
    from scipy.spatial import cKDTree

    soma_um = np.column_stack([net.soma_electrodes % 220, net.soma_electrodes // 220]) * PITCH
    keep = cKDTree(np.asarray(sorted_positions_um)[:, :2]).query(soma_um)[0] > radius_um
    return NetworkTemplates(net.templates[keep], net.electrodes, net.unit_ids[keep], net.fs, net.nbefore)


def detect_in_recording(recording, recording_electrodes, net: NetworkTemplates, method_kwargs=None, job_kwargs=None):
    """Spike trains (sample indices) of the network units whose soma electrode was recorded, by template matching
    on the electrodes shared by the network recording and `recording` (filtered, in uV)."""
    from spikeinterface.core import Templates
    from spikeinterface.sortingcomponents.matching import find_spikes_from_templates

    if not np.isclose(recording.get_sampling_frequency(), net.fs):
        raise ValueError("the network templates must have the sampling rate of the recording")
    recording_electrodes = np.asarray(recording_electrodes)
    shared = np.intersect1d(recording_electrodes, net.electrodes)
    units = np.flatnonzero(np.isin(net.soma_electrodes, shared))
    if not len(units) or not len(shared):
        return {}
    channel_ids = recording.get_channel_ids()[[int(np.flatnonzero(recording_electrodes == e)[0]) for e in shared]]
    sub = recording.select_channels(channel_ids)
    net_index = [int(np.flatnonzero(net.electrodes == e)[0]) for e in shared]
    templates = Templates(templates_array=net.templates[units][:, :, net_index].astype(np.float32),
                          sampling_frequency=net.fs, nbefore=net.nbefore, is_in_uV=True,
                          channel_ids=channel_ids, unit_ids=net.unit_ids[units], probe=sub.get_probe())
    kwargs = dict(method_kwargs or {})
    if "noise_levels" not in kwargs:  # of the traces as the matching reads them (uV)
        from spikeinterface.core import get_noise_levels

        kwargs["noise_levels"] = get_noise_levels(sub, return_in_uV=False)
    spikes = find_spikes_from_templates(sub, templates, method="tdc-peeler", method_kwargs=kwargs,
                                        job_kwargs=job_kwargs or dict(n_jobs=1, progress_bar=False))
    return {net.unit_ids[u]: np.sort(spikes["sample_index"][spikes["cluster_index"] == k]).astype(np.int64)
            for k, u in enumerate(units)}


def _share_near(t, s, tolerance):
    """Share of the spikes t within `tolerance` samples of a spike in s (both sorted)."""
    if not len(s) or not len(t):
        return 0.0
    i = np.searchsorted(s, t)
    left = np.abs(t - s[np.maximum(i - 1, 0)])
    right = np.abs(s[np.minimum(i, len(s) - 1)] - t)
    return float(np.mean(np.minimum(left, right) <= tolerance))


def drop_duplicates_of(trains, sorted_trains, tolerance=5, max_shared=0.3, min_spikes=3):
    """Leave out network units with fewer than min_spikes detections or with more than max_shared of their spikes
    within `tolerance` samples of the spikes of one sorted unit (the same neuron)."""
    sorted_trains = [np.sort(np.asarray(s)) for s in sorted_trains]
    return {u: t for u, t in trains.items()
            if len(t) >= min_spikes and all(_share_near(t, s, tolerance) <= max_shared for s in sorted_trains)}
