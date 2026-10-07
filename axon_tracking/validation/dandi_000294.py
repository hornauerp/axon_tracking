"""Patched neurons with imaged morphology on a MaxOne HD-MEA (DANDI 000294).

Files: the two NWB files of DANDI 000294 (patch clamp + HD-MEA on a shared clock) and, per cell, from
github.com/alejoe91/multimodalfitting/cell_models/<cell>: morphology_corrected.swc (SWC type 4 marks the axon
initial segment), fitting/efeatures/probe_BPO.json and template_BPO.npy (electrodes above threshold in the cell
frame, used to register the morphology onto the MEA).

    t = extract_template("sub-3148_....nwb")
    reg = register_morphology(t, "probe_BPO.json", "template_BPO.npy", "morphology_corrected.swc")
    grid = template_grid(t)                        # (220, 120, samples) at 20 kHz, -2 .. +6 ms
    gt = ground_truth_in_block(reg.morphology, t)  # axon inside the recorded block
"""

import json
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from axon_tracking.simulation.evaluation import GroundTruth, ground_truth
from axon_tracking.simulation.morphology import Morphology, load_swc

PITCH = 17.5


@dataclass
class PatchTemplate:
    template: np.ndarray  # (electrodes, samples) mean, uV
    sem: np.ndarray  # (electrodes, samples)
    n_spikes: int
    electrodes: np.ndarray  # MaxWell electrode ids (x = id % 220, y = id // 220)
    fs: float
    pre: int  # samples before the spike
    spike_times: np.ndarray  # s, shared clock

    @property
    def positions(self):
        return np.column_stack([self.electrodes % 220, self.electrodes // 220]) * PITCH


def patch_spike_times(nwb, threshold_mv=-20.0, refractory_s=2e-3):
    """Spike times (s) of the patched neuron: upward threshold crossings in all current-clamp sweeps."""
    acq = nwb["acquisition"]
    spikes = []
    for k in acq:
        if not k.startswith("response_"):
            continue
        v, t = acq[k]["data"][:], acq[k]["timestamps"][:]
        up = np.flatnonzero((v[:-1] < threshold_mv) & (v[1:] >= threshold_mv))
        if len(up):
            keep = np.r_[True, np.diff(t[up]) > refractory_s]
            spikes.append(t[up][keep])
    return np.sort(np.concatenate(spikes)) if spikes else np.zeros(0)


def extract_template(nwb_path, highpass_hz=150.0, ms_before=2.0, ms_after=6.0, margin=4000):
    """Spike-triggered average of the HD-MEA on the intracellular spikes (5th order Butterworth highpass,
    forward-backward, as the pipeline's filter)."""
    import h5py
    import scipy.signal

    f = h5py.File(nwb_path, "r")
    acq = f["acquisition"]
    spikes = patch_spike_times(f)
    el = f["general/extracellular_ephys/electrodes"]
    electrodes = el["electrode"][:].astype(int)
    gain, offset = el["gain_to_uV"][:].astype(np.float32), el["offset_to_uV"][:].astype(np.float32)
    runs = sorted(k for k in acq if k.startswith("ElectricalSeries_"))
    fs = 1.0 / np.median(np.diff(acq[runs[0]]["timestamps"][:1000]))
    pre, post = int(round(ms_before * fs / 1000)), int(round(ms_after * fs / 1000))
    sos = scipy.signal.butter(5, highpass_hz, btype="highpass", fs=fs, output="sos")
    total = np.zeros((len(electrodes), pre + post))
    total_sq = np.zeros_like(total)
    n, used = 0, []
    for run in runs:
        es = acq[run]
        ts = es["timestamps"][:]
        ref = es["electrodes"][:]
        idx = np.searchsorted(ts, spikes[(spikes >= ts[0]) & (spikes <= ts[-1])])
        idx = idx[(idx >= pre + margin) & (idx < len(ts) - post - margin)]
        block = es["data"].chunks[0] if es["data"].chunks else 50000
        for b0 in range(0, len(ts), block):
            in_block = idx[(idx >= b0) & (idx < b0 + block)]
            if not len(in_block):
                continue
            lo, hi = max(b0 - margin - pre, 0), min(b0 + block + margin + post, len(ts))
            filt = scipy.signal.sosfiltfilt(sos, es["data"][lo:hi].astype(np.float32) * gain[ref] + offset[ref], axis=0)
            for s in in_block:
                w = filt[s - lo - pre : s - lo + post].T
                total[ref] += w
                total_sq[ref] += w.astype(np.float64) ** 2
            n += len(in_block)
            used.append(ts[in_block])
    mean = total / max(n, 1)
    sd = np.sqrt(np.maximum(total_sq / max(n, 1) - mean ** 2, 0))
    return PatchTemplate(mean.astype(np.float32), (sd / np.sqrt(max(n, 1))).astype(np.float32), n, electrodes, fs, pre,
                         np.concatenate(used) if used else np.zeros(0))


@dataclass
class Registration:
    rotation_deg: float
    mirrored: bool
    translation_um: np.ndarray
    amplitude_correlation: float
    matched_fraction: float
    morphology: Morphology  # in MEA coordinates (um), z = 5 um


def register_morphology(t: PatchTemplate, probe_json, template_npy, swc, n_candidates=5, tol_um=4.0):
    """Rigid transform cell frame (soma at the origin) -> MEA from the electrodes the morphology was fitted with.

    The rotation (mod 90 deg) follows from the lattice of their positions; rotation by multiples of 90 deg,
    mirroring and translation are chosen to maximise the correlation of their template amplitudes with ours."""
    bpo = np.array(json.load(open(probe_json))["pos"])[:, :2]
    p2p_bpo = np.ptp(np.load(template_npy), axis=1)
    mea, p2p = t.positions, np.ptp(t.template, axis=1)
    nn = cKDTree(bpo).query(bpo, k=2)[1][:, 1]
    vec = bpo[nn] - bpo
    theta0 = np.median(np.mod(np.arctan2(vec[:, 1], vec[:, 0]), np.pi / 2))
    tree, best = cKDTree(mea), None
    for flip in (False, True):
        for k in range(4):
            th = theta0 + k * np.pi / 2
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            A = np.linalg.inv(R @ (np.diag([-1.0, 1.0]) if flip else np.eye(2)))
            q = bpo @ A.T
            for i in np.argsort(p2p_bpo)[-n_candidates:]:
                for j in np.argsort(p2p)[-n_candidates:]:
                    tr = mea[j] - q[i]
                    dist, idx = tree.query(q + tr)
                    ok = dist < tol_um
                    if ok.sum() < 0.8 * len(q):
                        continue
                    c = np.corrcoef(p2p_bpo[ok], p2p[idx[ok]])[0, 1]
                    if best is None or c > best[0]:
                        best = (c, A, tr, flip, np.degrees(th), ok.mean())
    if best is None:
        raise ValueError("no consistent registration found")
    c, A, tr, flip, th, matched = best
    m = load_swc(swc, type_map={4: "axon"})
    soma = m.mid[m.kind == "soma"].mean(axis=0)
    move = lambda p: np.column_stack([(p[:, :2] - soma[:2]) @ A.T + tr, np.full(len(p), 5.0)])
    return Registration(float(th % 360), bool(flip), tr, float(c), float(matched),
                        replace(m, start=move(m.start), end=move(m.end)))


def template_grid(t: PatchTemplate):
    """Template on the pipeline grid (220, 120, samples); zeros for electrodes that were not recorded."""
    grid = np.zeros((220, 120, t.template.shape[1]), np.float32)
    grid[t.electrodes % 220, t.electrodes // 220] = t.template
    return grid


def ground_truth_in_block(morph: Morphology, t: PatchTemplate, max_distance_um=PITCH) -> GroundTruth:
    """Ground truth restricted to the axon within max_distance_um of a recorded electrode."""
    gt = ground_truth(morph, np.zeros(len(morph)))
    inside = cKDTree(t.positions).query(gt.axon_points)[0] <= max_distance_um
    return replace(gt, axon_points=gt.axon_points[inside], axon_latency_ms=gt.axon_latency_ms[inside],
                   axon_detectable=gt.axon_detectable[inside])


def pipeline_params(t: PatchTemplate, **overrides):
    """part3 parameters for the 20 kHz templates (time upsampled 2x, as for 10 kHz templates upsampled 4x)."""
    p = dict(noise_threshold=-0.75, abs_threshold=-0.1, upsample=[0.5, 0.5, 0.5], sampling_rate=t.fs / 0.5,
             ms_cutout=[t.pre / t.fs * 1000, (t.template.shape[1] - t.pre) / t.fs * 1000], max_velocity=0.8,
             el_spacing=PITCH * 0.5, ais_detection="dev", buffer_frames=10)
    return {**p, **overrides}
