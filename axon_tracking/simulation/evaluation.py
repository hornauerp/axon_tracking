"""Compare reconstructed axons with the simulated ground truth."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from axon_tracking.simulation.forward import PITCH, Footprint, electrode_positions
from axon_tracking.simulation.morphology import Morphology


@dataclass
class GroundTruth:
    axon_points: np.ndarray  # (n, 2) axon sampled every `step` um
    axon_latency_ms: np.ndarray  # (n,) arrival time relative to the somatic spike
    axon_detectable: np.ndarray  # (n,) signal above k * template noise at a nearby electrode
    branch_points: np.ndarray  # (m, 2)
    soma_xy: np.ndarray  # (2,)
    foreign_points: np.ndarray  # (k, 2) somata and axons of all other neurons
    step_um: float
    template_noise_uv: float = None  # noise sd of the averaged template
    contamination_p2p: np.ndarray = None  # (220, 120) peak-to-peak of the other neurons' part of the template


def sample_segments(start, end, step=2.0):
    pts = []
    for a, b in zip(start, end):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        pts.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    return np.concatenate(pts) if pts else np.zeros((0, start.shape[1]))


def ground_truth(morph: Morphology, latency_ms, footprint: Footprint = None, template_noise_uv=None, others=(),
                 k_detect=3.0, step=2.0, contamination_p2p=None, max_latency_ms=None):
    """Ground truth of one neuron. Axon points are 'detectable' if the footprint's peak-to-peak amplitude at an
    electrode within one pitch exceeds k_detect times the noise of the averaged template. max_latency_ms: leave
    out the axon the AP reaches after the end of the template window (it cannot be reconstructed)."""
    axon = morph.kind == "axon"
    if max_latency_ms is not None:
        axon &= ~(np.asarray(latency_ms) > max_latency_ms)
    pts, lat = [], []
    for i in np.flatnonzero(axon):
        n = max(int(np.ceil(morph.length[i] / step)), 1)
        pts.append(morph.start[i, :2] + np.linspace(0, 1, n, endpoint=False)[:, None] * (morph.end[i, :2] - morph.start[i, :2]))
        lat.append(np.full(n, latency_ms[i]))
    pts, lat = np.concatenate(pts), np.concatenate(lat)
    # one point per step x step cell: uniform density also for densely sampled reconstructions (many tiny segments)
    _, first = np.unique(np.floor(pts / step).astype(np.int64), axis=0, return_index=True)
    first = np.sort(first)
    pts, lat = pts[first], lat[first]
    detectable = np.ones(len(pts), bool)
    if footprint is not None and template_noise_uv is not None:
        epos = electrode_positions()[footprint.electrodes]
        amp = np.ptp(footprint.data_uv, axis=1)
        nearby = cKDTree(epos).query_ball_point(pts, r=PITCH)
        detectable = np.array([len(n) > 0 and amp[n].max() >= k_detect * template_noise_uv for n in nearby])
    foreign = [sample_segments(o.start[o.kind != "dend", :2], o.end[o.kind != "dend", :2], step) for o in others]
    return GroundTruth(axon_points=pts, axon_latency_ms=lat, axon_detectable=detectable,
                       branch_points=morph.branch_points("axon")[:, :2],
                       soma_xy=morph.mid[morph.kind == "soma", :2].mean(axis=0),
                       foreign_points=np.concatenate(foreign) if foreign else np.zeros((0, 2)), step_um=step,
                       template_noise_uv=template_noise_uv, contamination_p2p=contamination_p2p)


def evaluate(paths, gt: GroundTruth, tol_um=25.0, branch_points=None, bp_tol_um=35.0, k_contamination=3.0):
    """Cable precision/recall of reconstructed paths (list of (n, >=2) arrays, um) against the ground truth.

    precision: share of reconstructed cable within tol_um of the true axon
    recall / recall_detectable: share of the (detectable) true axon within tol_um of the reconstruction
    fp_on_foreign: share of reconstructed cable that misses the axon but lies on another neuron (in dense
        cultures largely by chance)
    fp_contamination: share of reconstructed cable that misses the axon where the other neurons' part of the
        template exceeds k_contamination times the template noise (caused by co-active neurons)
    """
    paths = [np.asarray(p, float)[:, :2] for p in paths if len(p) > 1]
    rec = np.concatenate([sample_segments(p[:-1], p[1:], gt.step_um) for p in paths]) if paths else np.zeros((0, 2))
    out = dict(cable_um=len(rec) * gt.step_um, gt_cable_um=len(gt.axon_points) * gt.step_um,
               gt_detectable_cable_um=gt.axon_detectable.sum() * gt.step_um)
    if len(rec) == 0:
        out.update(precision=np.nan, recall=0.0, recall_detectable=0.0, f1=0.0, fp_on_foreign=np.nan)
    else:
        d_to_gt = cKDTree(gt.axon_points).query(rec)[0]
        hit = d_to_gt <= tol_um
        covered = cKDTree(rec).query(gt.axon_points)[0] <= tol_um
        precision = hit.mean()
        recall = covered.mean()
        recall_det = covered[gt.axon_detectable].mean() if gt.axon_detectable.any() else np.nan
        on_foreign = (cKDTree(gt.foreign_points).query(rec[~hit])[0] <= tol_um).mean() \
            if len(gt.foreign_points) and (~hit).any() else 0.0
        f1 = 2 * precision * recall_det / (precision + recall_det) if precision + recall_det > 0 else 0.0
        out.update(precision=precision, recall=recall, recall_detectable=recall_det, f1=f1,
                   fp_on_foreign=on_foreign * (1 - precision))
        if gt.contamination_p2p is not None and gt.template_noise_uv:
            col = np.clip(np.rint(rec[:, 0] / PITCH).astype(int), 0, gt.contamination_p2p.shape[0] - 1)
            row = np.clip(np.rint(rec[:, 1] / PITCH).astype(int), 0, gt.contamination_p2p.shape[1] - 1)
            contaminated = gt.contamination_p2p[col, row] > k_contamination * gt.template_noise_uv
            out["fp_contamination"] = float(np.mean(~hit & contaminated))
            out["fp_other"] = float(np.mean(~hit & ~contaminated))
    if branch_points is not None:
        est = np.asarray(branch_points, float).reshape(-1, 2)
        true = gt.branch_points
        if len(est) and len(true):
            out["bp_precision"] = (cKDTree(true).query(est)[0] <= bp_tol_um).mean()
            out["bp_recall"] = (cKDTree(est).query(true)[0] <= bp_tol_um).mean()
        else:
            out["bp_precision"] = np.nan if not len(est) else 0.0
            out["bp_recall"] = np.nan if not len(true) else 0.0
    return out
