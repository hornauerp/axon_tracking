"""Extracellular footprints on the MaxWell electrode grid with LFPykit (cells in medium on a non-conducting chip)."""

from dataclasses import dataclass

import numpy as np

from axon_tracking.simulation.currents import Currents
from axon_tracking.simulation.morphology import Morphology

N_COLS, N_ROWS, PITCH = 220, 120, 17.5


def electrode_positions():
    """(26400, 2) x/y positions (um); electrode id = row * 220 + col, as in MaxWell files."""
    ids = np.arange(N_COLS * N_ROWS)
    return np.column_stack([(ids % N_COLS) * PITCH, (ids // N_COLS) * PITCH])


@dataclass
class Footprint:
    data_uv: np.ndarray  # (n_electrodes, n_samples) extracellular potential of one spike (uV)
    electrodes: np.ndarray  # (n_electrodes,) electrode ids
    fs: float  # Hz
    spike_index: int  # sample of the somatic spike (trigger)


def compute_footprint(morph: Morphology, currents: Currents, fs=10000.0, sigma=1.5, max_distance_um=250.0,
                      method="linesource", chunk=1024, sigma_glass=0.0, sigma_saline=None):
    """Potential at all electrodes within max_distance_um of the cell, resampled to fs.

    By default the medium (conductivity sigma, S/m) is homogeneous above a non-conducting chip, for which the
    potential is exactly twice the infinite-medium one. Give sigma_saline / sigma_glass for LFPykit's layered
    MEA model (cell layer, saline above, chip below). Cells must lie above the electrode plane (z > 0).
    """
    from lfpykit import CellGeometry, RecExtElectrode, RecMEAElectrode
    from scipy.signal import resample_poly
    from scipy.spatial import cKDTree

    if np.min([morph.start[:, 2].min(), morph.end[:, 2].min()]) <= 0:
        raise ValueError("cells must lie above the chip (z > 0)")
    pos = electrode_positions()
    seg_pts = np.concatenate([morph.start[:, :2], morph.mid[:, :2], morph.end[:, :2]])
    near = cKDTree(seg_pts).query(pos, distance_upper_bound=max_distance_um)[0] < np.inf
    electrodes = np.flatnonzero(near)

    cell = CellGeometry(x=np.column_stack([morph.start[:, 0], morph.end[:, 0]]),
                        y=np.column_stack([morph.start[:, 1], morph.end[:, 1]]),
                        z=np.column_stack([morph.start[:, 2], morph.end[:, 2]]), d=morph.diam)
    h = float(max(morph.start[:, 2].max(), morph.end[:, 2].max()) + 100.0)
    # the forward model and the resampling are both linear: resample the currents first (far fewer samples)
    fs_in = 1000.0 / (currents.t_ms[1] - currents.t_ms[0])
    up, down = _ratio(fs, fs_in)
    imem = resample_poly(currents.imem_na, up, down, axis=1)
    v = np.zeros((len(electrodes), imem.shape[1]))
    for k in range(0, len(electrodes), chunk):
        e = electrodes[k : k + chunk]
        if sigma_glass == 0.0 and sigma_saline is None and method == "linesource":
            # homogeneous medium above a non-conducting plane: exactly twice the infinite-medium potential
            epos = np.column_stack([pos[e], np.zeros(len(e))])
            M = 2.0 * line_source_matrix(epos, morph.start, morph.end, morph.diam, sigma)
        elif sigma_glass == 0.0 and sigma_saline is None:
            el = RecExtElectrode(cell, sigma=sigma, x=pos[e, 0], y=pos[e, 1], z=np.zeros(len(e)), method=method)
            M = 2.0 * el.get_transformation_matrix()
        else:
            el = RecMEAElectrode(cell, sigma_T=sigma, sigma_S=sigma_saline or sigma, sigma_G=sigma_glass, h=h,
                                 z_shift=0.0, x=pos[e, 0], y=pos[e, 1], z=np.zeros(len(e)), method=method)
            M = el.get_transformation_matrix()
        v[k : k + len(e)] = M @ imem  # mV
    v *= 1000.0  # uV

    spike_index = int(round(np.searchsorted(currents.t_ms, currents.spike_time_ms) * up / down))
    return Footprint(data_uv=v.astype(np.float32), electrodes=electrodes, fs=fs, spike_index=spike_index)


def line_source_matrix(electrodes_xyz, start, end, diam, sigma):
    """Infinite-medium line-source transformation matrix (mV per nA), vectorised.

    For a segment of length L carrying current I uniformly, the potential at an electrode with axial
    coordinate a (along the segment, from its start) and perpendicular distance rho is
    I / (4 pi sigma L) * (asinh((L - a) / rho) - asinh(-a / rho)); rho is clamped to the segment radius
    as in LFPy/LFPykit.
    """
    r = electrodes_xyz[:, None, :] - start[None, :, :]  # (E, S, 3)
    seg = end - start
    L = np.maximum(np.linalg.norm(seg, axis=1), 1e-9)
    u = seg / L[:, None]
    a = np.einsum("esk,sk->es", r, u)
    rho = np.sqrt(np.maximum(np.einsum("esk,esk->es", r, r) - a ** 2, 0.0))
    rho = np.maximum(rho, diam[None, :] / 2)
    return (np.arcsinh((L[None, :] - a) / rho) - np.arcsinh(-a / rho)) / (4 * np.pi * sigma * L[None, :])


def _ratio(fs_out, fs_in):
    from fractions import Fraction
    f = Fraction(fs_out / fs_in).limit_denominator(1000)
    return f.numerator, f.denominator
