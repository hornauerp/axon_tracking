"""Background neurons: cheap footprints for the many neurons of a dense culture that are not simulated in detail.

Real axon-scan templates average a few hundred spikes. The spikes of thousands of unmodelled neighbours, partly
co-active in network bursts, do not average out and leave heavy-tailed, spatially correlated residuals: large
deviations are far more frequent than for white noise, as frequent before the trigger as after it, and
neighbouring electrodes correlate. A culture of a few dozen detailed neurons with white recording noise has none
of this. Each background neuron is a soma (point source above the chip) with a meandering axon of random length;
its spike is a fixed extracellular waveform that propagates along the axon.
"""

import numpy as np
from scipy.spatial import cKDTree

from axon_tracking.simulation.forward import Footprint, electrode_positions
from axon_tracking.simulation.morphology import CHIP_UM


def spike_waveform(t_ms, rise_ms=0.15, decay_ms=0.3, rebound=0.25, rebound_delay_ms=0.6, rebound_width_ms=0.4):
    """Extracellular spike with a unit negative peak at t = 0 and a slower positive rebound."""
    t = np.asarray(t_ms, float)
    width = np.where(t < 0, rise_ms, decay_ms)
    return -np.exp(-0.5 * (t / width) ** 2) + rebound * np.exp(-0.5 * ((t - rebound_delay_ms) / rebound_width_ms) ** 2)


def _field(d_um, height_um):
    """Spatial decay of a source height_um above the chip (between monopole and dipole: 1 below the source)."""
    return height_um ** 2 / (d_um ** 2 + height_um ** 2)


def background_footprints(n, rng, fs=10000.0, n_samples=80, spike_index=10, soma_amp_uv=(30.0, 0.7),
                          soma_height_um=15.0, soma_radius_um=80.0, axon_length_um=(0.0, 1500.0),
                          axon_amp_uv=(4.0, 0.5), axon_height_um=8.0, axon_radius_um=30.0, velocity_m_s=(0.3, 0.8),
                          step_um=10.0, turn_sd_rad=0.1, margin_um=100.0):
    """n Footprints of background neurons (somata uniform over the chip and margin_um around it).

    Amplitudes (negative peak on the electrode below the source) are lognormal: (median uV, sigma of log).
    The axon is a random walk (direction changes by turn_sd_rad per step_um) and is cut where its spike would
    leave the footprint window."""
    pos = electrode_positions()
    tree = cKDTree(pos)
    t_ms = (np.arange(n_samples) - spike_index) / fs * 1000
    max_latency = t_ms[-1] - 1.0
    footprints = []
    for _ in range(n):
        soma = rng.uniform([-margin_um, -margin_um], [CHIP_UM[0] + margin_um, CHIP_UM[1] + margin_um])
        soma_amp = rng.lognormal(np.log(soma_amp_uv[0]), soma_amp_uv[1])
        v = rng.uniform(*velocity_m_s)
        n_steps = int(rng.uniform(*axon_length_um) / step_um)
        angle = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, turn_sd_rad, n_steps))
        points = soma + step_um * np.cumsum(np.column_stack([np.cos(angle), np.sin(angle)]), axis=0)
        latency = 0.1 + step_um * np.arange(1, n_steps + 1) / (v * 1000)
        points, latency = points[latency <= max_latency], latency[latency <= max_latency]
        # per-point amplitude such that an electrode below the axon sees ~axon_amp (sum over nearby points)
        point_amp = rng.lognormal(np.log(axon_amp_uv[0]), axon_amp_uv[1]) * step_um / (np.pi * axon_height_um)
        electrodes = set(tree.query_ball_point(soma, soma_radius_um))
        for p in tree.query_ball_point(points, axon_radius_um) if len(points) else []:
            electrodes.update(p)
        electrodes = np.array(sorted(electrodes), int)
        if not len(electrodes):
            continue
        d_soma = np.linalg.norm(pos[electrodes] - soma, axis=1)
        data = soma_amp * _field(d_soma, soma_height_um)[:, None] * spike_waveform(t_ms)[None, :]
        if len(points):
            d = np.linalg.norm(pos[electrodes][:, None, :] - points[None, :, :], axis=2)
            weights = point_amp * _field(d, axon_height_um) * (d <= axon_radius_um)
            data += weights @ spike_waveform(t_ms[None, :] - latency[:, None])
        footprints.append(Footprint(data.astype(np.float32), electrodes, fs, spike_index))
    return footprints
