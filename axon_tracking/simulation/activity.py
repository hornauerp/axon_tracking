"""Spike trains of a culture: tonic firing plus network bursts with cell-type dependent participation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellTypeParams:
    tonic_rate_hz: float = 0.5  # Poisson firing outside bursts
    burst_participation: float = 0.9  # probability of firing in a network burst
    spikes_per_burst: float = 3.0  # mean number of spikes per participated burst
    intra_burst_isi_ms: float = 8.0  # mean extra inter-spike interval within a burst (on top of refractoriness)


@dataclass
class NetworkParams:
    duration_s: float = 1200.0
    burst_rate_hz: float = 0.1  # network bursts per second (0: no bursts)
    latency_sd_ms: float = 20.0  # spread of the neurons' mean latencies relative to burst onset
    jitter_ms: float = 10.0  # burst-to-burst jitter of a neuron's first spike around its mean latency
    refractory_ms: float = 2.0
    cell_types: dict = field(default_factory=lambda: {
        "excitatory": CellTypeParams(tonic_rate_hz=0.3, burst_participation=0.9, spikes_per_burst=4.0),
        "inhibitory": CellTypeParams(tonic_rate_hz=2.0, burst_participation=0.5, spikes_per_burst=2.0),
    })


def burst_onsets(params: NetworkParams, rng):
    """Network burst onsets (s): Poisson times."""
    T = params.duration_s
    return np.sort(rng.uniform(0, T, rng.poisson(params.burst_rate_hz * T)))


def simulate_spike_trains(types, params: NetworkParams, rng, onsets=None):
    """Spike times (s) for neurons of the given types (list of keys of params.cell_types).

    Bursts start at Poisson times; neuron i joins with its type's participation probability, first spike at
    onset + mean latency_i + jitter, followed by further spikes. A small jitter_ms makes neurons fire with
    ms-precise relative timing in bursts (leader/follower sequences), a large one only loosely co-active.
    onsets: burst onsets to use (to add neurons to the same network later); drawn with burst_onsets otherwise.
    """
    T = params.duration_s
    onsets = burst_onsets(params, rng) if onsets is None else onsets
    mean_latency = rng.normal(0, params.latency_sd_ms, len(types)) / 1000.0
    trains = []
    for i, cell_type in enumerate(types):
        p = params.cell_types[cell_type]
        spikes = [rng.uniform(0, T, rng.poisson(p.tonic_rate_hz * T))]
        for b in onsets:
            if rng.random() >= p.burst_participation:
                continue
            n = max(rng.poisson(p.spikes_per_burst), 1)
            first = b + mean_latency[i] + rng.normal(0, params.jitter_ms / 1000.0)
            isis = params.refractory_ms / 1000.0 + rng.exponential(p.intra_burst_isi_ms / 1000.0, n - 1)
            spikes.append(first + np.concatenate([[0], np.cumsum(isis)]))
        s = np.sort(np.concatenate(spikes))
        s = s[(s >= 0) & (s < T)]
        keep = np.ones(len(s), bool)  # enforce refractoriness
        last = -np.inf
        for k, x in enumerate(s):
            if x - last < params.refractory_ms / 1000.0:
                keep[k] = False
            else:
                last = x
        trains.append(s[keep])
    return trains
