"""Transmembrane currents of a single action potential.

prescribed_currents: fast model without NEURON. The AP propagates along the axon from the AIS with a given
    velocity; for a travelling wave V(x, t) = f(t - x/v) the cable equation gives the membrane current
    i_m = (pi d^2 / 4 R_i) * f''(t - x/v) / v^2 per unit length (triphasic extracellular waveform). The soma
    fires a spike (sink) balanced by the dendrites (sources); all currents sum to zero at every time.
simulate_neuron: biophysical simulation with NEURON (somatic synaptic stimulation, active soma and axon), ported
    from axon_velocity (Buccino et al. 2022, J Neural Eng, MIT license).
"""

import os
from dataclasses import dataclass

import numpy as np

from axon_tracking.simulation.morphology import Morphology


@dataclass
class Currents:
    t_ms: np.ndarray  # (n_t,)
    imem_na: np.ndarray  # (n_segments, n_t) membrane current, outward positive (nA)
    spike_time_ms: float  # time of the somatic spike (trigger of the spike-triggered average)
    latency_ms: np.ndarray  # (n_segments,) AP arrival time relative to the somatic spike (NaN: not axonal)


def ap_waveform(t_ms, amplitude_mv=100.0, width_ms=0.15, ahp_fraction=0.1, ahp_delay_ms=0.6, ahp_width_ms=0.8,
                decay_width_ms=None):
    """Smooth AP shape (depolarisation + afterhyperpolarisation) and its first and second derivatives (mV/ms^n).

    decay_width_ms: width of the repolarisation (default: width_ms, symmetric). Real APs rise faster than they
    repolarise; the width then changes smoothly from width_ms to decay_width_ms around the peak (derivatives are
    computed numerically along the last axis, which must be uniformly sampled)."""
    t = np.asarray(t_ms, float)
    def gauss(t, s):
        g = np.exp(-t ** 2 / (2 * s ** 2))
        return g, -t / s ** 2 * g, (t ** 2 / s ** 4 - 1 / s ** 2) * g
    h, dh, ddh = gauss(t - ahp_delay_ms, ahp_width_ms)
    a, b = amplitude_mv, -ahp_fraction * amplitude_mv
    if decay_width_ms is None or decay_width_ms == width_ms:
        g, dg, ddg = gauss(t, width_ms)
    else:
        s = width_ms + (decay_width_ms - width_ms) / (1 + np.exp(-t / (0.25 * width_ms)))
        g = np.exp(-t ** 2 / (2 * s ** 2))
        dt = np.diff(t, axis=-1).mean() if t.shape[-1] > 1 else 1.0
        dg = np.gradient(g, dt, axis=-1)
        ddg = np.gradient(dg, dt, axis=-1)
    return a * g + b * h, a * dg + b * dh, a * ddg + b * ddh


def prescribed_currents(morph: Morphology, velocity_m_s=0.5, dt_ms=0.005, t_pre_ms=2.0, t_post_ms=8.0,
                        ri_ohm_cm=100.0, soma_delay_ms=0.1, soma_gain=10.0, ap_kwargs=None,
                        dendrite_velocity_m_s=None, dendrite_length_constant_um=150.0, return_length_constant_um=None,
                        dendrite_ap_kwargs=None):
    """Membrane currents for one AP with prescribed propagation (see module docstring).

    dendrite_velocity_m_s: back-propagating AP in the dendrites, V = exp(-x / lambda) f(t - x / v), with the
    cable current pi d^2 / (4 R_i) * exp(-x / lambda) (f / lambda^2 + 2 f' / (lambda v) + f'' / v^2); None: passive
    dendrites that only return the net current of the cell.
    return_length_constant_um: the net current of the cell returns through the dendrites with weights
    area * exp(-x / lambda_return) (x: distance from the soma); None: in proportion to the area only.
    dendrite_ap_kwargs: AP shape in the dendrites (ap_waveform arguments; default: as in the axon). Dendritic APs
    are broader and smaller than axonal ones."""
    ap_kwargs = ap_kwargs or {}
    n_pre = int(round(t_pre_ms / dt_ms))
    t = (np.arange(n_pre + int(round(t_post_ms / dt_ms))) - n_pre) * dt_ms  # 0 = somatic spike
    axon = morph.kind == "axon"
    dist = morph.path_distance("axon")  # um from the AIS start
    latency = np.where(axon, dist / (velocity_m_s * 1000.0) - soma_delay_ms, np.nan)  # AIS fires before the soma
    imem = np.zeros((len(morph), len(t)))

    # axon: travelling wave, SI units: d [m], L [m], R_i [ohm m], f'' [V/s^2], v [m/s] -> A
    d, L = morph.diam[axon] * 1e-6, morph.length[axon] * 1e-6
    ri = ri_ohm_cm * 1e-2
    _, _, ddf = ap_waveform(t[None, :] - latency[axon][:, None], **ap_kwargs)  # mV/ms^2 = 1e3 V/s^2
    imem[axon] = (np.pi * d[:, None] ** 2 / (4 * ri)) * (ddf * 1e3) / velocity_m_s ** 2 * L[:, None] * 1e9  # nA

    # soma spike: inward current during the upstroke (sink), balanced by the dendrites below. soma_gain scales
    # the capacitive current c_m * A * dV/dt to the net active current (calibrate against recorded amplitudes)
    soma = morph.kind == "soma"
    _, df, _ = ap_waveform(t, **ap_kwargs)  # mV/ms = V/s
    cm = 1e-2  # F/m^2 (1 uF/cm^2)
    soma_area = morph.area[soma] * 1e-12  # m^2
    imem[soma] = -soma_gain * cm * soma_area[:, None] * df[None, :] * 1e9  # nA

    dend = morph.kind == "dend"
    if dendrite_velocity_m_s is not None and dend.any():
        x = morph.path_distance("dend")[dend] - morph.length[dend] / 2  # um from the soma (segment middle)
        lam, v = dendrite_length_constant_um * 1e-6, dendrite_velocity_m_s
        f, df_d, ddf_d = ap_waveform(t[None, :] - x[:, None] / (v * 1000.0), **(dendrite_ap_kwargs or ap_kwargs))
        d2v = np.exp(-x * 1e-6 / lam)[:, None] * (f * 1e-3 / lam ** 2 + 2 * df_d / (lam * v) + ddf_d * 1e3 / v ** 2)
        dd, Ld = morph.diam[dend] * 1e-6, morph.length[dend] * 1e-6
        imem[dend] = (np.pi * dd[:, None] ** 2 / (4 * ri)) * d2v * Ld[:, None] * 1e9  # nA

    # current conservation: the rest of the cell (dendrites, or soma if no dendrites) returns the net current
    sink = morph.kind == "dend" if np.any(morph.kind == "dend") else soma
    net = imem.sum(axis=0)
    w = morph.area[sink].copy()
    if return_length_constant_um is not None and np.any(morph.kind == "dend"):
        w *= np.exp(-(morph.path_distance("dend")[sink] - morph.length[sink] / 2) / return_length_constant_um)
    w /= w.sum()
    imem[sink] -= w[:, None] * net[None, :]
    return Currents(t_ms=t, imem_na=imem, spike_time_ms=0.0, latency_ms=latency)


# --------------------------------------------------------------------------------------------- NEURON
DEFAULT_BIOPHYSICS = dict(Kv1_soma=100, Kv1_axon=400, na_axon=500, na_soma=500, rm=15000, ra=80, cm=1, celsius=33,
                          v_init=-85, ena=55, ek=-98)  # axon_velocity.models.get_default_biophysics_params


def simulate_neuron(swc_path, mechanisms_dir, params=None, dt_ms=0.005, tstop_ms=20.0, max_segment_um=10.0,
                    syn_times_ms=(2.0, 3.0, 4.0), syn_weight=(0.02, 0.05, 0.1, 0.2), plane_z=None, t_pre_ms=2.0,
                    t_post_ms=8.0):
    """Simulate one AP with NEURON and return (Morphology, Currents) for the forward model.

    Requires NEURON (Windows installer or pip on Linux/macOS) and the compiled mechanisms of axon_velocity
    (na8st/nax8st -> 'na'/'nax', Kv1_axonal -> 'Kv1'): run nrnivmodl in `mechanisms_dir`.
    syn_weight: synaptic weight (uS), or increasing weights tried until the soma spikes (axon_velocity used
        0.02-0.05 depending on the cell).
    plane_z: flatten the morphology onto z = plane_z before simulating (as axon_velocity.planarize_swc), so that
    the electrical lengths match the flattened geometry of the forward model.
    """
    import tempfile

    import neuron
    from neuron import h

    p = {**DEFAULT_BIOPHYSICS, **(params or {})}
    neuron.load_mechanisms(str(mechanisms_dir))
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    if plane_z is not None:
        swc = np.loadtxt(swc_path, comments="#", ndmin=2)
        swc[:, 4] = plane_z
        with tempfile.NamedTemporaryFile("w", suffix=".swc", delete=False) as f:
            np.savetxt(f, swc, fmt=["%d", "%d", "%.4f", "%.4f", "%.4f", "%.4f", "%d"])
        swc_path = f.name
    reader = h.Import3d_SWC_read()
    reader.input(str(swc_path))
    h.Import3d_GUI(reader, 0).instantiate(None)
    if plane_z is not None:
        os.remove(swc_path)
    sections = list(h.allsec())

    for sec in sections:
        sec.nseg = max(int(np.ceil(sec.L / max_segment_um)), 1) | 1  # odd number of segments
        name = sec.name()
        if "soma" in name:
            sec.insert("na"); sec.gbar_na = p["na_soma"]
            sec.insert("Kv1"); sec.gbar_Kv1 = p["Kv1_soma"]
        elif "axon" in name:
            sec.insert("nax"); sec.gbar_nax = p["na_axon"]
            sec.insert("Kv1"); sec.gbar_Kv1 = p["Kv1_axon"]
        if "dend" not in name and "apic" not in name:
            sec.ena, sec.ek = p["ena"], p["ek"]
        sec.insert("pas")
        sec.g_pas, sec.e_pas = 1.0 / p["rm"], p["v_init"]
        sec.Ra, sec.cm = p["ra"], p["cm"]

    # segment geometry from the 3D points (arc-length interpolation)
    starts, ends, diams, kinds, segs = [], [], [], [], []
    for sec in sections:
        n3d = int(h.n3d(sec=sec))
        pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n3d)])
        arc = np.array([h.arc3d(i, sec=sec) for i in range(n3d)]) / max(sec.L, 1e-9)
        interp = lambda x: np.array([np.interp(x, arc, pts[:, k]) for k in range(3)])
        kind = "soma" if "soma" in sec.name() else ("axon" if "axon" in sec.name() else "dend")
        for seg in sec:
            x0, x1 = seg.x - 0.5 / sec.nseg, seg.x + 0.5 / sec.nseg
            starts.append(interp(x0)); ends.append(interp(x1)); diams.append(seg.diam); kinds.append(kind)
            segs.append(seg)
    starts, ends = np.array(starts), np.array(ends)

    # parent of each segment from NEURON's topology: the previous segment of the section, or for the first
    # segment the segment of the parent section it connects to
    first = np.cumsum([0] + [sec.nseg for sec in sections])
    sec_index = {sec.hname(): k for k, sec in enumerate(sections)}
    parents = np.arange(len(segs)) - 1
    for k, sec in enumerate(sections):
        ps = sec.parentseg()
        if ps is None:
            parents[first[k]] = -1
        else:
            j = sec_index[ps.sec.hname()]
            parents[first[k]] = first[j] + min(int(ps.x * ps.sec.nseg), ps.sec.nseg - 1)
    morph = Morphology(starts, ends, np.array(diams), np.array(kinds), parents)

    # stimulation and recording
    soma = next(s for s in sections if "soma" in s.name())
    syn = h.ExpSyn(soma(0.5)); syn.e, syn.tau = 0, 2
    stims = [h.NetStim() for _ in syn_times_ms]  # one event each (VecStim needs an extra mechanism)
    for stim, t_ms in zip(stims, syn_times_ms):
        stim.number, stim.start, stim.noise = 1, t_ms, 0
    netcons = [h.NetCon(stim, syn) for stim in stims]
    h.cvode.use_fast_imem(1)
    rec_i = [h.Vector().record(seg._ref_i_membrane_) for seg in segs]
    rec_v = [h.Vector().record(seg._ref_v) for seg in segs]
    h.dt = dt_ms
    h.celsius = p["celsius"]
    soma_idx = np.flatnonzero(morph.kind == "soma")
    for weight in np.atleast_1d(syn_weight):
        for nc in netcons:
            nc.weight[0] = weight
        h.finitialize(p["v_init"])
        h.continuerun(tstop_ms)
        vmem = np.array([np.array(r) for r in rec_v])
        if vmem[soma_idx].max() > -20:  # somatic APs of cells with large dendritic trees can peak below 0 mV
            break
    else:
        raise RuntimeError(f"the soma did not spike with synaptic weights {syn_weight} uS")
    imem = np.array([np.array(r) for r in rec_i])
    t = np.arange(imem.shape[1]) * dt_ms

    # spike time: first upward crossing of -20 mV at the soma; latency of every segment: its first crossing
    # (NaN if the AP does not reach it)
    up = (vmem[:, :-1] < -20) & (vmem[:, 1:] >= -20)
    first_up = np.where(up.any(axis=1), t[np.argmax(up, axis=1)], np.nan)
    t_soma = np.nanmin(first_up[soma_idx])
    latency = np.where(morph.kind == "axon", first_up - t_soma, np.nan)
    keep = (t >= t_soma - t_pre_ms) & (t < t_soma + t_post_ms)
    return morph, Currents(t_ms=t[keep] - t_soma, imem_na=imem[:, keep], spike_time_ms=0.0, latency_ms=latency)
