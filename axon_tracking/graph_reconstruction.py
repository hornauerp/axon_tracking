"""Axon reconstruction as a maximum-weight arborescence on the electrodes (experimental).

Nodes are electrodes whose negative peak exceeds z_min times their noise. Directed edges go from an electrode to
neighbours within max_edge_um whose peak comes later, if the implied conduction velocity is plausible; an edge
is worth the evidence of its target (z) minus penalties for its length and for deviating from the velocity
range. Since edges only point from earlier to later electrodes (latency order), the graph is acyclic and the
maximum-weight tree rooted at the electrode with the largest negative peak (the AIS) is found exactly in one pass:
no loops by construction, branches may start anywhere. Branches whose summed evidence stays below
min_branch_z are pruned. Electrodes with dendritic waveforms (broad, positive first) can be left out.

Node threshold: template noise of axon scans is heavy-tailed and spatially correlated (residual spikes of other
neurons that do not average out). By default, the template is smoothed across neighbouring electrodes first
(axonal footprints span several electrodes, most noise does not) and nodes need a fixed z_min. Alternatively
(node_fdr), the threshold is raised until the expected share of false nodes, estimated from a decoy (the window
before the trigger, or a null template of the unit), is at most node_fdr; weaker electrodes (z_min <= z < that
threshold) can then still join through short edges (weak_edge_um, i.e. adjacent electrodes) with a latency
consistent with propagation (hysteresis thresholding, as in Canny edge detection): faint axons are continuous,
noise rarely is. Decoy-based thresholds become unreliable when the decoy does not hold the same background as the
template (e.g. network bursts), which is why the fixed threshold is the default.
"""

import networkx as nx
import numpy as np

from axon_tracking import path_evidence as pe

PITCH = 17.5


def electrode_features(grid, fs, t0_index):
    """Per recorded electrode: (col, row), negative peak amplitude, sub-sample latency (ms), noise sd, z, half-width,
    positive/negative ratio."""
    recorded = np.abs(grid).max(axis=2) > 0
    noise = pe.template_noise_sd(grid)
    amp, lat = pe.peak_maps(grid, fs, t0_index)
    width, ratio = pe.waveform_maps(grid, fs, t0_index)
    # parabolic interpolation of the peak time
    start = max(int(t0_index - 0.5 * fs / 1000), 0)
    idx = np.clip(np.rint(lat * fs / 1000).astype(int) + t0_index, 1, grid.shape[2] - 2)
    c, r = np.nonzero(recorded)
    y0, y1, y2 = (grid[c, r, idx[c, r] + k] for k in (-1, 0, 1))
    denom = y0 - 2 * y1 + y2
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    latency = (idx[c, r] + np.clip(shift, -0.5, 0.5) - t0_index) / fs * 1000
    z = amp[c, r] / np.maximum(noise[c, r], 1e-9)
    return dict(cr=np.column_stack([c, r]), amp=amp[c, r], latency=latency, noise=noise[c, r], z=z,
                halfwidth=width[c, r], ratio=ratio[c, r])


def remove_zero_lag_copies(grid, t0_index, fs, min_distance_um=250.0, max_shift=1):
    """Remove scaled copies of the somatic waveform that appear at the same time far from the soma.

    A neuron's own signal reaches electrodes min_distance_um away only after conduction (> 0.25 ms at 1 m/s), so a
    copy of the soma waveform within +- max_shift samples there is crosstalk or another neuron. On every electrode
    beyond min_distance_um, the least-squares projection onto the (shifted) soma waveform is subtracted. Returns a
    new grid."""
    recorded = np.abs(grid).max(axis=2) > 0
    c, r = np.unravel_index(np.argmin(np.where(recorded, grid.min(axis=2), 0)), grid.shape[:2])
    soma = grid[c, r].astype(np.float64)
    cc, rr = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
    far = recorded & (np.hypot(cc - c, rr - r) * PITCH >= min_distance_um)
    out = grid.astype(np.float64).copy()
    W = out[far]  # (electrodes, samples)
    best = np.zeros(len(W))
    best_copy = np.zeros_like(W)
    for shift in range(-max_shift, max_shift + 1):
        s = np.roll(soma, shift)
        if shift > 0:
            s[:shift] = 0
        elif shift < 0:
            s[shift:] = 0
        a = W @ s / max(s @ s, 1e-12)
        explained = a ** 2 * (s @ s)
        better = explained > best
        best[better] = explained[better]
        best_copy[better] = a[better, None] * s[None, :]
    out[far] = W - best_copy
    return out.astype(grid.dtype)


def spatial_smooth(grid, sigma_el=0.7):
    """Gaussian smoothing across neighbouring electrodes (sigma in electrode pitches) at every time sample, over the
    recorded electrodes only (normalised convolution): axonal footprints span 2-3 electrodes, noise less so."""
    from scipy.ndimage import gaussian_filter

    recorded = (np.abs(grid).max(axis=2) > 0).astype(np.float64)
    weight = gaussian_filter(recorded, sigma_el, mode="constant")
    out = gaussian_filter(grid * recorded[:, :, None], (sigma_el, sigma_el, 0), mode="constant")
    out = np.divide(out, weight[:, :, None], out=np.zeros_like(out), where=weight[:, :, None] > 1e-6)
    return (out * recorded[:, :, None]).astype(grid.dtype)


def false_node_rate(grid, fs, t0_index, z, min_lag_ms=-0.5, null_grid=None):
    """Share of electrodes that reach z by noise alone: from the null template if given (same search window), else
    from the window before min_lag_ms, scaled to the length of the search window (as decoy_threshold)."""
    recorded = np.abs(grid).max(axis=2) > 0
    start = max(int(t0_index + min_lag_ms * fs / 1000), 0)
    sd = np.maximum(pe.template_noise_sd(grid), 1e-9)[recorded]
    if null_grid is not None:
        return float(np.mean(-np.nan_to_num(np.asarray(null_grid, float))[recorded, start:].min(axis=1) / sd >= z))
    if start < 5:
        return 0.0
    decoy = -grid[recorded, :start].min(axis=1) / sd
    return float(1 - (1 - np.mean(decoy >= z)) ** ((grid.shape[2] - start) / start))


def auto_spur_nodes(n_tree_nodes, noise_rate, budget=1.0, neighbours=8, max_nodes=10):
    """Shortest spur length (electrodes) that noise is unlikely to produce: a chain of L noise electrodes next to
    the tree occurs about n_tree_nodes * (neighbours * noise_rate)^L times; L is chosen so that this stays below
    budget."""
    q = neighbours * noise_rate
    if q <= 0:
        return 1
    if q >= 1:
        return max_nodes
    return int(np.clip(np.ceil(np.log(budget / max(n_tree_nodes, 1)) / np.log(q)), 1, max_nodes))


def centerline_cable(paths, window=5):
    """Cable length (um) of paths measured along a moving average of their points (window electrodes): the length
    of the course of an axon rather than of the electrode-to-electrode steps through its footprint."""
    total = 0.0
    for p in paths:
        xy = np.asarray(p, float)[:, :2]
        if len(xy) < 2:
            continue
        k = min(window, len(xy))
        kernel = np.ones(k) / k
        pad = np.pad(xy, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        smooth = np.column_stack([np.convolve(pad[:, i], kernel, "valid") for i in range(2)])
        smooth[0], smooth[-1] = xy[0], xy[-1]
        total += float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
    return total


def decoy_threshold(grid, fs, t0_index, fdr=0.1, z_min=4.0, min_lag_ms=-0.5, z_max=50.0, step=0.25, null_grid=None):
    """Smallest z threshold >= z_min at which the expected share of false nodes is at most fdr.

    The node statistic (largest negative deflection from min_lag_ms on, in noise sd) is compared with the same
    statistic of a decoy: by default the window before min_lag_ms, its exceedance rate scaled to the length of the
    search window assuming independent blocks; with null_grid (a null template of the unit, e.g. from windows of
    the axon scan in which it was silent: template_extraction null_templates), the same window of the null.
    Returns z_max if no threshold qualifies."""
    recorded = np.abs(grid).max(axis=2) > 0
    start = max(int(t0_index + min_lag_ms * fs / 1000), 0)
    if (start < 5 and null_grid is None) or not recorded.any():
        return z_min
    sd = np.maximum(pe.template_noise_sd(grid), 1e-9)[recorded]
    target = -grid[recorded, start:].min(axis=1) / sd
    if null_grid is not None:
        decoy = -np.nan_to_num(np.asarray(null_grid, float))[recorded, start:].min(axis=1) / sd
        blocks = 1.0
    else:
        decoy = -grid[recorded, :start].min(axis=1) / sd
        blocks = (grid.shape[2] - start) / start
    for tau in np.arange(z_min, z_max, step):
        n = np.sum(target >= tau)
        false = len(target) * (1 - (1 - np.mean(decoy >= tau)) ** blocks)
        if n == 0 or false <= fdr * n:
            return float(tau)
    return z_max


def reconstruct(grid, fs, t0_index, z_min=5.0, max_edge_um=150.0, velocity_range=(0.1, 2.0), latency_sd_ms=0.1,
                length_penalty=0.05, min_branch_z=20.0, dendrites="exclude", dendrite_penalty=5.0, max_ratio=1.0,
                max_halfwidth_ms=0.35, root_radius_um=60.0, node_fdr=None, weak_edge_um=None, weak_max_dev=1.0,
                null_grid=None, crosstalk_um=None, smooth_sigma=0.8, z_strong=None, min_path_r2=None,
                soma_region_ms=None, soma_region_um=200.0, min_spur_nodes=None, width_spurs=None,
                min_spur_latency_ms=0.1, ridge_refine=True, ridge_max_um=60.0, bridge_um=None,
                bridge_velocity=(0.2, 1.0), bridge_min_nodes=4):
    """Tree (list of paths in um, each (n, 3): x, y, latency ms) from a template grid (220, 120, samples).

    The defaults were chosen on simulated ground truth: simulated cultures and hybrids of simulated footprints on
    real template backgrounds, checked against patched neurons with imaged morphology. They favour a fixed node
    threshold after spatial smoothing and ridge refinement; the remaining options (decoy thresholds, pruning,
    somatic region, bridging) are off by default: they trade recall for precision (or the reverse) and depend on
    the data.

    dendrites: electrodes with dendritic waveforms (positive/negative ratio > max_ratio or half-width >
    max_halfwidth_ms) are left out ("exclude"), cost dendrite_penalty (in z units) per node ("penalty"), or are
    treated like the others ("none").

    node_fdr: if not None, nodes need the decoy threshold for this false node rate (see decoy_threshold); with
    weak_edge_um, electrodes between z_min and that threshold are kept, but only reached by edges up to weak_edge_um
    whose latency deviates by at most weak_max_dev latency sds from the velocity range. null_grid: null template
    of the unit as decoy (see decoy_threshold) instead of the pre-trigger window.

    crosstalk_um: remove zero-lag copies of the soma waveform beyond this distance first (remove_zero_lag_copies);
    smooth_sigma: Gaussian smoothing across electrodes first (in electrode pitches; spatial_smooth, also applied to
    null_grid; None: off).
    z_strong: fixed threshold for strong nodes instead of the decoy threshold (hysteresis with weak_edge_um).
    min_path_r2: prune terminal paths (>= 4 electrodes) whose latency is not linear in the distance along the path
    (R2 below this), repeated up to 3 times.
    soma_region_ms: the somatic region (electrodes connected to the root through neighbours, within soma_region_um,
    peaking within +- soma_region_ms of the root: soma and dendrites, which fire almost simultaneously) is treated as
    part of the root: its electrodes connect without velocity check, and segments within it are not returned (the
    axon starts where the tree leaves it). Without it, the tree fills large somatic/dendritic footprints.
    min_spur_nodes: remove terminal branches (leaf back to its branch point) with fewer electrodes, repeatedly
    (spurs through the width of an axon's footprint; each is also a false branch point); "auto": the length noise
    is unlikely to produce given this template's false node rate (auto_spur_nodes, false_node_rate).
    width_spurs: remove terminal branches that stay within width_spurs times the local half-width of the footprint
    at their branch point (measured from the amplitude profile, so it scales with the axon's diameter) and gain less
    than min_spur_latency_ms of latency: steps through the width of a thick axon's footprint, not collaterals.
    ridge_refine: move every node across the path to the centre of the footprint (amplitude-weighted centre of the
    contiguous part of the perpendicular amplitude profile above half its maximum, within ridge_max_um): the tree
    otherwise runs along the edges of wide (thick axons') footprints. Branch points are moved once, so paths stay
    connected.
    bridge_um: attach fragments the tree did not reach (>= bridge_min_nodes strong, connected electrodes) through a
    gap of up to bridge_um, if the latency difference implies a velocity within bridge_velocity (m/s, +- 2 latency
    sds) and the fragment's latency increases away from its entry point (it propagates outwards). Fragments are
    attached in latency order, so attached fragments can anchor further ones. Bridges are straight segments."""
    if crosstalk_um is not None:
        grid = remove_zero_lag_copies(grid, t0_index, fs, crosstalk_um)  # (the null has no somatic copies)
    if smooth_sigma:
        grid = spatial_smooth(grid, smooth_sigma)
        if null_grid is not None:
            null_grid = spatial_smooth(null_grid, smooth_sigma)
    if z_strong is None:
        z_strong = (decoy_threshold(grid, fs, t0_index, node_fdr, z_min, null_grid=null_grid) if node_fdr is not None
                    else z_min)
    if weak_edge_um is None:
        z_min = z_strong
    f = electrode_features(grid, fs, t0_index)
    keep = f["z"] >= z_min
    dendritic = (f["ratio"] > max_ratio) | (f["halfwidth"] > max_halfwidth_ms)
    if dendrites == "exclude":
        keep &= ~dendritic
    node_penalty = np.where(dendritic, dendrite_penalty, 0.0) if dendrites == "penalty" else np.zeros(len(keep))
    root_candidates = np.flatnonzero(f["z"] >= z_min)
    if not len(root_candidates):
        return []
    root = root_candidates[np.argmax(f["amp"][root_candidates])]
    keep[root] = True
    nodes = np.flatnonzero(keep)
    xy = f["cr"][nodes] * PITCH
    lat, z = f["latency"][nodes], f["z"][nodes] - node_penalty[nodes]
    weak = f["z"][nodes] < z_strong
    from scipy.spatial import cKDTree

    root_local = int(np.flatnonzero(nodes == root)[0])
    # edges only from earlier to later electrodes (latency, then index): the graph is acyclic, so the
    # maximum-weight arborescence is found exactly by giving every node, in this order, its best incoming edge
    # from a node that is already in the tree
    # the root comes first: the AIS region fires within ~0.2 ms, so electrodes within root_radius_um of the root
    # may peak slightly before it and connect to it without the velocity check
    rank = np.empty(len(nodes), int)
    rank[np.lexsort((np.arange(len(nodes)), lat))] = np.arange(len(nodes))
    rank[root_local] = -1
    region = np.zeros(len(nodes), bool)
    region[root_local] = True
    if soma_region_ms is not None:
        # grow from the root over neighbouring electrodes that peak about simultaneously with it
        from scipy.spatial import cKDTree as _Tree

        near = _Tree(xy).query_ball_point(xy, 1.5 * PITCH)
        ok_node = (np.abs(lat - lat[root_local]) <= soma_region_ms) & \
            (np.linalg.norm(xy - xy[root_local], axis=1) <= soma_region_um)
        stack = [root_local]
        while stack:
            v = stack.pop()
            for w in near[v]:
                if not region[w] and ok_node[w]:
                    region[w] = True
                    stack.append(w)
    pairs = cKDTree(xy).query_pairs(max(max_edge_um, root_radius_um), output_type="ndarray")
    if not len(pairs):
        return []
    a, b = np.where((rank[pairs[:, 0]] < rank[pairs[:, 1]])[:, None], pairs, pairs[:, ::-1]).T
    d = np.linalg.norm(xy[a] - xy[b], axis=1)
    dt = lat[b] - lat[a]
    lo, hi = d / (velocity_range[1] * 1000), d / (velocity_range[0] * 1000)  # ms (um / (mm/s))
    # deviation of the latency difference from the plausible range, in units of the latency uncertainty
    dev = np.maximum(np.maximum(lo - dt, dt - hi), 0.0) / latency_sd_ms
    from_root = ((a == root_local) & (d <= root_radius_um)) | (region[a] & region[b])
    dev[from_root] = 0.0
    ok = ((dev <= 3) & (d <= max_edge_um)) | from_root
    if weak_edge_um is not None:
        ok &= ~weak[b] | ((d <= weak_edge_um) & (dev <= weak_max_dev))
    a, b, weight = a[ok], b[ok], z[b[ok]] - length_penalty * d[ok] - 0.5 * dev[ok] ** 2
    incoming = {}
    for k in np.argsort(rank[b], kind="stable"):
        incoming.setdefault(int(b[k]), []).append(k)
    T = nx.DiGraph()
    T.add_node(root_local)
    for v in np.argsort(rank):
        if v == root_local or int(v) not in incoming:
            continue
        candidates = [k for k in incoming[int(v)] if a[k] in T]
        if candidates:
            best = max(candidates, key=lambda k: weight[k])
            T.add_edge(int(a[best]), int(v))
    if T.number_of_nodes() < 2:
        return []
    # prune: drop subtrees whose summed evidence is below min_branch_z (bottom-up)
    order = list(nx.dfs_postorder_nodes(T, root_local))
    total = {}
    for v in order:
        total[v] = z[v] + sum(total[k] for k in T.successors(v))
    for v in order:
        if v != root_local and v in T and total[v] < min_branch_z:
            T.remove_nodes_from(nx.descendants(T, v) | {v})
    if bridge_um is not None:
        from scipy.spatial import cKDTree as _Tree

        adjacency = _Tree(xy).query_ball_point(xy, 1.5 * PITCH)
        free = np.array([v not in T and z[v] >= z_strong for v in range(len(nodes))])
        seen = np.zeros(len(nodes), bool)
        fragments = []
        for v in np.flatnonzero(free):
            if seen[v]:
                continue
            comp, stack = [], [v]
            seen[v] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adjacency[u]:
                    if free[w] and not seen[w]:
                        seen[w] = True
                        stack.append(w)
            if len(comp) >= bridge_min_nodes:
                fragments.append(np.array(comp))
        for comp in sorted(fragments, key=lambda c: lat[c].min()):
            entry = comp[np.argmin(lat[comp])]
            # the fragment must propagate away from its entry point
            dist = np.linalg.norm(xy[comp] - xy[entry], axis=1)
            if len(comp) >= 3 and np.ptp(lat[comp]) > 0 and np.corrcoef(dist, lat[comp])[0, 1] < 0.3:
                continue
            tree_nodes = np.array(list(T.nodes))
            d = np.linalg.norm(xy[tree_nodes] - xy[entry], axis=1)
            dt = lat[entry] - lat[tree_nodes]
            lo_t, hi_t = d / (bridge_velocity[1] * 1000), d / (bridge_velocity[0] * 1000)
            ok = (d <= bridge_um) & (dt >= lo_t - 2 * latency_sd_ms) & (dt <= hi_t + 2 * latency_sd_ms)
            if not ok.any():
                continue
            anchor = int(tree_nodes[np.flatnonzero(ok)[np.argmin(d[ok])]])
            T.add_edge(anchor, int(entry))
            # wire the fragment: every node (latency order) to its nearest earlier node of the fragment
            order = comp[np.argsort(lat[comp], kind="stable")]
            placed = [int(entry)]
            for u in order:
                if u == entry:
                    continue
                pd_ = np.linalg.norm(xy[placed] - xy[u], axis=1)
                T.add_edge(placed[int(np.argmin(pd_))], int(u))
                placed.append(int(u))
    if min_spur_nodes == "auto":
        min_spur_nodes = auto_spur_nodes(T.number_of_nodes(),
                                         false_node_rate(grid, fs, t0_index, z_strong if z_strong is not None else z_min,
                                                         null_grid=null_grid))
    if min_spur_nodes is not None:
        for _ in range(5):
            removed = False
            for leaf in [v for v in T if T.out_degree(v) == 0 and v != root_local]:
                if leaf not in T:
                    continue
                branch = [leaf]
                while branch[-1] != root_local and T.out_degree(next(iter(T.predecessors(branch[-1])))) == 1:
                    branch.append(next(iter(T.predecessors(branch[-1]))))
                parent = next(iter(T.predecessors(branch[-1]))) if branch[-1] != root_local else None
                # only spurs that leave a branch point (an unbranched axon from the root is kept whole)
                if parent is not None and T.out_degree(parent) > 1 and len(branch) < min_spur_nodes:
                    T.remove_nodes_from(branch)
                    removed = True
            if not removed:
                break
    if width_spurs is not None:
        amp_all = f["amp"]
        cr_all = f["cr"]
        from scipy.spatial import cKDTree as _Tree

        el_tree = _Tree(cr_all * PITCH)
        node_amp = amp_all[nodes]

        def half_width(v):
            """Distance from node v within which the footprint stays above half of v's amplitude (contiguously:
            the largest distance of such electrodes within 150 um, stepping outwards in pitch-wide rings)."""
            idx = el_tree.query_ball_point(xy[v], 150.0)
            d = np.linalg.norm(cr_all[idx] * PITCH - xy[v], axis=1)
            above = amp_all[idx] >= 0.5 * node_amp[v]
            w = 0.0
            for ring in np.arange(PITCH, 150.0 + 1e-9, PITCH):
                in_ring = (d > ring - PITCH) & (d <= ring)
                if not in_ring.any() or not above[in_ring].any():
                    break
                w = ring
            return w

        for _ in range(5):
            removed = False
            for leaf in [v for v in T if T.out_degree(v) == 0 and v != root_local]:
                if leaf not in T:
                    continue
                branch = [leaf]
                while branch[-1] != root_local and T.out_degree(next(iter(T.predecessors(branch[-1])))) == 1:
                    branch.append(next(iter(T.predecessors(branch[-1]))))
                if branch[-1] == root_local:
                    continue
                parent = next(iter(T.predecessors(branch[-1])))
                if T.out_degree(parent) < 2:
                    continue
                reach = np.linalg.norm(xy[branch] - xy[parent], axis=1).max()
                gain = lat[branch].max() - lat[parent]
                if reach <= width_spurs * max(half_width(parent), PITCH) and gain < min_spur_latency_ms:
                    T.remove_nodes_from(branch)
                    removed = True
            if not removed:
                break
    if min_path_r2 is not None:
        # terminal paths (from the last branch point to a leaf) whose latencies do not follow the distance along the
        # path (a propagating AP conducts at a roughly constant velocity; noise is only ordered, not linear)
        for _ in range(3):
            removed = False
            for leaf in [v for v in T if T.out_degree(v) == 0 and v != root_local]:
                path = [leaf]
                while path[-1] != root_local and T.out_degree(next(iter(T.predecessors(path[-1])))) == 1:
                    path.append(next(iter(T.predecessors(path[-1]))))
                if path[-1] != root_local:
                    path.append(next(iter(T.predecessors(path[-1]))))  # include the branch point
                path = path[::-1]
                if len(path) < 4:
                    continue
                dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy[path], axis=0), axis=1))]
                r = np.corrcoef(dist, lat[path])[0, 1] if np.ptp(lat[path]) > 0 else 0.0
                if r ** 2 < min_path_r2:
                    T.remove_nodes_from(path[1:])
                    removed = True
            if not removed:
                break
    if ridge_refine:
        from scipy.ndimage import map_coordinates

        amp_grid = np.zeros(grid.shape[:2])
        amp_grid[f["cr"][:, 0], f["cr"][:, 1]] = f["amp"]
        offsets = np.arange(-ridge_max_um, ridge_max_um + 1e-9, PITCH / 2)
        refined = xy.astype(float).copy()

        def along(v, steps):
            """Node `steps` positions up (negative) or down (positive, first child) the tree from v."""
            u = v
            for _ in range(abs(steps)):
                nxt = (next(iter(T.predecessors(u)), None) if steps < 0 else next(iter(T.successors(u)), None))
                if nxt is None:
                    break
                u = nxt
            return u

        for v in T:
            if v == root_local:
                continue
            direction = xy[along(v, 2)] - xy[along(v, -2)]
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            perp = np.array([-direction[1], direction[0]]) / norm
            pos = xy[v][None, :] + offsets[:, None] * perp[None, :]
            profile = map_coordinates(amp_grid, (pos / PITCH).T, order=1, mode="constant")
            k = int(np.argmax(profile))
            if profile[k] <= 0:
                continue
            above = profile >= 0.5 * profile[k]
            lo, hi = k, k
            while lo > 0 and above[lo - 1]:
                lo -= 1
            while hi < len(profile) - 1 and above[hi + 1]:
                hi += 1
            w = profile[lo : hi + 1]
            refined[v] = xy[v] + perp * float(np.sum(offsets[lo : hi + 1] * w) / np.sum(w))
        xy = refined
    # paths between root/branch points and leaves/branch points
    paths = []
    for v in nx.dfs_preorder_nodes(T, root_local):
        if v == root_local or T.out_degree(next(iter(T.predecessors(v)))) > 1:
            path = [next(iter(T.predecessors(v)))] if v != root_local else []
            u = v
            while True:
                path.append(u)
                succ = list(T.successors(u))
                if len(succ) != 1:
                    break
                u = succ[0]
            if len(path) > 1:
                # leave out segments within the somatic region
                piece = [path[0]]
                for u, w in zip(path[:-1], path[1:]):
                    if region[u] and region[w]:
                        if len(piece) > 1:
                            paths.append(np.column_stack([xy[piece], lat[piece]]))
                        piece = [w]
                    else:
                        piece.append(w)
                if len(piece) > 1:
                    paths.append(np.column_stack([xy[piece], lat[piece]]))
    return paths
