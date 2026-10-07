"""Neuron morphologies as line segments (um), from SWC files or generated as random branching axons."""

from dataclasses import dataclass, replace

import numpy as np

CHIP_UM = (220 * 17.5, 120 * 17.5)  # MaxOne/MaxTwo electrode area (x, y)
SWC_KIND = {1: "soma", 2: "axon", 3: "dend", 4: "dend"}


@dataclass
class Morphology:
    start: np.ndarray  # (n, 3) segment start points (um)
    end: np.ndarray  # (n, 3) segment end points (um)
    diam: np.ndarray  # (n,) diameters (um)
    kind: np.ndarray  # (n,) "soma", "axon" or "dend"
    parent: np.ndarray  # (n,) index of the parent segment, -1 for the root

    @property
    def mid(self):
        return (self.start + self.end) / 2

    @property
    def length(self):
        return np.linalg.norm(self.end - self.start, axis=1)

    @property
    def area(self):
        return np.pi * self.diam * np.maximum(self.length, 1e-3)

    def __len__(self):
        return len(self.diam)

    def children(self):
        kids = [[] for _ in range(len(self))]
        for i, p in enumerate(self.parent):
            if p >= 0:
                kids[p].append(i)
        return kids

    def path_distance(self, from_kind="axon"):
        """Distance along the tree from the start of the first segment of `from_kind` (e.g. the AIS)
        to the end of every segment of that kind; NaN for other segments."""
        dist = np.full(len(self), np.nan)
        roots = [i for i in range(len(self)) if self.kind[i] == from_kind
                 and (self.parent[i] < 0 or self.kind[self.parent[i]] != from_kind)]
        kids = self.children()
        stack = [(r, self.length[r]) for r in roots]
        while stack:
            i, d = stack.pop()
            dist[i] = d
            stack.extend((k, d + self.length[k]) for k in kids[i] if self.kind[k] == from_kind)
        return dist

    def branch_points(self, kind="axon"):
        """End points of segments of `kind` with more than one child of the same kind (um, (m, 3))."""
        kids = self.children()
        idx = [i for i in range(len(self)) if self.kind[i] == kind and sum(self.kind[k] == kind for k in kids[i]) > 1]
        return self.end[idx]

    def transformed(self, rotation_deg=0.0, offset=(0.0, 0.0, 0.0), plane_z=None, flip=False):
        """Mirror (x, if flip), rotate around the soma (z axis), translate, and optionally flatten onto z = plane_z."""
        soma = self.mid[self.kind == "soma"].mean(axis=0) if np.any(self.kind == "soma") else self.mid.mean(axis=0)
        th = np.deg2rad(rotation_deg)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        if flip:
            R = R @ np.diag([-1.0, 1.0, 1.0])
        def move(p):
            q = (p - soma) @ R.T + soma + np.asarray(offset, float)
            if plane_z is not None:
                q[:, 2] = plane_z
            return q
        return replace(self, start=move(self.start), end=move(self.end))

    def coarsened(self, min_length=10.0):
        """Merge unbranched chains of short segments (densely traced reconstructions) into segments of about
        min_length (um), keeping branch points and compartment boundaries; diameters are length-weighted."""
        kids = self.children()
        new_of = np.full(len(self), -1)
        starts, ends, diams, kinds, parents = [], [], [], [], []
        heads = [i for i in range(len(self)) if self.parent[i] < 0]
        while heads:
            i = heads.pop()
            chain = [i]
            length = self.length[i]
            while length < min_length:
                same = [k for k in kids[chain[-1]] if self.kind[k] == self.kind[i]]
                if len(kids[chain[-1]]) != 1 or len(same) != 1:
                    break
                chain.append(same[0])
                length += self.length[same[0]]
            k = len(starts)
            starts.append(self.start[chain[0]]); ends.append(self.end[chain[-1]])
            diams.append(np.average(self.diam[chain], weights=np.maximum(self.length[chain], 1e-9)))
            kinds.append(self.kind[i])
            parents.append(new_of[self.parent[i]] if self.parent[i] >= 0 else -1)
            new_of[chain] = k
            heads.extend(kids[chain[-1]])
        return Morphology(np.array(starts), np.array(ends), np.array(diams), np.array(kinds), np.array(parents, int))

    def subdivided(self, max_length=10.0):
        """Split segments longer than max_length (um) for an accurate forward model."""
        starts, ends, diams, kinds, parents = [], [], [], [], []
        first_piece_of, last_piece_of = {}, {}
        for i in range(len(self)):
            n = max(int(np.ceil(self.length[i] / max_length)), 1)
            pts = self.start[i] + np.linspace(0, 1, n + 1)[:, None] * (self.end[i] - self.start[i])
            first_piece_of[i] = len(starts)
            for k in range(n):
                starts.append(pts[k]); ends.append(pts[k + 1]); diams.append(self.diam[i]); kinds.append(self.kind[i])
                parents.append(len(starts) - 2 if k > 0 else None)  # first piece: resolved below
            last_piece_of[i] = len(starts) - 1
        for i in range(len(self)):
            p = self.parent[i]
            parents[first_piece_of[i]] = last_piece_of[p] if p >= 0 else -1
        return Morphology(np.array(starts), np.array(ends), np.array(diams), np.array(kinds), np.array(parents, int))


def load_swc(path, soma_length=None, type_map=None):
    """Read an SWC file into segments (point-to-parent). A single-point soma becomes a short cylinder.
    type_map: SWC type -> kind overrides (e.g. {4: "axon"} where type 4 marks the axon initial segment)."""
    kind_of = {**SWC_KIND, **(type_map or {})}
    data = np.loadtxt(path, comments="#", ndmin=2)
    ids = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    xyz = data[:, 2:5]
    radius = data[:, 5]
    parent_ids = data[:, 6].astype(int)
    index = {pid: k for k, pid in enumerate(ids)}
    starts, ends, diams, kinds, parents = [], [], [], [], []
    seg_ending_at = {}
    order = np.argsort(ids)
    for k in order:
        p = parent_ids[k]
        if p < 0 or p not in index:
            continue
        pk = index[p]
        starts.append(xyz[pk]); ends.append(xyz[k]); diams.append(2 * radius[k])
        kinds.append(kind_of.get(types[k], "dend"))
        parents.append(seg_ending_at.get(ids[pk], -1))
        seg_ending_at[ids[k]] = len(starts) - 1
    soma_pts = np.flatnonzero(types == 1)
    if len(soma_pts) == 1 or not any(k == "soma" for k in kinds):
        k = soma_pts[0] if len(soma_pts) else 0
        r = radius[k]
        L = soma_length or 2 * r
        c = xyz[k]
        starts.insert(0, c - [L / 2, 0, 0]); ends.insert(0, c + [L / 2, 0, 0]); diams.insert(0, 2 * r)
        kinds.insert(0, "soma"); parents = [-1] + [p + 1 if p >= 0 else 0 for p in parents]
    return Morphology(np.array(starts, float), np.array(ends, float), np.array(diams, float),
                      np.array(kinds), np.array(parents, int))


def with_random_dendrites(morph: Morphology, rng, total_length_um=4000.0, n_primary=(5, 9), step=10.0,
                          persistence_deg=15.0, branch_prob=0.04, max_radius_um=300.0, diam=(2.0, 0.6)):
    """Replace the dendrites by a random branching tree around the soma (low-magnification reconstructions miss
    most of the dendrites). Diameters taper from diam[0] at the soma to diam[1] at max_radius_um."""
    keep = morph.kind != "dend"
    index = np.cumsum(keep) - 1
    starts, ends = list(morph.start[keep]), list(morph.end[keep])
    diams, kinds = list(morph.diam[keep]), list(morph.kind[keep])
    parents = [index[p] if p >= 0 and keep[p] else -1 for p in morph.parent[keep]]
    soma_idx = int(np.flatnonzero(np.array(kinds) == "soma")[0])
    centre = morph.mid[morph.kind == "soma"].mean(axis=0)
    radius = max(morph.diam[morph.kind == "soma"].max() / 2, 3.0)
    n0 = rng.integers(n_primary[0], n_primary[1] + 1)
    tips = []
    for a in rng.uniform(0, 2 * np.pi, n0):
        tips.append((centre + radius * np.array([np.cos(a), np.sin(a), 0.0]), a, soma_idx))
    added = 0.0
    while tips and added < total_length_um:
        k = rng.integers(len(tips))
        p, a, parent = tips.pop(k)
        a = a + np.deg2rad(rng.normal(0, persistence_deg))
        q = p + step * np.array([np.cos(a), np.sin(a), 0.0])
        r = np.linalg.norm(q[:2] - centre[:2])
        if r > max_radius_um:
            continue
        starts.append(p); ends.append(q); kinds.append("dend"); parents.append(parent)
        diams.append(diam[0] + (diam[1] - diam[0]) * min(r / max_radius_um, 1.0))
        added += step
        tips.append((q, a, len(starts) - 1))
        if rng.random() < branch_prob:
            tips.append((q, a + np.deg2rad(rng.choice([-1, 1]) * rng.uniform(20, 60)), len(starts) - 1))
    return Morphology(np.array(starts), np.array(ends), np.array(diams), np.array(kinds), np.array(parents, int))


def with_diameters(morph: Morphology, axon_um=0.5, ais_um=1.0, ais_length_um=40.0, dend_um=1.5, soma_um=None):
    """Assign diameters by compartment (for reconstructions without reliable diameters, e.g. low magnification):
    axon_um, ais_um for the first ais_length_um of every axon root, dend_um (None: keep); the soma keeps its
    diameter unless soma_um is given."""
    diam = np.where(morph.kind == "axon", axon_um, morph.diam).astype(float)
    if dend_um is not None:
        diam[morph.kind == "dend"] = dend_um
    if soma_um is not None:
        diam[morph.kind == "soma"] = soma_um
    dist = morph.path_distance("axon")
    ais = (morph.kind == "axon") & (dist - morph.length <= ais_length_um)
    diam[ais] = ais_um
    return replace(morph, diam=diam)


def random_neuron(rng, soma_xy, plane_z=5.0, axon_length=(1500, 4000), step=10.0, persistence_deg=8.0,
                  branch_prob=0.015, branch_angle_deg=(25, 70), ais_length=30.0, axon_diam=0.8, ais_diam=1.2,
                  soma_diam=15.0, n_dendrites=(4, 7), dendrite_length=(60, 150), dendrite_diam=1.5, bounds=CHIP_UM):
    """Random neuron: soma, short dendrites and a branching axon (persistent random walk) on the chip."""
    soma_xy = np.asarray(soma_xy, float)
    segs = []  # (start, end, diam, kind, parent)
    c = np.array([*soma_xy, plane_z])
    segs.append((c - [soma_diam / 2, 0, 0], c + [soma_diam / 2, 0, 0], soma_diam, "soma", -1))

    def grow(start, angle, length, diam, kind, parent, branching):
        tips = [(start, angle, length, parent)]
        while tips:
            pos, ang, remaining, par = tips.pop()
            while remaining > 0:
                ang += np.deg2rad(rng.normal(0, persistence_deg))
                new = pos + step * np.array([np.cos(ang), np.sin(ang), 0])
                if not (0 <= new[0] <= bounds[0] and 0 <= new[1] <= bounds[1]):
                    break
                segs.append((pos, new, diam, kind, par))
                par, pos, remaining = len(segs) - 1, new, remaining - step
                if branching and rng.random() < branch_prob:
                    side = rng.choice([-1, 1]) * np.deg2rad(rng.uniform(*branch_angle_deg))
                    tips.append((pos, ang + side, remaining * rng.uniform(0.3, 0.9), par))

    for _ in range(rng.integers(*n_dendrites, endpoint=True)):
        ang = rng.uniform(0, 2 * np.pi)
        start = c + soma_diam / 2 * np.array([np.cos(ang), np.sin(ang), 0])
        grow(start, ang, rng.uniform(*dendrite_length), dendrite_diam, "dend", 0, branching=False)
    ang = rng.uniform(0, 2 * np.pi)
    start = c + soma_diam / 2 * np.array([np.cos(ang), np.sin(ang), 0])
    ais_end = start + ais_length * np.array([np.cos(ang), np.sin(ang), 0])
    segs.append((start, ais_end, ais_diam, "axon", 0))
    grow(ais_end, ang, rng.uniform(*axon_length), axon_diam, "axon", len(segs) - 1, branching=True)

    s, e, d, k, p = zip(*segs)
    return Morphology(np.array(s, float), np.array(e, float), np.array(d, float), np.array(k), np.array(p, int))
