"""Data-driven checks of an axon scan sorting (no ground truth needed).

- Trigger recall: in triggered axon tracking files, every online spike on a trigger electrode starts or extends a
  stored snippet. A sorting should have a spike within `tolerance` samples of each such event, from a unit whose
  peak electrode lies in the trigger electrode's block (within `block_um`). The online detector is a simple
  threshold, so the recall is a relative measure (between sortings of the same data), not an absolute one.
- Per unit: Kilosort label, spike count, refractory violations (< refractory_ms), presence (share of the
  configurations with at least min_spikes_per_config spikes; the common electrodes are recorded in all of them).

Usage: python evaluate_axon_scan_sorting.py <sorter_output folder> <rec_path> <well>
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parents[1]))
from axon_tracking import spike_sorting as ss  # noqa: E402


def trigger_events(rec_path, well, rec_names):
    """Online spikes on trigger channels that lie in stored data: (concatenated sample index, electrode),
    and the stored samples per configuration (concatenation order: rec_names)."""
    times, electrodes, n_stored = [], [], []
    offset = 0
    with h5py.File(rec_path, "r") as f:
        for rn in rec_names:
            r = f["wells"][well][rn]
            g = r["groups/routed"]
            frames = g["frame_nos"][:].astype(np.int64)
            m = r["settings/mapping"][:]
            ch2el = dict(zip(m["channel"].tolist(), m["electrode"].tolist()))
            sp = r["spikes"][:]
            on = np.isin(sp["channel"], g["trigger_channels"][:])
            fr = sp["frameno"][on].astype(np.int64)
            idx = np.searchsorted(frames, fr)
            stored = (idx < len(frames)) & (frames[np.minimum(idx, len(frames) - 1)] == fr)
            times.append(offset + idx[stored])
            electrodes.append(np.array([ch2el[int(c)] for c in sp["channel"][on][stored]], int))
            n_stored.append(len(frames))
            offset += len(frames)
    return np.concatenate(times), np.concatenate(electrodes), np.array(n_stored)


def evaluate(sorter_output, rec_path, well, tolerance=3, block_um=25.0, refractory_ms=1.5,
             min_spikes_per_config=5):
    sorter_output = Path(sorter_output)
    rec_names, common_el, _ = ss.find_common_electrodes(rec_path, well)
    st = np.load(sorter_output / "spike_times.npy").ravel().astype(np.int64)
    sc = np.load(sorter_output / "spike_clusters.npy").ravel()
    templates = np.load(sorter_output / "templates.npy")  # (clusters, nt, channels)
    chan_map = np.load(sorter_output / "channel_map.npy").ravel()
    labels = {}
    label_file = sorter_output / "cluster_KSLabel.tsv"
    if label_file.exists():
        lab = pd.read_csv(label_file, sep="\t")
        labels = dict(zip(lab["cluster_id"], lab["KSLabel"]))
    units = np.unique(sc)
    peak_ch = {u: chan_map[np.argmax(np.ptp(templates[u], axis=0))] for u in units if u < len(templates)}
    peak_el = {u: common_el[c] for u, c in peak_ch.items()}
    xy = lambda e: np.column_stack([np.asarray(e) % 220, np.asarray(e) // 220]) * 17.5

    trig_t, trig_el, n_stored = trigger_events(rec_path, well, rec_names)
    bounds = np.r_[0, np.cumsum(n_stored)]
    # trigger recall: a sorted spike near the event from a unit with its peak in the trigger's block
    unit_xy = {u: xy([e])[0] for u, e in peak_el.items()}
    matched = np.zeros(len(trig_t), bool)
    order = np.argsort(st)
    st_s, sc_s = st[order], sc[order]
    lo, hi = np.searchsorted(st_s, trig_t - tolerance), np.searchsorted(st_s, trig_t + tolerance, side="right")
    txy = xy(trig_el)
    for k in np.flatnonzero(hi > lo):
        for u in np.unique(sc_s[lo[k] : hi[k]]):
            if u in unit_xy and np.linalg.norm(unit_xy[u] - txy[k]) <= block_um:
                matched[k] = True
                break
    rows = []
    for u in units:
        t = np.sort(st[sc == u])
        per_config = np.histogram(t, bins=bounds)[0]
        rows.append(dict(unit=int(u), label=labels.get(u, ""), n_spikes=len(t),
                         refractory_violations=float(np.mean(np.diff(t) < refractory_ms * 10)) if len(t) > 1 else 0.0,
                         presence=float(np.mean(per_config >= min_spikes_per_config)),
                         peak_electrode=int(peak_el.get(u, -1)),
                         amplitude=float(np.ptp(templates[u], axis=0).max()) if u < len(templates) else np.nan))
    df = pd.DataFrame(rows)
    blocks = np.unique(trig_el)
    per_block = {int(e): float(matched[trig_el == e].mean()) for e in blocks}
    unit_block = {u: blocks[np.argmin(np.linalg.norm(xy(blocks) - unit_xy[u], axis=1))] for u in unit_xy}
    good = df[(df.refractory_violations < 0.01) & (df.presence >= 0.9)]
    summary = dict(
        units=len(df), ks_good=int((df.label == "good").sum()),
        clean_units=len(good),  # < 1% refractory violations and present in >= 90% of the configurations
        blocks_with_clean_unit=len({unit_block[u] for u in good.unit if u in unit_block}),
        trigger_events=len(trig_t), trigger_recall=float(matched.mean()),
        trigger_recall_block_median=float(np.median(list(per_block.values()))),
        spikes=len(st))
    return summary, df, per_block


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    out, rec_path, well = sys.argv[1:4]
    summary, df, per_block = evaluate(out, rec_path, well)
    print(summary)
    df.to_csv(os.path.join(out, "..", "unit_checks.csv"), index=False)
