"""Averaged vs deconvolved templates of the same units (output of deconvolved_template_extraction.py).

Per template: number of other units' somata (> min_distance_um away) whose peak-to-peak in the template exceeds
k_sigma times the template noise (foreign somata, e.g. from network bursts), and the part3 reconstruction.
Usage: compare_deconvolved_templates.py <OUTPUT_ROOT of deconvolved_template_extraction.py>
"""

import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from axon_tracking import path_evidence as pe
from axon_tracking.simulation import scenarios

PITCH = 17.5


def foreign_somata(grid, own_soma, somata, k_sigma=10.0, min_distance_um=100.0):
    noise = np.median(pe.template_noise_sd(grid)[np.abs(grid).max(axis=2) > 0])
    p2p = np.ptp(grid, axis=2)
    far = [s for s in somata if np.linalg.norm((np.array(s) - own_soma) * PITCH) > min_distance_um]
    return int(sum(p2p[s] > k_sigma * noise for s in far)), len(far)


def compare_well(well_folder, reconstruct=True):
    avg_files = sorted(glob(os.path.join(well_folder, "templates", "*.npy")))
    rows = []
    templates = {Path(f).stem: (np.load(f), np.load(os.path.join(well_folder, "templates_deconvolved", Path(f).name)))
                 for f in avg_files if os.path.exists(os.path.join(well_folder, "templates_deconvolved", Path(f).name))}
    somata = {u: np.unravel_index(np.argmax(np.ptp(a, axis=2)), a.shape[:2]) for u, (a, _) in templates.items()}
    for u, (avg, dec) in templates.items():
        row = dict(well=Path(well_folder).name, unit=u)
        for label, grid in (("average", avg), ("deconvolved", dec)):
            n, n_far = foreign_somata(grid, np.array(somata[u]), [s for v, s in somata.items() if v != u])
            row[f"{label}_foreign_somata"] = n
            row["other_units"] = n_far
            if reconstruct:
                paths, _ = scenarios.reconstruct_part3(grid)
                row[f"{label}_cable_um"] = sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() for p in paths)
        rows.append(row)
    return rows


if __name__ == "__main__":
    root = sys.argv[1]
    wells = sorted({str(Path(f).parent.parent) for f in glob(os.path.join(root, "**", "templates_deconvolved", "*.npy"),
                                                                recursive=True)})
    df = pd.DataFrame([r for w in wells for r in compare_well(w)])
    df.to_csv(os.path.join(root, "comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("well")[[c for c in df.columns if c not in ("well", "unit")]].median().to_string())
    print(df[[c for c in df.columns if c not in ("well", "unit")]].median().to_string())
