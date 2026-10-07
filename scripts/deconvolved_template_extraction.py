"""Deconvolved (and, for comparison, averaged) templates of axon scans, written outside the sorting folders.

Same QC as scripts/template_extraction.py. The raw data paths stored by the sorting (Linux mount) are remapped
with PATH_MAP when running elsewhere (e.g. Windows). Output: OUTPUT_ROOT/<path below SORTING_ROOT>/templates,
.../templates_deconvolved and .../templates_deconvolved_null (null templates from windows in which each unit was
silent, for data-driven thresholds: graph_reconstruction.reconstruct(null_grid=...)). Axon scans store only
snippets around trigger events; the deconvolution runs on the stored samples as concatenated (see
te_params["triggered"]), the frame numbers of the raw file are used to place the null windows.
"""

import os
import sys
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from axon_tracking import template_extraction as te

SORTING_ROOT = "S:/group/hierlemann02/intermediate_data/Maxtwo/phornauer/"
PATH_PATTERN = ["Torsten_2", "241021", "*", "AxonTracking", "well*", "sorter_output"]
OUTPUT_ROOT = "E:/Data/axon_tracking_templates/"
# raw data path in spikeinterface_recording.json -> path on this machine (None: use as stored)
PATH_MAP = {"/net/bs-filesvr02/export/group/hierlemann": "S:/group/hierlemann02"}
COMPARE_WITH_AVERAGE = True  # also extract averaged templates with the same QC
# network recording of the same wells (its units not sorted in the axon scan become extra regressors); None: off
NETWORK_PATTERN = ["Torsten_2", "241023", "*", "Network", "{well}", "sorter_output"]

te_params = dict(
    n_jobs=8,
    filter_band=150,
    overwrite=False,
    max_spikes_per_unit=1000,
    deconvolve=True,
    deconvolve_units="all",  # also the units rejected by the QC are regressors
    deconvolve_pad_ms=(0.5, 2.0),
    deconvolve_max_unknowns=15000,
    triggered=False,  # "auto": fit on the real time axis if the file stores trigger snippets
    null_templates=True,
)
qc_params = dict(min_n_spikes=500, exclude_mua=True, use_bc=False, use_si=True, auto_merge=True,
                 remove_redundant=True)

sorting_list = sorted(glob(os.path.join(SORTING_ROOT, *PATH_PATTERN)))
print(f"Found {len(sorting_list)} sortings")
relative = [os.path.relpath(os.path.dirname(s), SORTING_ROOT) for s in sorting_list]


def network_sorting(sorting_path):
    if NETWORK_PATTERN is None:
        return None
    well = Path(sorting_path).parent.name
    found = glob(os.path.join(SORTING_ROOT, *[p.format(well=well) for p in NETWORK_PATTERN]))
    return found[0] if len(found) == 1 else None


networks = [network_sorting(s) for s in sorting_list]
print(f"Network sortings found for {sum(n is not None for n in networks)} of {len(sorting_list)} wells")
te.extract_templates_from_sorting_list(
    sorting_list, qc_params, dict(te_params, verbose=True), path_map=PATH_MAP, network_sortings=networks,
    save_folders=[os.path.join(OUTPUT_ROOT, r, "templates_deconvolved") for r in relative])
if COMPARE_WITH_AVERAGE:
    te.extract_templates_from_sorting_list(
        sorting_list, qc_params, dict(te_params, deconvolve=False), path_map=PATH_MAP,
        save_folders=[os.path.join(OUTPUT_ROOT, r, "templates") for r in relative])
