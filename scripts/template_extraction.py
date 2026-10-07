import os
import sys
from glob import glob

sys.path.append("/home/phornauer/Git/axon_tracking/")
from axon_tracking import template_extraction as te

te_params = dict()
te_params["n_jobs"] = 64  # Number of cores to use for waveform extraction
te_params["filter_band"] = (
    150  # Either float for the highpass filter frequency or list for the bandpass filter frequencies
)
te_params["overwrite"] = (
    True  # Flag if templates should be recalculated if already existing
)
te_params["max_spikes_per_unit"] = (
    1000  # Maximum number of spikes to be used for template extraction
)

qc_params = dict()
qc_params["min_n_spikes"] = (
    500  # Minimum number of spikes to be detected for a unit for template extraction to take place
)
qc_params["exclude_mua"] = (
    True  # Exclude units that were labelled multi unit activity by kilosort
)
qc_params["use_bc"] = False  # Use bombcell for QC
qc_params["use_si"] = True  # Use spikeinterface for QC
qc_params["auto_merge"] = (
    True  # Automatically merge units (spikeinterface implementation)
)
qc_params["remove_redundant"] = (
    True  # Remove redundant units (spikeinterface implementation)
)

# root_path = "/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Maxtwo/phornauer/" # Fixed path root that all recordings have in common
# path_pattern = ["EI_iNeurons", "250108",  "*","AxonTracking","w*","sorter_output"] # Variable part of the path, where we collect all possible combinations using wildcards (*). It is still recommended to be as specific as possible to avoid ambiguities.
root_path = "/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Maxtwo/phornauer/"  # Fixed path root that all recordings have in common
path_pattern = [
    "iNeurons_2",
    "250212",
    "*",
    "AxonTracking",
    "w*",
    "sorter_output",
]  # Variable part of the path, where we collect all possible combinations using wildcards (*). It is still recommended to be as specific as possible to avoid ambiguities.

full_path = os.path.join(root_path, *path_pattern)
sorting_list = glob(full_path)
sorting_list.sort()
print(
    f"Found {len(sorting_list)} sorting paths matching the description:\n{full_path}\n"
)

te.extract_templates_from_sorting_list(sorting_list, qc_params, te_params)
