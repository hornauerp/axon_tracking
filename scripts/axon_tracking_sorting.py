import os, sys
from glob import glob
import spikeinterface.full as si

sys.path.append("/home/phornauer/Git/axon_tracking/")
from axon_tracking import spike_sorting as ss

sorter = "kilosort2_5"
si.Kilosort2_5Sorter.set_kilosort2_5_path(
    "/home/phornauer/Git/Kilosort_2020b"
)  # Change
sorter_params = si.get_default_sorter_params(si.Kilosort2_5Sorter)
sorter_params["n_jobs"] = -1
sorter_params["detect_threshold"] = 5.5
sorter_params["minFR"] = 0.01
sorter_params["minfr_goodchannels"] = 0.01
sorter_params["keep_good_only"] = False
sorter_params["do_correction"] = False

root_path = "/net/bs-filesvr02/export/group/hierlemann/intermediate_data/Maxtwo/mpascual/"  # Fixed path root that all recordings have in common
path_pattern = [
    "iNeurons",
    "iNeurons_II",
    "250212",
    "*0*",
    "AxonTracking",
    "000*",
]  # Variable part of the path, where we collect all possible combinations using wildcards (*). It is still recommended to be as specific as possible to avoid ambiguities.
# path_pattern = ["25010*", "EI_iNeurons", "*0*","AxonTracking","000*"] # Variable part of the path, where we collect all possible combinations using wildcards (*). It is still recommended to be as specific as possible to avoid ambiguities.
file_name = "data.raw.h5"  # File name of the recording

full_path = os.path.join(root_path, *path_pattern, file_name)
path_list = glob(full_path)
print(
    f"Found {len(path_list)} recording paths matching the description:\n{full_path}\n"
)

save_path_changes = {
    "pos": [0, 6, 8, 9, 10, 14, 15],
    "vals": ["/", "intermediate_data", "phornauer/iNeurons_2", "", "", "", ""],
}
# save_path_changes = {'pos': [0, 6, 8, 9, 11, 15, 16], 'vals': ['/', 'intermediate_data', 'phornauer','', '', '','']}

save_path = ss.convert_rec_path_to_save_path(full_path, save_path_changes)

print(f"The save path corresponds to the pattern:\n {save_path}\n")

save_flag = input("Do you want to continue and save the files? (y/n) ")

if save_flag != "y":
    sys.exit()

ss.sort_recording_list(path_list, save_path_changes, sorter, sorter_params)
