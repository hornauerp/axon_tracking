import os, sys
from glob import glob

sys.path.append("/home/phornauer/Git/axon_tracking/")
from axon_tracking import spike_sorting as ss

root_path = "/net/bs-filesvr02/export/group/hierlemann/recordings/Maxtwo/phornauer/"  # Fixed path root that all recordings have in common
path_pattern = [
    "241128",
    "*0*",
    "AxonTracking",
    "000*",
]  # Variable part of the path, where we collect all possible combinations using wildcards (*). It is still recommended to be as specific as possible to avoid ambiguities.
file_name = "data.raw.h5"  # File name of the recording

full_path = os.path.join(root_path, *path_pattern, file_name)
path_list = glob(full_path)
print(
    f"Found {len(path_list)} recording paths matching the description:\n{full_path}\n"
)

save_path_changes = {
    "pos": [0, 6, 8, 12, 13],
    "vals": ["/", "intermediate_data", "phornauer/EI_iNeurons", "", ""],
}

save_path = ss.convert_rec_path_to_save_path(full_path, save_path_changes)

print(f"The save path corresponds to the pattern:\n {save_path}\n")

save_flag = input("Do you want to continue and save the files? (y/n) ")

if save_flag != "y":
    exit
