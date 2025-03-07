import os

import h5py
import numpy as np


def infer_stream_id(sorting_path):
    """
    Infer the stream ID from the sorting path.

    Args:
        sorting_path (str): Path to the sorting directory.

    Returns:
        stream_id (str): Stream ID.
    """
    stream_id = [p for p in sorting_path.split("/") if p.startswith("well")][
        0
    ]  # Find well ID

    return stream_id


def get_sorting_stream_id(sorting):
    """
    Get the stream ID of the sorting object.

    Args:
        sorting (SortingExtractor): Sorting object.

    Returns:
        stream_id (str): Stream ID.
    """
    try:
        stream_id = sorting._kwargs["recording_or_recording_list"][
            0
        ]._parent_recording._parent_recording.stream_id
    except:
        stream_id = sorting._recording._kwargs["recording_list"][
            0
        ]._parent._parent.stream_id

    return stream_id


def get_cutout_info(rec_path):
    """Extracts pre and post trigger cutout samples and their corresponding times in
    milliseconds from a given recording file.

    Args:
        rec_path (str): Path to the HDF5 recording file.

    Returns:
        tuple: A tuple containing:
            - cutout_samples (list): List of pre and post trigger cutout samples.
            - cutout_ms (list): List of pre and post trigger cutout times in milliseconds.
    """
    h5 = h5py.File(rec_path)
    pre, post, well_id = -1, -1, 0
    while (
        pre <= 0 or post <= 0
    ):  # some failed axon trackings give negative trigger_post values, so we try different wells
        well_name = list(h5["wells"].keys())[well_id]
        rec_name = list(h5["wells"][well_name].keys())[well_id]
        sampling_rate = h5["wells"][well_name][rec_name]["settings"]["sampling"][0]
        try:
            pre = h5["wells"][well_name][rec_name]["groups"]["routed"]["trigger_pre"][0]
            post = h5["wells"][well_name][rec_name]["groups"]["routed"]["trigger_post"][
                0
            ]
        except:
            break
        well_id += 1

    cutout_samples = [pre, post]

    # Workaround to accomodate waveform extraction from network recordings
    if cutout_samples[0] < 0:
        print("Network recording detected, using default [1.5, 5]")
        cutout_ms = np.array([1.5, 5])
        cutout_samples = (cutout_ms * (sampling_rate / 1000)).astype(int)
    else:
        cutout_ms = [
            x / (sampling_rate / 1000) for x in cutout_samples
        ]  # convert cutout to ms

    return cutout_samples, cutout_ms


def find_successful_sortings(path_list, save_path_changes):
    """
    Finds and returns a dictionary of successful sorting paths for given recording paths.

    This function traverses through directories specified in `path_list`, applies changes to the paths
    using `save_path_changes`, and identifies directories containing the "templates.npy" file, which
    indicates a successful sorting.

    Args:
        path_list (list of str): List of recording paths to search for sorting results.
        save_path_changes (dict): Dictionary containing changes to be applied to recording paths to
                                  convert them to save paths.

    Returns:
        dict: A dictionary where keys are the original recording paths from `path_list` and values are
              lists of paths to directories containing the "templates.npy" file.
    """

    sorting_dict = dict()
    for rec_path in path_list:
        save_root = ss.convert_rec_path_to_save_path(rec_path, save_path_changes)

        # Takes into account different sorting folder names, subfolder depth, well IDs etc.
        sorting_files = [
            root
            for root, dirs, files in os.walk(save_root)
            for name in files
            if name == "templates.npy"
        ]
        sorting_dict[rec_path] = sorting_files

    return sorting_dict


def find_files(save_root, file_name="templates.npy", folder_name="sorter_output"):
    file_list = [
        root
        for root, dirs, files in os.walk(save_root)
        for dir in dirs
        if dir == folder_name
        and os.path.exists(os.path.join(root, folder_name, file_name))
    ]

    return file_list


def generate_max_template(template, peak="pos", absolute_value=False):
    """
    Generates a maximum template from a given template of shape (x*y*t).

    Args:
        template (np.ndarray): Template to generate the maximum template from.
        peak (str): Peak to use for the maximum template. Can be "pos" or "neg" or "both".
        absolute_value (bool): Whether to return the absolute value.

    Returns:
        np.ndarray: Maximum template of shape (x*y*1).
    """
    if peak == "pos":
        max_template = np.max(template, axis=2)
    elif peak == "neg":
        max_template = np.min(template, axis=2)
    elif peak == "both":
        max_template = np.max(template, axis=2)
        min_template = np.min(template, axis=2)
        diff_idx = np.where(np.abs(min_template) > max_template)
        max_template[diff_idx] = min_template[diff_idx]
    else:
        raise ValueError("Invalid peak value. Must be 'pos', 'neg' or 'both'.")

    if absolute_value:
        max_template = np.abs(max_template)

    return max_template


def check_if_scaled(path_list, params):
    """
    Checks if the coordinates are scaled to um.

    Args:
        path_list (list of np.ndarrays): Coordinates to check.

    Returns:
        bool: True if the coordinates are scaled, False otherwise.
    """
    coors = np.concatenate(path_list)[:, :2]  # Concatenate all coordinates

    # Check if all coordinates are multiples of the electrode spacing
    # return np.all(np.remainder(coors, params["el_spacing"]) == 0) # Does not work when paths are smoothed
    return (
        np.max(coors[:, 0]) > 220 / params["upsample"][0]
        and np.max(coors[:, 1]) > 220 / params["upsample"][1]
    )


def convert_coor_scale(path_list, params, scale="um"):
    """
    Converts the coordinates to a different scale ('um' or 'el').

    Args:
        path_list (list of np.ndarrays): Coordinates to convert.
        params (dict): Parameters dictionary containing the electrode spacing.
        scale (str): Scale to convert to. Can be "um" or "el".

    Returns:
        list of np.ndarrays: List of converted coordinates.
    """
    if scale == "um":
        if check_if_scaled(path_list, params):  # Check if already scaled to um
            return path_list
        scale_factor = params["el_spacing"]
    elif scale == "el":
        if not check_if_scaled(
            path_list, params
        ):  # Check if already in electrode spacing
            return path_list
        scale_factor = 1 / params["el_spacing"]
    else:
        raise ValueError("Invalid scale value. Must be 'um' or 'el'.")

    rescaled = [
        np.concatenate((path[:, :2] * scale_factor, path[:, 2:]), axis=1)
        for path in path_list
    ]  # Rescale the coordinates without changing the z values
    return rescaled
