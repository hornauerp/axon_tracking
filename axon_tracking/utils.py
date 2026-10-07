import os
from pathlib import Path

import h5py
import numpy as np

# MaxOne/MaxTwo electrode grid
N_ELECTRODES = 26400
N_COLS, N_ROWS = 220, 120
EL_SPACING = 17.5  # um


def infer_stream_id(sorting_path):
    """
    Infer the stream ID from the sorting path.

    Args:
        sorting_path (str): Path to the sorting directory.

    Returns:
        stream_id (str): Stream ID.
    """
    well_parts = [p for p in Path(sorting_path).parts if p.startswith("well")]
    if not well_parts:
        raise ValueError(f"No well ID (e.g. 'well000') found in {sorting_path}")

    return well_parts[0]


def remap_path(path, path_map=None):
    """Replace the first matching prefix of path_map {old_prefix: new_prefix} (e.g. a Linux mount by the
    corresponding Windows drive); separators are normalised to '/' for the comparison."""
    if not path_map:
        return path
    p = str(path).replace("\\", "/")
    for old, new in path_map.items():
        old = str(old).replace("\\", "/").rstrip("/")
        if p == old or p.startswith(old + "/"):
            return str(new).replace("\\", "/").rstrip("/") + p[len(old):]
    return path


def load_recording_json(json_path, path_map=None):
    """Load a recording saved with dump_to_json (e.g. spikeinterface_recording.json of a sorting), with the
    file paths of its source recordings remapped by path_map (see remap_path)."""
    import json

    from spikeinterface.core.base import BaseExtractor

    with open(json_path) as f:
        d = json.load(f)

    def walk(o):
        if isinstance(o, dict):
            return {k: (remap_path(v, path_map) if k == "file_path" and isinstance(v, str) else walk(v))
                    for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    d = walk(d)
    base_folder = Path(json_path).parent if d.get("relative_paths") else None
    return BaseExtractor.from_dict(d, base_folder=base_folder)


def find_source_recording(sort_or_rec):
    """
    Walk back through the processing chain of a spikeinterface object (sorting or recording)
    until the source recording that was read from file (e.g. the MaxwellRecordingExtractor).

    Args:
        sort_or_rec (BaseExtractor): Sorting or recording object.

    Returns:
        BaseRecording: Source recording, whose _kwargs contain "file_path" (and "stream_id" for MaxWell files).
    """
    from spikeinterface.core.base import BaseExtractor

    queue, seen = [sort_or_rec], set()
    while queue:
        extractor = queue.pop(0)
        if id(extractor) in seen:
            continue
        seen.add(id(extractor))

        kwargs = getattr(extractor, "_kwargs", None) or {}
        if "file_path" in kwargs:
            return extractor

        parents = []
        registered_recording = getattr(extractor, "_recording", None)  # recording registered to a sorting
        if registered_recording is not None:
            parents.append(registered_recording)
        for value in kwargs.values():
            if isinstance(value, BaseExtractor):
                parents.append(value)
            elif isinstance(value, (list, tuple)):
                parents.extend(v for v in value if isinstance(v, BaseExtractor))
        queue.extend(parents)

    raise ValueError("Could not find the source recording (with a 'file_path') of this object.")


def get_sorting_stream_id(sorting):
    """
    Get the stream ID of the sorting object.

    Args:
        sorting (SortingExtractor): Sorting object with a registered recording.

    Returns:
        stream_id (str): Stream ID.
    """
    return find_source_recording(sorting)._kwargs["stream_id"]


def get_cutout_info(rec_path, stream_id=None):
    """Extracts pre and post trigger cutout samples and their corresponding times in
    milliseconds from a given recording file.

    Args:
        rec_path (str): Path to the HDF5 recording file.
        stream_id (str, optional): Well to read the trigger settings from first. Other wells are used as
            fallback, since some failed axon trackings have non-positive trigger values.

    Returns:
        tuple: A tuple containing:
            - cutout_samples (list): List of pre and post trigger cutout samples.
            - cutout_ms (list): List of pre and post trigger cutout times in milliseconds.
    """
    pre, post, sampling_rate = -1, -1, None
    with h5py.File(rec_path, "r") as h5:
        well_names = list(h5["wells"].keys())
        if stream_id in well_names:
            well_names.remove(stream_id)
            well_names.insert(0, stream_id)

        for well_name in well_names:
            for rec_name in h5["wells"][well_name].keys():
                rec = h5["wells"][well_name][rec_name]
                sampling_rate = rec["settings"]["sampling"][0]
                routed = rec["groups"].get("routed")
                if routed is None or "trigger_pre" not in routed or "trigger_post" not in routed:
                    continue
                pre, post = int(routed["trigger_pre"][0]), int(routed["trigger_post"][0])
                if pre > 0 and post > 0:
                    break
            if pre > 0 and post > 0:
                break

    if sampling_rate is None:
        raise ValueError(f"No recordings found in {rec_path}")

    if pre > 0 and post > 0:
        cutout_samples = [pre, post]
        cutout_ms = [x / (sampling_rate / 1000) for x in cutout_samples]  # convert cutout to ms
    else:
        # Workaround to accomodate waveform extraction from network recordings
        print("No valid trigger cutout found (network recording?), using default [1.5, 5] ms")
        cutout_ms = [1.5, 5.0]
        cutout_samples = [int(x * sampling_rate / 1000) for x in cutout_ms]

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
    from axon_tracking import spike_sorting as ss  # local import: spike_sorting imports this module

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

    Heuristic: in (upsampled) electrode units, coordinates cannot exceed the size of the electrode
    grid, so any coordinate beyond it means the paths are in um. Paths in um that lie entirely
    within the top-left corner of the chip cannot be told apart; pass the scale explicitly to
    `convert_coor_scale` in that case.

    Args:
        path_list (list of np.ndarrays): Coordinates to check.
        params (dict): Parameters dictionary containing the upsampling factors.

    Returns:
        bool: True if the coordinates are scaled, False otherwise.
    """
    coors = np.concatenate(path_list)[:, :2]  # Concatenate all coordinates

    return (
        np.max(coors[:, 0]) > N_COLS / params["upsample"][0]
        or np.max(coors[:, 1]) > N_ROWS / params["upsample"][1]
    )


def convert_coor_scale(path_list, params, scale="um", current_scale=None):
    """
    Converts the coordinates to a different scale ('um' or 'el').

    Args:
        path_list (list of np.ndarrays): Coordinates to convert.
        params (dict): Parameters dictionary containing the electrode spacing.
        scale (str): Scale to convert to. Can be "um" or "el".
        current_scale (str, optional): Scale the coordinates are currently in ("um" or "el").
            If None, it is guessed with `check_if_scaled`.

    Returns:
        list of np.ndarrays: List of converted coordinates.
    """
    if scale not in ("um", "el"):
        raise ValueError("Invalid scale value. Must be 'um' or 'el'.")
    if len(path_list) == 0:
        return path_list
    if current_scale is None:
        current_scale = "um" if check_if_scaled(path_list, params) else "el"
    if current_scale == scale:
        return path_list

    scale_factor = params["el_spacing"] if scale == "um" else 1 / params["el_spacing"]
    rescaled = [
        np.concatenate((path[:, :2] * scale_factor, path[:, 2:]), axis=1)
        for path in path_list
    ]  # Rescale the coordinates without changing the z values
    return rescaled
