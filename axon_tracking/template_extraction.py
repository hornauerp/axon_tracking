"""
This module provides functions for extracting templates from spike sorting data,
performing quality control, and organizing the extracted templates into a grid format.
Functions:
    extract_templates_from_sorting_list(sorting_list, qc_params={}, te_params={}):
        Performs template extraction from a list of sorting paths.
        Does not require a recording path.
    preprocess_sorting(sorting_path, qc_params={}):
        Performs QC and splitting of a sorting object.
    select_good_units(sorting, min_n_spikes=1500, exclude_mua=True, use_bc=False):
        Selects good units based on quality control parameters from KS and BC.
    extract_all_templates(stream_id, segment_sorting, save_root, pos, te_params):
        Extracts templates from all units in a sorting object.
    find_files(save_root, file_name="templates.npy", folder_name="sorter_output"):
        Finds files with a specific name in a directory tree.
    find_successful_sortings(path_list, save_path_changes):
        Finds successful sorting paths based on the presence of specific files.
    convert_to_grid(template_matrix, pos):
        Converts a template matrix into a grid format based on electrode positions.
"""

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import spikeinterface.full as si
from axon_tracking import spike_sorting as ss
from axon_tracking import utils as ut


def extract_templates_from_sorting_list(sorting_list, qc_params={}, te_params={}):
    """Performs template extraction from a list of sorting paths. Does not require a recording path.

    Args:
        sorting_list (list): Sorting path list with sorter_output suffix.
        qc_params (dict, optional): Dict of quality control parameters. Defaults to {}.
        te_params (dict, optional): Dict of template extraction parameters. Defaults to {}.
    """
    si.set_global_job_kwargs(n_jobs=te_params["n_jobs"], progress_bar=False)
    for sorting_path in tqdm(sorting_list):
        if os.path.isdir(
            os.path.join(sorting_path, "templates")
        ):  # Check if output folder exists
            _, _, files = next(os.walk(os.path.join(sorting_path, "templates")))
            file_count = len(files)  # Check for existing output in the folder
            if file_count > 0 and not te_params["overwrite"]:
                print(f"Templates already extracted for {sorting_path}")
                continue
        try:
            segment_sorting = preprocess_sorting(sorting_path, qc_params)
            template_matrix, pos = extract_all_templates(segment_sorting, te_params)
            unit_ids = segment_sorting.get_unit_ids()
            save_folder = os.path.join(sorting_path, "templates")
            os.makedirs(save_folder, exist_ok=True)
            save_individual_templates(template_matrix, save_folder, unit_ids, pos)

        except Exception as e:
            print(f"Error: {e}")
            continue


def preprocess_sorting(sorting_path, qc_params):
    """Performs QC and splitting of a sorting object.

    Args:
        sorting_path (str): Path to the folder containing the sorting output.
        qc_params (dict, optional): QC parameter dictionary.

    Returns:
        segment_sorting (SegmentSorting): Sorting that was split according to the
        corresponding (axon tracking) recording.
    """
    # Load sorting
    sorting = si.KiloSortSortingExtractor(sorting_path)

    # Find recording path
    json_path = os.path.join(
        Path(sorting_path).parent.absolute(), "spikeinterface_recording.json"
    )

    # Load concatenated recording to use for splitting the sorting
    multirecording = si.load_extractor(json_path, base_folder=True)

    # Clean sorting (perform quality control)
    cleaned_sorting = select_good_units(sorting, multirecording, qc_params)
    cleaned_sorting = si.remove_excess_spikes(
        cleaned_sorting, multirecording
    )  # Relevant if last spike time == recording_length
    cleaned_sorting.register_recording(multirecording)

    # Split sorting into segments if it is a ConcatenateSegmentRecording
    if isinstance(multirecording, si.ConcatenateSegmentRecording):
        cleaned_sorting = si.SplitSegmentSorting(cleaned_sorting, multirecording)

    return cleaned_sorting


def perform_si_qc(sorting, recording, qc_params):
    """Performs quality control on a sorting object using spikeinterface metrics.

    Args:
        sorting (BaseSorting): Sorting extractor object.
        qc_params (dict): Dict of quality control parameters.

    Returns:
        BaseSorting: Sorting object with units that passed the QC.
    """
    full_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)

    full_analyzer.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "spike_amplitudes",
            "unit_locations",
            "template_similarity",
            "correlograms",
        ]
    )

    if qc_params["auto_merge"]:
        merge_unit_groups = si.get_potential_auto_merge(
            full_analyzer, resolve_graph=True
        )
        full_analyzer = full_analyzer.merge_units(merge_unit_groups=merge_unit_groups)

    if qc_params["remove_redundant"]:
        sorting = si.remove_redundant_units(
            full_analyzer, duplicate_threshold=0.8, remove_strategy="minimum_shift"
        )

    return sorting


def select_good_units(sorting, recording, qc_params):
    """Selects good units based on quality control parameters from KS and BC.

    Args:
        sorting (BaseSorting): Sorting extractor object.
        min_n_spikes (int, optional): Minimum number of spikes required for a unit to be considered good. Defaults to 1500.
        exclude_mua (bool, optional): Flag on whether to exclude MUA as labeled by KS. Defaults to True.
        use_bc (bool, optional): Flag on whether to consider bombcell QC output. Defaults to False.

    Returns:
        UnitSelectionSorting: Sorting containing units that passed the QC.
    """
    # Kilosort QC
    if qc_params["exclude_mua"]:
        ks_idx = sorting.get_property("KSLabel") == "good"
    else:
        ks_idx = np.full((sorting.get_num_units(),), True, dtype="bool")

    # Bombcell QC (if present)
    if qc_params["use_bc"] and len(sorting.get_property("bc_unitType")) > 0:
        bc_idx = sorting.get_property("bc_unitType") == "GOOD"
    else:
        # print('No bombcell output found')
        bc_idx = np.full((sorting.get_num_units(),), True, dtype="bool")

    # Minimum number of spikes QC
    n_spikes = [
        len(sorting.get_unit_spike_train(x, segment_index=0))
        for x in sorting.get_unit_ids()
    ]
    good_n_spikes_idx = np.array(n_spikes) > qc_params["min_n_spikes"]

    # Combine QC
    good_idx = ks_idx & bc_idx & good_n_spikes_idx
    good_ids = sorting.get_unit_ids()[good_idx]
    cleaned_sorting = sorting.select_units(good_ids)

    if qc_params["use_si"]:
        cleaned_sorting = perform_si_qc(cleaned_sorting, recording, qc_params)

    print(f"Keeping {len(good_ids)} good units")

    return cleaned_sorting


def extract_all_templates(segment_sorting, te_params):
    """Extracts templates from all units in a sorting object.

    Args:
        stream_id (str): Stream ID of the recording.
        segment_sorting (BaseSorting): SegmentSorting object.
        save_root (str): Root path for saving the templates.
        pos (dict): Dict containing electrode positions.
        te_params (dict): Dict of template extraction parameters.
    """

    # Find recording path
    full_path = ss.get_recording_path(segment_sorting)
    # Find cutout for waveform extraction
    cutout_samples, cutout_ms = ut.get_cutout_info(full_path)
    # Find out which well this belongs to
    stream_id = ut.get_sorting_stream_id(segment_sorting)

    # Find electrode positions
    rec_names, _, pos = ss.find_common_electrodes(full_path, stream_id)
    n_units = segment_sorting.get_num_units()
    # Initialize template matrix
    template_matrix = np.full([n_units, sum(cutout_samples), 26400], np.nan)

    # Extract templates (main loop)
    for sel_idx, rec_name in enumerate(rec_names):
        rec = si.MaxwellRecordingExtractor(
            full_path, stream_id=stream_id, rec_name=rec_name
        )
        seg_sort = si.SelectSegmentSorting(segment_sorting, sel_idx)
        tmp_data = extract_templates(rec, seg_sort, te_params, cutout_ms)

        els = rec.get_property("contact_vector")["electrode"]
        template_matrix[:, :, els] = tmp_data

    return template_matrix, pos


def extract_templates(recording, sorting, te_params, cutout_ms):
    """
    Extracts templates from a given recording and corresponding sorting.
    Args:
    recording : si.RecordingExtractor
        The recording extractor object containing the neural recordings.
    sorting : si.SortingExtractor
        The sorting extractor object containing the spike sorting results.
    te_params : dict
        A dictionary containing template extraction parameters:
        - 'freq_min': Minimum frequency for high-pass filtering.
        - 'overwrite': Boolean indicating whether to overwrite existing data.
        - 'max_spikes_per_unit': Maximum number of spikes per unit for spike extraction.
    cutout_ms : tuple
        A tuple containing two elements:
        - ms_before: Time in milliseconds before the spike to include in the waveform.
        - ms_after: Time in milliseconds after the spike to include in the waveform.
    Returns:
    np.ndarray
        A numpy array containing the extracted templates.
    """

    # Filter recording
    if isinstance(te_params["filter_band"], list):
        btype = "bandpass"
    else:
        btype = "highpass"

    rec_centered = si.filter(recording, btype=btype, band=te_params["filter_band"])

    sorting = si.remove_excess_spikes(sorting, rec_centered)
    sorting.register_recording(rec_centered)

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec_centered,
        sparse=False,
    )

    analyzer.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
        ],
        extension_params={
            "random_spikes": {"max_spikes_per_unit": te_params["max_spikes_per_unit"]},
            "waveforms": {"ms_before": cutout_ms[0], "ms_after": cutout_ms[1]},
        },
    )

    tmp = analyzer.get_extension(extension_name="templates")
    tmp_data = tmp.get_data()
    return tmp_data


def convert_to_grid(template_matrix, pos):
    """
    Converts a template matrix into a grid based on provided positions.
    Parameters:
    template_matrix (numpy.ndarray): The template matrix to be converted.
    pos (dict): A dictionary containing 'x' and 'y' positions. The positions
                should be numpy arrays with the same length as the number of columns
                in the template_matrix.
    Returns:
    numpy.ndarray: A 3D grid where the first two dimensions correspond to the
                   x and y indices derived from the positions, and the third
                   dimension corresponds to the voltage over time.
    """

    # Clean up template matrix and positions to remove NaNs from sparse recordings
    clean_template = np.delete(template_matrix, np.isnan(pos["x"]), axis=1)
    clean_x = pos["x"][~np.isnan(pos["x"])]
    clean_y = pos["y"][~np.isnan(pos["y"])]

    # Convert positions to the grid
    x_idx = np.int16(clean_x / 17.5)
    y_idx = np.int16(clean_y / 17.5)
    grid = np.full(
        [np.max(x_idx) + 1, np.max(y_idx) + 1, clean_template.shape[0]], 0
    ).astype("float32")
    for i, _ in enumerate(y_idx):
        grid[x_idx[i], y_idx[i], :] = clean_template[:, i]

    return grid


def save_individual_templates(template_matrix, save_folder, unit_ids, pos):
    """
    Saves the template matrix to a specified path.
    Parameters:
    template_matrix (numpy.ndarray): The template matrix to be saved.
    save_path (str): The path to save the templates to.
    """

    for i, id in enumerate(unit_ids):
        save_path_i = os.path.join(save_folder, f"{id}.npy")
        tmp = np.squeeze(template_matrix[i, :, :])
        grid = convert_to_grid(tmp, pos)
        np.save(save_path_i, grid)
