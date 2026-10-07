import os
import shutil
import time
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import spikeinterface.full as si
from tqdm import tqdm

from axon_tracking import utils as ut


def sort_recording_list(
    path_list,
    save_path_changes,
    sorter,
    sorter_params=None,
    clear_files=True,
    verbose=True,
):
    """
    Function that iterates over a list of axon scans, finds common electrodes, concatenates and spike sorts the recording slices.

    Arguments
    ----------
    path_list: list
        List of string referring to the paths of axon scan recording.
    save_path_changes: dict
        Dictionary containing keys 'pos' and 'vals' that indicate the changes to be made to the rec_path.
        Refer to the inidices after splitting the path by '/'.
    sorter: str
        Name of the sorter, as per the spikeinterface convention (e.g. 'kilosort2_5')
    sorter_params: dict
        Dictionary containing parameters for the spike sorter. If left empty, will default to default parameters.
    clear_files: bool
        Flag if large temporary files should be deleted after the sorting. Default: True
    verbose: bool
        Default: True

    Returns
    ----------
    sorting_list: list of sorting objects
        Specific type depends on the sorter.

    """

    sorting_list = []

    for rec_path in tqdm(path_list, desc="Sorting recordings"):
        with h5py.File(rec_path, "r") as h5:
            rec_names_per_well = {
                stream_id: list(h5["wells"][stream_id].keys())
                for stream_id in h5["wells"].keys()
            }

        save_root = convert_rec_path_to_save_path(rec_path, save_path_changes)

        for stream_id, rec_names in tqdm(rec_names_per_well.items(), desc="Sorting wells"):
            sorter_output_file = os.path.join(
                save_root, stream_id, "sorter_output", "amplitudes.npy"
            )
            if os.path.exists(sorter_output_file):
                continue

            # Check if axon scan or network recording
            if len(rec_names) > 1:
                recording, _, _ = concatenate_recording_slices(rec_path, stream_id)
            else:
                recording = si.read_maxwell(
                    rec_path, stream_id=stream_id, rec_name=rec_names[0]
                )

            sorting = clean_sorting(
                recording,
                save_root,
                stream_id,
                sorter,
                sorter_params,
                clear_files=clear_files,
                verbose=verbose,
            )
            sorting_list.append(sorting)

    return sorting_list


def convert_rec_path_to_save_path(rec_path, save_path_changes):
    """
    Function that converts a recording path to the corresponding save path for a spike sorting.

    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    save_path_changes: dict
        Dictionary containing keys 'pos' and 'vals' that indicate the changes to be made to the rec_path.
        Refer to the inidices after splitting the path by '/'.

    Returns
    ----------
    save_path: str
        Root save path. Well ID will be appended during the sorting.
    """

    path_parts = rec_path.split("/")
    for x, y in zip(save_path_changes["pos"], save_path_changes["vals"]):
        path_parts[x] = y

    save_path = os.path.join(*path_parts)

    return save_path


def get_electrode_ids(recording):
    """
    Returns the MaxWell electrode ID of each channel of a recording (in channel order).

    Arguments
    ----------
    recording: BaseRecording
        Recording with a MaxWell probe attached (e.g. from si.read_maxwell).

    Returns
    ----------
    electrode_ids: np.ndarray
    """
    probegroup = recording.get_probegroup()
    return np.concatenate(
        [np.asarray(probe.contact_annotations["electrode"]) for probe in probegroup.probes]
    )


def select_and_rename_channels(recording, channel_ids, renamed_channel_ids=None):
    """Selects channels of a recording and optionally renames them."""
    sliced = recording.select_channels(channel_ids)
    if renamed_channel_ids is not None:
        sliced = sliced.rename_channels(new_channel_ids=renamed_channel_ids)
    return sliced


def find_common_electrodes(rec_path, stream_id):
    """
    Function that returns the common electrodes of the successive axon scan recordings.

    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    stream_id: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"

    Returns
    ----------
    rec_names: list
        List of rec_names for the specified recording/well.
    common_el: np.ndarray
        Electrodes that are present in all axon scan recordings.
    pos: dict
        Dictionary with keys 'x' and 'y' containing the positions of all recorded electrodes
        (NaN for electrodes that were never recorded).
    """

    assert os.path.exists(rec_path)

    with h5py.File(rec_path, "r") as h5:
        rec_names = list(h5["wells"][stream_id].keys())

    x, y = np.full(ut.N_ELECTRODES, np.nan), np.full(ut.N_ELECTRODES, np.nan)
    common_el = None

    for rec_name in rec_names:
        rec = si.read_maxwell(rec_path, stream_id=stream_id, rec_name=rec_name)
        rec_el = get_electrode_ids(rec)
        locations = rec.get_channel_locations()
        x[rec_el] = locations[:, 0]
        y[rec_el] = locations[:, 1]

        common_el = rec_el if common_el is None else np.intersect1d(common_el, rec_el)

    pos = {"x": x, "y": y}

    return rec_names, common_el, pos


def concatenate_recording_slices(rec_path, stream_id, center=True):
    """
    Function that centers and concatenates the recordings of an axon scan for all common electrodes.

    Arguments
    ----------
    rec_path: str
        Path to the axon scan file.
    stream_id: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"
    center: bool
        Flag if each recording slice should be median-centered before concatenation. Default: True

    Returns
    ----------
    multirecording: ConcatenateSegmentRecording (or MaxwellRecordingExtractor if there is only one recording)
        Concatenated recording across common electrodes (spikeinterface object)
    common_el: np.ndarray
        Electrodes that are present in all recordings (channel i of multirecording is common_el[i]).
    pos: dict
        Electrode positions, see find_common_electrodes.
    """

    rec_names, common_el, pos = find_common_electrodes(rec_path, stream_id)
    if len(rec_names) == 1:
        rec = si.read_maxwell(rec_path, stream_id=stream_id, rec_name=rec_names[0])
        return rec, common_el, pos

    rec_list = []
    for rec_name in rec_names:
        rec = si.read_maxwell(rec_path, stream_id=stream_id, rec_name=rec_name)
        rec_el = get_electrode_ids(rec)

        chan_idx = [np.flatnonzero(rec_el == el)[0] for el in common_el]
        sel_channels = rec.get_channel_ids()[chan_idx]
        if center:
            # Fallback for ultra short recordings (too little activity)
            chunk_size = max(min(10000, rec.get_num_samples()) - 100, 1)
            rec = si.center(rec, chunk_size=chunk_size)

        rec_list.append(
            select_and_rename_channels(
                rec, sel_channels, renamed_channel_ids=list(range(len(chan_idx)))
            )
        )

    multirecording = si.concatenate_recordings(rec_list)

    return multirecording, common_el, pos


def intersect_and_concatenate_recording_list(rec_list):
    assert len(rec_list) > 1
    rec_el_list = [get_electrode_ids(rec) for rec in rec_list]
    common_el = rec_el_list[0]
    for rec_el in rec_el_list[1:]:
        common_el = np.intersect1d(common_el, rec_el)

    sliced_rec_list = []
    for rec, els in zip(rec_list, rec_el_list):
        chan_idx = [np.flatnonzero(els == el)[0] for el in common_el]
        channel_ids = rec.get_channel_ids()[chan_idx]
        slice_rec = select_and_rename_channels(
            rec, channel_ids, renamed_channel_ids=list(range(len(channel_ids)))
        )
        sliced_rec_list.append(slice_rec.astype("float32"))

    concatenated = si.concatenate_recordings(sliced_rec_list)
    return concatenated


def clean_sorting(
    rec,
    save_root,
    stream_id,
    sorter,
    sorter_params=None,
    clear_files=True,
    verbose=True,
):
    """
    Function that creates output folder if it does not exist, sorts the recording using the specified sorter
    and clears up large files afterwards.

    Arguments
    ----------
    rec: BaseRecording
        Recording to be sorted.
    save_root: str
        Root path where the sorted data will be stored. Stream name (i.e. well ID) will be appended.
    stream_id: str
        Well ID in the format "well***"; Well 1 would be "well001", Well 20 would be "well020"
    sorter: str
        Name of the sorter, as per the spikeinterface convention (e.g. 'kilosort2_5')
    sorter_params: dict
        Dictionary containing parameters for the spike sorter. If left empty, will default to default parameters.
    clear_files: bool
        Flag if large temporary files should be deleted after the sorting. Default: True
    verbose: bool
        Default: True

    Returns
    ----------
    sorting: Sorting object
        Specific type depends on the sorter.
    """
    sorter_params = sorter_params or {}

    output_folder = Path(os.path.join(save_root, stream_id))
    sorter_output_file = os.path.join(output_folder, "sorter_output", "amplitudes.npy")
    sorting = []
    # Creates output folder if sorting has not yet been done
    if os.path.exists(sorter_output_file):
        return sorting
    elif rec.get_total_duration() < 30:
        full_output_folder = Path(os.path.join(output_folder, "sorter_output"))
        full_output_folder.mkdir(parents=True, exist_ok=True)
        np.save(
            sorter_output_file, np.empty(0)
        )  # Empty file to indicate a failed sorting for future loops
        return sorting
    else:
        raw_file = os.path.join(output_folder, "sorter_output", "recording.dat")
        wh_file = os.path.join(output_folder, "sorter_output", "temp_wh.dat")

        if verbose:
            print(
                f"DURATION: {rec.get_num_frames() / rec.get_sampling_frequency()} s -- "
                f"NUM. CHANNELS: {rec.get_num_channels()}"
            )

        # We use try/catch to not break loops when iterating over several sortings (e.g. when not all wells were recorded)
        try:
            t_start_sort = time.time()
            sorting = si.run_sorter(
                sorter,
                rec,
                folder=output_folder,
                verbose=verbose,
                remove_existing_folder=True,
                **sorter_params,
            )
            if verbose:
                print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")
        except Exception as e:
            print(e)
        finally:
            # Making sure we clean up the largest temporary files
            if clear_files:
                for tmp_file in (wh_file, raw_file):
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)

    return sorting


def generate_rec_list(path_parts):
    """
    Function that takes a list of strings (path parts) and finds all recordings matching the path pattern, and returns the stream ids for the first recordings.

    Arguments
    ----------
    path_parts: List of strings
        Parts of the path pattern to be concatenated to look for recordings matching the description (when using *wildcard)

    Returns
    ----------
    path_list: List of strings
        List of the recordings matching the pattern provided in path_parts.
    stream_ids: List of strings
        List of stream_ids (wells) recorded from the first recording.
    """
    path_pattern = os.path.join(*path_parts)
    path_list = sorted(glob(path_pattern))
    if not path_list:
        raise FileNotFoundError(f"No recordings found matching {path_pattern}")

    stream_ids = get_stream_ids(path_list[0])

    return path_list, stream_ids


def concatenate_recording_list(path_list, stream_id):
    well_recording_list = []
    for rec_path in path_list:  # Iterate over recordings to be concatenated
        try:
            well_recording_list.append(si.read_maxwell(rec_path, stream_id=stream_id))
        except Exception as e:  # If not all wells were recorded, should be the only cause for an error
            raise ValueError(f"Could not load {stream_id} from {rec_path}") from e

    multirecording = si.concatenate_recordings(well_recording_list)

    saturated_count = find_saturated_channels(well_recording_list)
    clean_multirecording = multirecording.remove_channels(
        multirecording.get_channel_ids()[saturated_count > 0]
    )

    return clean_multirecording


def cut_concatenated_recording(concat_rec, cutout=np.inf):
    rec_list = concat_rec.recording_list
    sliced_list = []
    for rec in rec_list:
        duration = rec.get_total_duration()
        if cutout < duration:
            end_frame = rec.get_num_frames()
            start_frame = int(end_frame - cutout * rec.get_sampling_frequency())
            sliced_rec = rec.frame_slice(start_frame, end_frame)
            sliced_list.append(sliced_rec)
        else:
            sliced_list.append(rec)

    concat_sliced = si.concatenate_recordings(sliced_list)
    return concat_sliced


def split_concatenated_sorting(sorting_path, path_suffix="sorter_output"):
    """
    Function that takes the path of concatenated sorting and returns a SegmentSorting based on the durations of the individual recordings.

    Arguments
    ----------
    sorting_path: string
        Output path indicated in spikeinterface that contains the 'spikeinterface_recording.json' file.
    path_suffix: string
        Subfolder in sorting_path that contains the sorter_output.

    Returns
    ----------
    segment_sorting: Spikeinterface SegmentSorting object
    concat_rec: Concatenated recording that was sorted
    """
    sorting_output = os.path.join(sorting_path, path_suffix)
    sorting = si.read_kilosort(sorting_output)
    recording_path = os.path.join(sorting_path, "spikeinterface_recording.json")
    concat_rec = si.load(recording_path, base_folder=True)
    cleaned_sorting = si.remove_excess_spikes(sorting, concat_rec)
    cleaned_sorting.register_recording(concat_rec)
    segment_sorting = si.SplitSegmentSorting(cleaned_sorting, concat_rec)

    return segment_sorting, concat_rec


def save_split_sorting(
    seg_sorting, subfolder="segment_", keep_unit_ids=None, cutout=(0, np.inf), phy_folder=None
):
    """Saves the split sorting into subfolders for each segment in the phy format.

    Args:
        seg_sorting (SegmentSorting): Spikeinterface SegmentSorting object.
        subfolder (str, optional): Prefix of the subfolders. Defaults to 'segment_'.
        keep_unit_ids (list, optional): List of unit ids to be kept, e.g., as a QC result. Defaults to None, which uses all units.
        cutout (list or np.array, optional): Cutout in seconds to be kept (relevant for wash-in artefacts). Can be 2D if different cutouts should be used. Defaults to [0, np.inf], which uses the entire duration for each segment.
        phy_folder (str, optional): Folder of the original phy/kilosort output. Defaults to the 'phy_folder' annotation of seg_sorting.
    """
    N_segments = seg_sorting.get_num_segments()
    cutout = np.asarray(cutout, dtype=float)
    if cutout.ndim == 1:  # If only one cutout is provided, we assume it applies to all segments
        cutout = np.tile(cutout, (N_segments, 1))

    if phy_folder is None:
        phy_folder = seg_sorting.get_annotation("phy_folder")
    if phy_folder is None:
        raise ValueError("Could not infer the phy folder, please provide phy_folder.")

    if len(seg_sorting.get_unit_ids()) == 0:
        return

    for seg_id in range(N_segments):
        seg = si.SelectSegmentSorting(seg_sorting, seg_id)
        if keep_unit_ids is not None:
            seg = seg.select_units(np.atleast_1d(np.squeeze(keep_unit_ids)).tolist())

        spikes = seg.to_spike_vector()

        if not (cutout[seg_id][0] == 0 and cutout[seg_id][1] == np.inf):
            if cutout[seg_id][1] == np.inf:
                end_frame = spikes["sample_index"].max() + 1
            else:
                end_frame = int(cutout[seg_id][1] * seg.get_sampling_frequency())

            start_frame = int(cutout[seg_id][0] * seg.get_sampling_frequency())

            seg = seg.frame_slice(start_frame, end_frame)
            spikes = seg.to_spike_vector()

        save_path = os.path.join(phy_folder, subfolder + str(seg_id))
        Path(save_path).mkdir(exist_ok=True)
        template_mat_path = os.path.join(phy_folder, "qc_output", "templates.npy")
        if not os.path.exists(template_mat_path):
            template_mat_path = os.path.join(
                phy_folder, "templates.npy"
            )  # In case bc output was not exported

        np.save(os.path.join(save_path, "spike_times.npy"), spikes["sample_index"])
        np.save(
            os.path.join(save_path, "spike_templates.npy"),
            np.asarray(seg.unit_ids)[spikes["unit_index"]],
        )
        shutil.copy(template_mat_path, save_path)
        shutil.copy(os.path.join(phy_folder, "channel_positions.npy"), save_path)
        shutil.copy(os.path.join(phy_folder, "params.py"), save_path)


def find_saturated_channels(rec_list, threshold=0):
    """
    Function that finds channels that are saturated (stuck at the minimum or maximum ADC value)
    in a list of recordings.

    Arguments
    ----------
    rec_list: List of MaxwellRecordingExtractor objects.
        List of (potentially to be concatenated) recordings to be checked for saturated channels.
    threshold: float
        Maximum ratio of saturated signal for the channel to still be accepted as non-saturated.

    Returns
    ----------
    saturated_count: np.array
        Number of recordings in which the saturation threshold was crossed (channel was considered to be saturated). Values go from 0 to len(rec_list).
    """
    saturated_count = np.zeros((rec_list[0].get_num_channels()))

    for rec in rec_list:
        random_data = si.get_random_data_chunks(
            rec,
            return_in_uV=False,
            num_chunks_per_segment=max(int(rec.get_total_duration() / 60), 1),
        )
        saturated = (
            np.sum((random_data == 0) | (random_data == 1023), axis=0)
            / random_data.shape[0]
        )
        saturated_count += saturated > threshold
    return saturated_count


def get_stream_ids(rec_path):
    with h5py.File(rec_path, "r") as h5:
        stream_ids = list(h5["wells"].keys())
    return stream_ids


def get_recording_path(sort_or_rec):
    """Returns the file path of the source recording of a sorting or (processed) recording."""
    return ut.find_source_recording(sort_or_rec)._kwargs["file_path"]
