import os, time, h5py, shutil
import spikeinterface.full as si
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob


def sort_recording_list(
    path_list,
    save_path_changes,
    sorter,
    sorter_params=dict(),
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

        h5 = h5py.File(rec_path)
        # Check that all wells are recorded throughout all recordings (should not fail)
        stream_ids = list(h5["wells"].keys())

        save_root = convert_rec_path_to_save_path(rec_path, save_path_changes)

        for stream_id in tqdm(stream_ids, desc="Sorting wells"):
            sorter_output_file = Path(
                os.path.join(save_root, stream_id, "sorter_output", "amplitudes.npy")
            )
            if not os.path.exists(sorter_output_file):
                # Check if axon scan or network recording
                rec_names = list(h5["wells"][stream_id].keys())
                if len(rec_names) > 1:
                    recording, common_el, pos = concatenate_recording_slices(
                        rec_path, stream_id
                    )
                else:
                    recording = si.MaxwellRecordingExtractor(
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
    common_el: list
        List of electrodes that are present in all axon scan recordings.
    """

    assert os.path.exists(rec_path)

    h5 = h5py.File(rec_path)
    rec_names = list(h5["wells"][stream_id].keys())
    pos = dict()
    x, y = np.full([1, 26400], np.nan), np.full([1, 26400], np.nan)

    for i, rec_name in enumerate(rec_names):
        # rec_name = 'rec' + '%0*d' % (4, rec_id)
        rec = si.MaxwellRecordingExtractor(
            rec_path, stream_id=stream_id, rec_name=rec_name
        )
        rec_el = rec.get_property("contact_vector")["electrode"]
        x[:, rec_el] = rec.get_property("contact_vector")["x"]
        y[:, rec_el] = rec.get_property("contact_vector")["y"]

        if i == 0:
            common_el = rec_el
        else:
            common_el = list(set(common_el).intersection(rec_el))

    pos = {"x": x[0], "y": y[0]}

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

    Returns
    ----------
    multirecording: ConcatenatedRecordingSlice
        Concatenated recording across common electrodes (spikeinterface object)
    """

    rec_names, common_el, pos = find_common_electrodes(rec_path, stream_id)
    if len(rec_names) == 1:
        rec = si.MaxwellRecordingExtractor(
            rec_path, stream_id=stream_id, rec_name=rec_names[0]
        )
        return rec
    else:
        rec_list = []
        for rec_name in rec_names:
            # rec_name = 'rec' + '%0*d' % (4, r)
            rec = si.MaxwellRecordingExtractor(
                rec_path, stream_id=stream_id, rec_name=rec_name
            )

            ch_id = rec.get_property("contact_vector")["device_channel_indices"]
            rec_el = rec.get_property("contact_vector")["electrode"]

            chan_idx = [np.where(rec_el == el)[0][0] for el in common_el]
            sel_channels = rec.get_channel_ids()[chan_idx]
            if center:
                chunk_size = (
                    np.min([10000, rec.get_num_samples()]) - 100
                )  # Fallback for ultra short recordings (too little activity)
                rec = si.center(rec, chunk_size=chunk_size)

            rec_list.append(
                rec.channel_slice(
                    sel_channels, renamed_channel_ids=list(range(len(chan_idx)))
                )
            )

        multirecording = si.concatenate_recordings(rec_list)

        return multirecording, common_el, pos


def intersect_and_concatenate_recording_list(rec_list):
    assert len(rec_list) > 1
    rec_el_list = []
    sliced_rec_list = []
    for i, rec in enumerate(rec_list):
        rec_el = rec.get_property("contact_vector")["electrode"]
        rec_el_list.append(rec_el)
        if i == 0:
            common_el = rec_el
        else:
            common_el = list(set(common_el).intersection(rec_el))

    for i, els in enumerate(rec_el_list):
        chan_idx = [np.where(els == el)[0][0] for el in common_el]
        channel_ids = rec_list[i].get_channel_ids()[chan_idx]
        slice_rec = rec_list[i].channel_slice(
            channel_ids, renamed_channel_ids=list(range(len(channel_ids)))
        )
        sliced_rec_list.append(slice_rec.astype("float32"))

    concatenated = si.concatenate_recordings(sliced_rec_list)
    return concatenated


def clean_sorting(
    rec,
    save_root,
    stream_id,
    sorter,
    sorter_params=dict(),
    clear_files=True,
    verbose=True,
):
    """
    Function that creates output folder if it does not exist, sorts the recording using the specified sorter
    and clears up large files afterwards.

    Arguments
    ----------
    rec: MaxwellRecordingExtractor
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
        # output_folder.mkdir(parents=True, exist_ok=True)
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
                output_folder=output_folder,
                verbose=verbose,
                remove_existing_folder=True,
                **sorter_params,
            )
            if verbose:
                print(f"\n\nSpike sorting elapsed time {time.time() - t_start_sort} s")

            # Making sure we clean up the largest temporary files
            if clear_files & os.path.exists(wh_file):
                os.remove(wh_file)
            if clear_files & os.path.exists(raw_file):
                os.remove(raw_file)
        except Exception as e:
            print(e)
            if clear_files & os.path.exists(wh_file):
                os.remove(wh_file)
            if clear_files & os.path.exists(raw_file):
                os.remove(raw_file)

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
    path_list = glob(path_pattern)
    h5 = h5py.File(path_list[-1])
    stream_ids = list(h5["wells"].keys())
    path_list.sort()

    return path_list, stream_ids


def concatenate_recording_list(path_list, stream_id):
    well_recording_list = []
    for rec_path in path_list:  # Iterate over recordings to be concatenated
        try:  # If not all wells were recorded, should be the only cause for an error
            rec = si.MaxwellRecordingExtractor(rec_path, stream_id=stream_id)
            well_recording_list.append(rec)
        except Exception:
            continue

    if len(well_recording_list) == len(path_list):
        multirecording = si.concatenate_recordings(well_recording_list)
    else:
        raise ValueError("Could not load all recordings!")

    saturated_count = find_saturated_channels(well_recording_list)
    clean_multirecording = multirecording.remove_channels(
        multirecording.get_channel_ids()[saturated_count > 0]
    )

    return clean_multirecording


def cut_concatenated_recording(concat_rec, cutout=np.inf):
    rec_list = concat_rec._kwargs["recording_list"]
    sliced_list = []
    for rec in rec_list:
        duration = rec.get_total_duration()
        if cutout < duration:
            end_frame = rec.get_num_frames()
            start_frame = end_frame - cutout * rec.get_sampling_frequency()
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
        Output path indicated in spikeinterface that contains the 'recording.json' file.
    path_prefix: string
        Subfolder in sorting_path that contains the sorter_output.

    Returns
    ----------
    segment_sorting: Spikeinterface SegmentSorting object
    """
    sorting_output = os.path.join(sorting_path, path_suffix)
    sorting = si.KiloSortSortingExtractor(sorting_output)
    recording_path = os.path.join(sorting_path, "spikeinterface_recording.json")
    concat_rec = si.load_extractor(recording_path, base_folder=True)
    cleaned_sorting = si.remove_excess_spikes(sorting, concat_rec)
    cleaned_sorting.register_recording(concat_rec)
    segment_sorting = si.SplitSegmentSorting(cleaned_sorting, concat_rec)

    return segment_sorting, concat_rec


def save_split_sorting(
    seg_sorting, subfolder="segment_", keep_unit_ids=None, cutout=[0, np.inf]
):
    """Saves the split sorting into subfolders for each segment in the phy format.

    Args:
        seg_sorting (SegmentSorting): Spikeinterface SegmentSorting object.
        subfolder (str, optional): Prefix of the subfolders. Defaults to 'segment_'.
        keep_unit_ids (list, optional): List of unit ids to be kept, e.g., as a QC result. Defaults to None, which uses all units.
        cutout (list or np.array, optional): Cutout in seconds to be kept (relevant for wash-in artefacts). Can be 2D if different cutouts should be used. Defaults to [0, np.inf], which uses the entire duration for each segment.
    """
    if (
        len(cutout.shape) == 1
    ):  # If only one cutout is provided, we assume it applies to all segments
        cutout = np.tile(cutout, (seg_sorting.get_num_segments(), 1))
    N_segments = seg_sorting.get_num_segments()
    if len(seg_sorting.get_unit_ids()) > 0:
        for seg_id in range(N_segments):
            seg = si.SelectSegmentSorting(seg_sorting, seg_id)
            if keep_unit_ids is not None:
                seg = seg.select_units(
                    np.squeeze(keep_unit_ids).tolist()
                )  # ,renamed_unit_ids=list(range(len(keep_unit_ids)))

            spikes = seg.to_spike_vector()
            # duration = np.ceil(spikes['sample_index'].max()/seg.get_sampling_frequency())

            if cutout[seg_id][0] == 0 and cutout[seg_id][1] == np.inf:
                pass
            else:
                if cutout[seg_id][1] == np.inf:
                    end_frame = spikes["sample_index"].max() + 1
                else:
                    end_frame = cutout[seg_id][1] * seg.get_sampling_frequency()

                start_frame = cutout[seg_id][0] * seg.get_sampling_frequency()

                seg = seg.frame_slice(start_frame, end_frame)

            # spike_vector = seg.to_spike_vector(concatenated=True) #Removes original unit IDs
            save_path = os.path.join(
                seg_sorting._annotations["phy_folder"], subfolder + str(seg_id)
            )
            Path(save_path).mkdir(exist_ok=True)
            spike_times_path = os.path.join(save_path, "spike_times.npy")
            spike_templates_path = os.path.join(save_path, "spike_templates.npy")
            template_mat_path = os.path.join(
                seg_sorting._annotations["phy_folder"], "qc_output", "templates.npy"
            )
            if not os.path.exists(template_mat_path):
                template_mat_path = os.path.join(
                    seg_sorting._annotations["phy_folder"], "templates.npy"
                )  # In case bc output was not exported

            channel_pos_path = os.path.join(
                seg_sorting._annotations["phy_folder"], "channel_positions.npy"
            )
            params_pos_path = os.path.join(
                seg_sorting._annotations["phy_folder"], "params.py"
            )
            np.save(
                spike_times_path, seg.get_all_spike_trains()[0][0]
            )  # spike_vector['sample_index'])
            np.save(
                spike_templates_path, seg.get_all_spike_trains()[0][1]
            )  # spike_vector['unit_index'])
            shutil.copy(template_mat_path, save_path)
            shutil.copy(channel_pos_path, save_path)
            shutil.copy(params_pos_path, save_path)


def find_saturated_channels(rec_list, threshold=0):
    """
    Function that creates output folder if it does not exist, sorts the recording using the specified sorter
    and clears up large files afterwards.

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

    for i in range(0, len(rec_list)):
        random_data = si.get_random_data_chunks(
            rec_list[i],
            num_chunks_per_segment=int((rec_list[i].get_total_duration() / 60)),
        )
        saturated = (
            np.sum(
                (random_data == 0).astype("int16")
                + (random_data == 1023).astype("int16"),
                axis=0,
            )
        ) / random_data.shape[0]
        saturated_count += saturated > threshold
    return saturated_count


def get_stream_ids(rec_path):
    h5 = h5py.File(rec_path)
    stream_ids = list(h5["wells"].keys())
    return stream_ids


def get_recording_path(sort_or_rec):
    start_dict = sort_or_rec
    while "file_path" not in start_dict._kwargs.keys():
        if "_recording" in vars(start_dict) and start_dict._recording is not None:
            start_dict = start_dict._recording
        elif "sorting" in start_dict._kwargs.keys():
            start_dict = start_dict._kwargs["sorting"]
        elif "recording" in start_dict._kwargs.keys():
            start_dict = start_dict._kwargs["recording"]
        elif "recording_or_recording_list" in start_dict._kwargs.keys():
            start_dict = start_dict._kwargs["recording_or_recording_list"]
        elif "parent_recording" in start_dict._kwargs.keys():
            start_dict = start_dict._kwargs["parent_recording"]
        elif "recording_list" in start_dict._kwargs.keys():
            start_dict = start_dict._kwargs["recording_list"]
        else:
            print("Could not find recording path")
            file_path = []
            break
        try:
            start_dict = start_dict[0]

        except Exception as e:
            continue

    file_path = start_dict._kwargs["file_path"]

    return file_path
