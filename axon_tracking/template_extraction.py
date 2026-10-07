"""
This module provides functions for extracting templates from spike sorting data,
performing quality control, and organizing the extracted templates into a grid format.
Functions:
    extract_templates_from_sorting_list(sorting_list, qc_params=None, te_params=None, raise_errors=False):
        Performs template extraction from a list of sorting paths.
        Does not require a recording path.
    preprocess_sorting(sorting_path, qc_params):
        Performs QC and splitting of a sorting object.
    select_good_units(sorting, recording, qc_params):
        Selects good units based on quality control parameters from KS, BC and SI.
    extract_all_templates(segment_sorting, te_params):
        Extracts templates from all units in a sorting object.
    convert_to_grid(template_matrix, pos):
        Converts a template matrix into a grid format based on electrode positions.
"""

import os
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

import spikeinterface.full as si
from axon_tracking import spike_sorting as ss
from axon_tracking import utils as ut

DEFAULT_QC_PARAMS = {
    "min_n_spikes": 1500,  # Minimum number of spikes for a unit to be kept
    "exclude_mua": True,  # Exclude units that were labelled multi unit activity by kilosort
    "use_bc": False,  # Use bombcell output (cluster_bc_unitType.tsv) for QC
    "use_si": False,  # Use spikeinterface for QC (auto merge / redundant unit removal)
    "auto_merge": True,  # Automatically merge units (spikeinterface implementation)
    "remove_redundant": True,  # Remove redundant units (spikeinterface implementation)
    "filter_band": 150,  # Filter applied before the spikeinterface QC (see filter_recording)
}

DEFAULT_TE_PARAMS = {
    "n_jobs": 1,  # Number of cores to use for template extraction
    "filter_band": 150,  # Either float for the highpass filter frequency or list for the bandpass filter frequencies
    "overwrite": False,  # Flag if templates should be recalculated if already existing
    "max_spikes_per_unit": 1000,  # Maximum number of spikes to be used for template extraction
    # Least-squares footprints of all regressor units per configuration (axon_tracking.deconvolution) instead of
    # spike-triggered averages: removes the activity of co-active sorted units (e.g. network bursts) from the
    # templates without discarding spikes. Saved to <sorter_output>/templates_deconvolved.
    "deconvolve": False,
    "deconvolve_units": "all",  # regressors: "all" units of the sorter (also those removed by the QC) or "good"
    "deconvolve_pad_ms": (0.5, 2.0),  # window extension of the solve (the footprints must fit into the window)
    "deconvolve_max_unknowns": 15000,  # regressors x window samples (memory ~ 8 * max_unknowns^2 bytes)
    "deconvolve_chunk_s": 2.0,  # traces are read in chunks of this duration
    # axon tracking files store only snippets around trigger events. False (default) solves the deconvolution on
    # the stored samples as concatenated; "auto"/True on the real time axis (frame numbers), fitting only stored
    # samples (axon_tracking.triggered). The real-time solve handles the gaps exactly but is worse conditioned
    # (lags beyond the trigger window are rarely stored); since spike positions within the snippets vary, the
    # glued-together samples act mostly as noise in the concatenated solve
    "triggered": False,
    # with deconvolve: also null templates per unit (decoy windows of stored data in which the unit was silent,
    # through the same solve; deconvolution.null_footprint), saved next to the templates in <folder>_null
    "null_templates": False,
    # independent null templates per unit (disjoint decoy windows), saved to <folder>_null, <folder>_null2, ...: one
    # can serve as the decoy for thresholds, others e.g. as backgrounds of hybrid ground truth
    "null_draws": 1,
    # draw the null windows to match the network activity around the unit's spikes (all regressors' spikes within
    # +- 5 ms), so that the null contains the burst-related background of unsorted neurons (triggered.silent_decoys)
    "null_match_activity": False,
    # with network_sortings (extract_templates_from_sorting_list): network units within this distance of a sorted
    # unit's soma are treated as the same neuron and not matched
    "network_exclusion_um": 35.0,
    "network_matching_kwargs": {},  # spikeinterface tdc-peeler parameters (e.g. detect_threshold, amplitude_limits)
    "verbose": False,  # report the network units used per configuration
}


def extract_templates_from_sorting_list(
    sorting_list, qc_params=None, te_params=None, raise_errors=False, save_folders=None, path_map=None,
    network_sortings=None,
):
    """Performs template extraction from a list of sorting paths. Does not require a recording path.

    Args:
        sorting_list (list): Sorting path list with sorter_output suffix.
        qc_params (dict, optional): Dict of quality control parameters. Missing keys are taken from DEFAULT_QC_PARAMS.
        te_params (dict, optional): Dict of template extraction parameters. Missing keys are taken from DEFAULT_TE_PARAMS.
        raise_errors (bool, optional): Raise errors instead of printing them and continuing with the next sorting.
        save_folders (list, optional): Output folder per sorting (default: <sorting_path>/templates, or
            <sorting_path>/templates_deconvolved if te_params["deconvolve"]), e.g. to leave the sorting untouched.
        path_map (dict, optional): {old_prefix: new_prefix} for the raw data paths stored in
            spikeinterface_recording.json (e.g. {"/net/server/export/group": "S:/group"} on Windows).
        network_sortings (list, optional): Per sorting, the sorter_output of a network recording of the same well
            (or None). With te_params["deconvolve"], its units that were not sorted in the axon scan are detected
            in every configuration by template matching and used as additional regressors (network_units).
    """
    qc_params = {**DEFAULT_QC_PARAMS, **(qc_params or {})}
    te_params = {**DEFAULT_TE_PARAMS, **(te_params or {})}
    if save_folders is not None and len(save_folders) != len(sorting_list):
        raise ValueError("save_folders must have one entry per sorting")
    if network_sortings is not None and len(network_sortings) != len(sorting_list):
        raise ValueError("network_sortings must have one entry per sorting")

    si.set_global_job_kwargs(n_jobs=te_params["n_jobs"], progress_bar=False)
    for k, sorting_path in enumerate(tqdm(sorting_list)):
        default = os.path.join(sorting_path, "templates_deconvolved" if te_params["deconvolve"] else "templates")
        save_folder = str(save_folders[k]) if save_folders is not None else default
        if os.path.isdir(save_folder):  # Check if output folder exists
            _, _, files = next(os.walk(save_folder))
            if len(files) > 0 and not te_params["overwrite"]:  # Check for existing output in the folder
                print(f"Templates already extracted for {sorting_path}")
                continue
        try:
            regressor_sorting = None
            if te_params["deconvolve"] and te_params["deconvolve_units"] == "all":
                segment_sorting, regressor_sorting = preprocess_sorting(sorting_path, qc_params,
                                                                       return_excluded=True, path_map=path_map)
            else:
                segment_sorting = preprocess_sorting(sorting_path, qc_params, path_map=path_map)
            network = None
            if te_params["deconvolve"] and network_sortings is not None and network_sortings[k] is not None:
                from axon_tracking import network_units as nu

                net_output = network_sortings[k]
                templates = (nu.load_network_templates(net_output) if os.path.isdir(os.path.join(net_output, "qc_output"))
                             else nu.network_templates_from_sorting(net_output))
                network = nu.exclude_sorted_neurons(templates,
                                                    nu.sorter_unit_positions(sorting_path),
                                                    te_params["network_exclusion_um"])
            out = extract_all_templates(segment_sorting, te_params, regressor_sorting, network)
            template_matrix, pos = out[:2]
            unit_ids = segment_sorting.get_unit_ids()
            os.makedirs(save_folder, exist_ok=True)
            save_individual_templates(template_matrix, save_folder, unit_ids, pos)
            if len(out) > 2:
                for d, null_matrix in enumerate(out[2]):
                    null_folder = os.path.normpath(save_folder) + "_null" + (str(d + 1) if d else "")
                    os.makedirs(null_folder, exist_ok=True)
                    save_individual_templates(null_matrix, null_folder, unit_ids, pos)
                noise_table(template_matrix, out[2][0], unit_ids).to_csv(
                    os.path.join(save_folder, "unit_noise.csv"), index=False)

        except Exception as e:
            if raise_errors:
                raise
            print(f"Error in {sorting_path}: {type(e).__name__}: {e}")
            continue


def filter_recording(recording, filter_band):
    """Highpass (float) or bandpass (list/tuple) filters a recording, converting unsigned data
    (e.g. raw MaxWell uint16) to signed first, as required by spikeinterface preprocessing."""
    if recording.get_dtype().kind == "u":
        recording = si.unsigned_to_signed(recording)

    btype = "bandpass" if isinstance(filter_band, (list, tuple, np.ndarray)) else "highpass"
    # margin_ms has no default since spikeinterface 0.105; 5 ms was the previous default
    return si.filter(recording, btype=btype, band=filter_band, margin_ms=5.0)


def preprocess_sorting(sorting_path, qc_params, return_excluded=False, path_map=None):
    """Performs QC and splitting of a sorting object.

    Args:
        sorting_path (str): Path to the folder containing the sorting output.
        qc_params (dict): QC parameter dictionary.
        return_excluded (bool): Also return the units rejected by the KS/bombcell/spike count QC (split the same
            way), e.g. as additional regressors for the deconvolution. Units removed or merged by the
            spikeinterface QC are not among them (they duplicate kept units).
        path_map (dict, optional): Prefix remapping of the raw data paths (see utils.remap_path).

    Returns:
        segment_sorting (SegmentSorting): Sorting that was split according to the
        corresponding (axon tracking) recording.
        excluded_sorting (SegmentSorting or None): Rejected units (only if return_excluded; None if there are
            none).
    """
    # Load sorting
    sorting = si.read_kilosort(sorting_path)

    # Find recording path
    json_path = os.path.join(
        Path(sorting_path).parent.absolute(), "spikeinterface_recording.json"
    )

    # Load concatenated recording to use for splitting the sorting
    multirecording = ut.load_recording_json(json_path, path_map) if path_map else si.load(json_path, base_folder=True)

    # Clean sorting (perform quality control)
    cleaned_sorting = select_good_units(sorting, multirecording, qc_params)
    cleaned_sorting = si.remove_excess_spikes(
        cleaned_sorting, multirecording
    )  # Relevant if last spike time == recording_length
    cleaned_sorting.register_recording(multirecording)

    # Split sorting into segments if it is a ConcatenateSegmentRecording
    if isinstance(multirecording, si.ConcatenateSegmentRecording):
        cleaned_sorting = si.SplitSegmentSorting(cleaned_sorting, multirecording)
    if not return_excluded:
        return cleaned_sorting

    with warnings.catch_warnings():  # the same warnings were raised by select_good_units
        warnings.simplefilter("ignore")
        kept = set(basic_qc_ids(sorting, qc_params))
    excluded_ids = [u for u in sorting.get_unit_ids() if u not in kept]
    if not excluded_ids:
        return cleaned_sorting, None
    excluded = si.remove_excess_spikes(sorting.select_units(excluded_ids), multirecording)
    excluded.register_recording(multirecording)
    if isinstance(multirecording, si.ConcatenateSegmentRecording):
        excluded = si.SplitSegmentSorting(excluded, multirecording)
    return cleaned_sorting, excluded


def perform_si_qc(sorting, recording, qc_params):
    """Performs quality control on a sorting object using spikeinterface curation.

    Args:
        sorting (BaseSorting): Sorting extractor object.
        recording (BaseRecording): Recording that was sorted.
        qc_params (dict): Dict of quality control parameters.

    Returns:
        BaseSorting: Sorting object with units that passed the QC (merged units get new IDs).
    """
    recording = filter_recording(
        recording, qc_params.get("filter_band", DEFAULT_QC_PARAMS["filter_band"])
    )
    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)
    analyzer.compute(
        [
            "random_spikes",
            "templates",
            "unit_locations",
            "template_similarity",
            "correlograms",
        ]
    )

    if qc_params["auto_merge"]:
        analyzer = si.auto_merge_units(analyzer, presets=["similarity_correlograms"])

    sorting = analyzer.sorting
    if qc_params["remove_redundant"]:
        sorting = si.remove_redundant_units(
            analyzer, duplicate_threshold=0.8, remove_strategy="minimum_shift"
        )

    return sorting


def select_good_units(sorting, recording, qc_params):
    """Selects good units based on quality control parameters from KS, BC and SI.

    Args:
        sorting (BaseSorting): Sorting extractor object.
        recording (BaseRecording): Recording that was sorted (used for the spikeinterface QC).
        qc_params (dict): Dict of quality control parameters, see DEFAULT_QC_PARAMS:
            min_n_spikes (int): Units need more spikes than this to be kept.
            exclude_mua (bool): Exclude MUA as labeled by KS.
            use_bc (bool): Consider bombcell QC output (cluster_bc_unitType.tsv).
            use_si (bool): Perform spikeinterface QC (auto merge/redundant unit removal).

    Returns:
        BaseSorting: Sorting containing units that passed the QC.
    """
    qc_params = {**DEFAULT_QC_PARAMS, **qc_params}
    good_ids = basic_qc_ids(sorting, qc_params)
    cleaned_sorting = sorting.select_units(good_ids)

    if qc_params["use_si"] and len(good_ids) > 0:
        cleaned_sorting = perform_si_qc(cleaned_sorting, recording, qc_params)

    print(f"Keeping {cleaned_sorting.get_num_units()} good units")

    return cleaned_sorting


def basic_qc_ids(sorting, qc_params):
    """Units passing the Kilosort label, bombcell and spike count QC (before the spikeinterface QC)."""
    qc_params = {**DEFAULT_QC_PARAMS, **qc_params}
    n_units = sorting.get_num_units()
    keep_all = np.full((n_units,), True, dtype="bool")

    # Kilosort QC
    ks_labels = sorting.get_property("KSLabel")
    if qc_params["exclude_mua"] and ks_labels is not None:
        ks_idx = ks_labels == "good"
    else:
        if qc_params["exclude_mua"]:
            warnings.warn("No KSLabel found, cannot exclude MUA")
        ks_idx = keep_all

    # Bombcell QC (if present)
    bc_labels = sorting.get_property("bc_unitType")
    if qc_params["use_bc"] and bc_labels is not None and len(bc_labels) > 0:
        bc_idx = bc_labels == "GOOD"
    else:
        if qc_params["use_bc"]:
            warnings.warn("No bombcell output (bc_unitType) found, skipping bombcell QC")
        bc_idx = keep_all

    # Minimum number of spikes QC
    n_spikes = [
        len(sorting.get_unit_spike_train(x, segment_index=0))
        for x in sorting.get_unit_ids()
    ]
    good_n_spikes_idx = np.array(n_spikes) > qc_params["min_n_spikes"]

    # Combine QC
    good_idx = ks_idx & bc_idx & good_n_spikes_idx
    return sorting.get_unit_ids()[good_idx]


def noise_table(template_matrix, null_matrix, unit_ids):
    """Per unit: template noise, null noise (median over recorded electrodes of the robust successive-difference
    sd) and their ratio. The ratio is close to 1 for units the deconvolution separates; much larger values flag units
    it cannot separate from another regressor (e.g. a neuron sorted twice), whose templates are noise-dominated."""
    import pandas as pd

    from axon_tracking import path_evidence as pe

    rows = []
    for k, u in enumerate(unit_ids):
        T, N = (np.nan_to_num(m[k]).T[:, None, :] for m in (template_matrix, null_matrix))  # (electrodes, 1, t)
        recorded = np.abs(T[:, 0]).max(axis=1) > 0
        noise, null_noise = (float(np.median(pe.template_noise_sd(x)[recorded, 0])) for x in (T, N))
        rows.append(dict(unit=u, noise_uv=noise, null_noise_uv=null_noise,
                         noise_inflation=noise / null_noise if null_noise > 0 else np.nan))
    return pd.DataFrame(rows)


def extract_all_templates(segment_sorting, te_params, regressor_sorting=None, network=None):
    """Extracts templates from all units in a sorting object.

    Args:
        segment_sorting (BaseSorting): SegmentSorting object (one segment per recording of the axon scan).
        te_params (dict): Dict of template extraction parameters.
        regressor_sorting (BaseSorting, optional): Units used as regressors if te_params["deconvolve"]
            (split like segment_sorting; default: the units of segment_sorting).

    Returns:
        template_matrix (np.ndarray): Templates of shape (n_units, n_samples, 26400 electrodes); NaN for
            electrodes that were never recorded.
        pos (dict): Dict containing electrode positions.
        null_matrix (np.ndarray): only with te_params["deconvolve"] and te_params["null_templates"]: null
            templates, (te_params["null_draws"], n_units, n_samples, 26400).
    """
    te_params = {**DEFAULT_TE_PARAMS, **te_params}

    # Find the source recording, its path and which well it belongs to
    source_recording = ut.find_source_recording(segment_sorting)
    full_path = source_recording._kwargs["file_path"]
    stream_id = source_recording._kwargs["stream_id"]

    # Find cutout for waveform extraction
    _, cutout_ms = ut.get_cutout_info(full_path, stream_id=stream_id)

    # Find electrode positions
    rec_names, _, pos = ss.find_common_electrodes(full_path, stream_id)
    n_units = segment_sorting.get_num_units()
    template_matrix = null_matrix = None
    with_null = te_params["deconvolve"] and te_params["null_templates"]

    # Extract templates (main loop)
    for sel_idx, rec_name in enumerate(rec_names):
        rec = si.read_maxwell(full_path, stream_id=stream_id, rec_name=rec_name)
        seg_sort = si.SelectSegmentSorting(segment_sorting, sel_idx)
        if te_params["deconvolve"]:
            regressors = si.SelectSegmentSorting(regressor_sorting or segment_sorting, sel_idx)
            tmp_data = extract_templates_deconvolved(rec, seg_sort, regressors, te_params, cutout_ms, network,
                                                     return_null=with_null)
            if with_null:
                tmp_data, tmp_null = tmp_data
        else:
            tmp_data = extract_templates(rec, seg_sort, te_params, cutout_ms)

        if template_matrix is None:  # Initialize template matrix
            template_matrix = np.full(
                [n_units, tmp_data.shape[1], ut.N_ELECTRODES], np.nan, dtype="float32"
            )
            if with_null:
                null_matrix = np.full((te_params["null_draws"],) + template_matrix.shape, np.nan, dtype="float32")
        els = ss.get_electrode_ids(rec)
        template_matrix[:, :, els] = tmp_data
        if with_null:
            null_matrix[:, :, :, els] = tmp_null

    return (template_matrix, pos, null_matrix) if with_null else (template_matrix, pos)


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
        - 'filter_band': Highpass frequency (float) or bandpass frequencies (list).
        - 'max_spikes_per_unit': Maximum number of spikes per unit for spike extraction.
    cutout_ms : tuple
        A tuple containing two elements:
        - ms_before: Time in milliseconds before the spike to include in the waveform.
        - ms_after: Time in milliseconds after the spike to include in the waveform.
    Returns:
    np.ndarray
        A numpy array containing the extracted templates (n_units, n_samples, n_channels).
    """

    rec_filtered = filter_recording(recording, te_params["filter_band"])

    sorting = si.remove_excess_spikes(sorting, rec_filtered)
    sorting.register_recording(rec_filtered)
    if "main_channel_id" in sorting.get_property_keys():
        # set by Kilosort4 readers for the sorted (concatenated) recording, whose channel ids differ from those of
        # the configuration's recording
        sorting.delete_property("main_channel_id")

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec_filtered,
        sparse=False,
    )

    # Templates are averaged on the fly from the random spikes; storing all waveforms is not needed
    analyzer.compute(
        ["random_spikes", "templates"],
        extension_params={
            "random_spikes": {"max_spikes_per_unit": te_params["max_spikes_per_unit"]},
            "templates": {"ms_before": cutout_ms[0], "ms_after": cutout_ms[1]},
        },
    )

    return analyzer.get_extension("templates").get_data()


def extract_templates_deconvolved(recording, sorting, regressor_sorting, te_params, cutout_ms, network=None,
                                  return_null=False):
    """Least-squares footprints of the units of `sorting` in one recording (configuration), estimated jointly
    with all units of `regressor_sorting` (see axon_tracking.deconvolution) and, if `network`
    (network_units.NetworkTemplates) is given, with the network units detected in this recording by template
    matching (those that duplicate a regressor unit are left out).

    All spikes are used (not max_spikes_per_unit), since the spike-triggered sums and the Gram matrix of the spike
    trains must refer to the same spikes. The solve uses a window padded by te_params["deconvolve_pad_ms"]; the
    output is cropped to cutout_ms like extract_templates. If the regressors exceed
    te_params["deconvolve_max_unknowns"] (units x window samples), the extra regressors with the fewest spikes are
    left out; the units of `sorting` are always included.

    Triggered acquisition (te_params["triggered"]): if the file stores snippets around trigger events (gaps in the
    frame numbers), spikes and (filtered) traces are placed on the real time axis and only stored samples are
    fitted (axon_tracking.triggered); otherwise the recording is treated as continuous.

    return_null: also return null templates (te_params["null_draws"] independent ones): per unit and draw, as many
    decoy spike times as it has spikes (disjoint between draws), in fully stored
    windows in which it did not fire (mostly snippets triggered by other units), through the same solve
    (deconvolution.null_footprint).

    Returns:
        np.ndarray: Templates (n_units of sorting, n_samples, n_channels), in uV if the recording is scaleable;
        with return_null, (templates, null templates).
    """
    from axon_tracking import deconvolution as dc
    from axon_tracking import triggered as tg

    rec = filter_recording(recording, te_params["filter_band"])
    fs = rec.get_sampling_frequency()
    scaled = rec.has_scaleable_traces()
    frames = None
    if te_params["triggered"] or return_null:
        try:
            frames = tg.maxwell_frame_numbers(recording)
        except (OSError, KeyError) as e:
            if te_params["triggered"] is True:
                raise
            warnings.warn(f"No frame numbers ({type(e).__name__}: {e}); treating the recording as continuous")
        if frames is not None and np.all(np.diff(frames) == 1):
            frames = None  # continuous
    if frames is not None and te_params["triggered"]:
        # the stored samples are filtered as concatenated (snippets of a few ms are too short for a high-pass of
        # their own; the steps between consecutive snippets are small), then placed on the real time axis
        timeline = tg.TriggeredTraces(lambda a, b: rec.get_traces(segment_index=0, start_frame=a, end_frame=b,
                                                                 return_in_uV=scaled), frames)
        to_real, n_samples, recorded = timeline.to_real, timeline.n_samples, timeline.recorded
        get_traces = timeline.get_traces
    else:
        to_real, n_samples, recorded = (lambda t: np.asarray(t, np.int64)), rec.get_num_samples(segment_index=0), None
        get_traces = lambda start, end: rec.get_traces(segment_index=0, start_frame=start, end_frame=end,
                                                       return_in_uV=scaled)
    nbefore, nafter = int(cutout_ms[0] * fs / 1000.0), int(cutout_ms[1] * fs / 1000.0)  # as spikeinterface
    pad = te_params["deconvolve_pad_ms"]
    pre, post = nbefore + int(round(pad[0] * fs / 1000.0)), nafter + int(round(pad[1] * fs / 1000.0))
    window = pre + post

    output_ids = list(sorting.get_unit_ids())
    extra_ids = [u for u in regressor_sorting.get_unit_ids() if u not in set(output_ids)]
    trains = {u: to_real(s.get_unit_spike_train(u, segment_index=0))
              for s, ids in ((sorting, output_ids), (regressor_sorting, extra_ids)) for u in ids}
    trains = {u: t[(t >= pre) & (t <= n_samples - post)].astype(np.int64) for u, t in trains.items()}
    network_ids = []
    if network is not None:
        from axon_tracking import network_units as nu

        matched = nu.detect_in_recording(si.scale_to_uV(rec) if rec.has_scaleable_traces() else rec,
                                         ss.get_electrode_ids(recording), network,
                                         method_kwargs=te_params["network_matching_kwargs"])
        kept = nu.drop_duplicates_of(matched, [trains[u] for u in output_ids + extra_ids])
        if te_params["verbose"]:
            print(f"network units: {len(matched)} with soma in this configuration, {len(kept)} used "
                  f"({', '.join(f'{u}: {len(t)} spikes' for u, t in kept.items())}); left out as duplicates "
                  f"or too few spikes: {sorted(set(matched) - set(kept))}")
        matched = kept
        for u, t in matched.items():
            t = to_real(t)
            trains[f"network_{u}"] = t[(t >= pre) & (t <= n_samples - post)]
        network_ids = [f"network_{u}" for u in matched]
    budget = te_params["deconvolve_max_unknowns"] // window
    if len(output_ids) > budget:
        raise ValueError(f"{len(output_ids)} units x {window} samples exceed deconvolve_max_unknowns "
                         f"({te_params['deconvolve_max_unknowns']}); increase it or reduce deconvolve_pad_ms")
    # network units (somata recorded in this configuration) before the other extra regressors
    extra_ids = (sorted(network_ids, key=lambda u: -len(trains[u])) +
                 sorted(extra_ids, key=lambda u: -len(trains[u])))[: budget - len(output_ids)]
    unit_ids = output_ids + extra_ids
    spike_trains = [trains[u] for u in unit_ids]

    decoys = []
    if return_null:
        rng = np.random.default_rng(0)
        # decoy windows (cutout) stored without a gap, like the windows of the unit's own trigger spikes
        width = nbefore + nafter
        allowed = np.zeros(n_samples, bool)
        if recorded is not None:  # real time axis
            complete = np.convolve(recorded.astype(int), np.ones(width, int))[width - 1 : n_samples] == width
            allowed[np.flatnonzero(complete) + nbefore] = True
        elif frames is not None:  # concatenated stored samples: the cutout must not span a gap
            i = np.arange(nbefore, n_samples - nafter + 1)
            allowed[i[frames[i + nafter - 1] - frames[i - nbefore] == width - 1]] = True
        else:
            allowed[nbefore : n_samples - nafter + 1] = True
        n_draws = int(te_params["null_draws"])
        population = np.sort(np.concatenate(spike_trains)) if te_params["null_match_activity"] else None
        for u in output_ids:
            pool = tg.silent_decoys(trains[u], allowed, pre, post, rng, n=n_draws * len(trains[u]),
                                    population=population, half_window=int(5 * fs / 1000))
            pool = rng.permutation(pool)
            decoys += [np.sort(pool[d::n_draws]) for d in range(n_draws)]

    S = dc.spike_triggered_sums_chunked(get_traces, n_samples, spike_trains + decoys, pre, post,
                                        chunk_samples=int(te_params["deconvolve_chunk_s"] * fs))
    S, S_decoys = S[: len(spike_trains)], S[len(spike_trains) :]
    # one neuron sorted twice (locked at a fixed lag): solve with one of them, give the other the shifted footprint
    counts = np.maximum([len(t) for t in spike_trains], 1)[:, None, None]
    trough_channel = (S[:, pre - 2 : pre + 3] / counts).min(axis=1).argmin(axis=1)  # largest trough at the spike
    duplicates = dc.locked_duplicates(spike_trains, int(fs / 1000), sites=rec.get_channel_locations()[trough_channel])
    solve = [k for k in range(len(spike_trains)) if k not in {d for d, _, _ in duplicates}]
    F_solved, info = dc.solve_footprints([spike_trains[k] for k in solve], S[solve], window, recorded=recorded, pre=pre)
    F = np.zeros_like(S, dtype=float)
    F[solve] = F_solved
    for drop, keep, lag in duplicates:
        F[drop] = dc.shift_footprint(F[keep], lag)
    if duplicates and te_params["verbose"]:
        print("units locked to another unit (one neuron sorted twice), solved jointly: " +
              ", ".join(f"{unit_ids[d]} = {unit_ids[k]} shifted by {lag}" for d, k, lag in duplicates))
    if info["ridge"]:
        warnings.warn(f"Singular spike train Gram matrix (units locked to each other?): ridge {info['ridge']:.2g}")
    crop = slice(pre - nbefore, pre + nafter)
    templates = F[: len(output_ids), crop, :].astype(np.float32)
    if not return_null:
        return templates
    nulls = np.zeros((n_draws,) + templates.shape, np.float32)
    for k, d in enumerate(decoys):
        if len(d):
            nulls[k % n_draws, k // n_draws] = dc.null_footprint(d, S_decoys[k], [spike_trains[j] for j in solve],
                                                                  F[solve], pre, recorded)[0][crop]
    return templates, nulls


def deconvolve_saved_analyzers(analyzer_folders, save_folder=None):
    """Deconvolved templates from saved per-configuration sorting analyzers (zarr; e.g. <...>/spikes/seg_XX.zarr
    of an axon scan), without the raw recordings.

    Each analyzer must have the average templates of all spikes of its units (random_spikes covering every spike,
    i.e. fewer spikes than max_spikes_per_unit): the spike-triggered sums are then the averages times the spike
    counts, and the Gram matrix follows from the stored spike times. Regressors are the units of the analyzers
    (window as stored, no padding). Configurations are merged in the given order, the last one winning.

    Returns:
        unit_ids, average grids (units, 220, 120, samples), deconvolved grids; if save_folder is given, the grids
        are also saved to save_folder/templates/<unit>.npy and save_folder/templates_deconvolved/<unit>.npy.
    """
    import zarr

    from axon_tracking import deconvolution as dc

    unit_ids = grid_avg = grid_dec = None
    for folder in analyzer_folders:
        z = zarr.open(str(folder), mode="r")
        ids = z["sorting/unit_ids"][:]
        if unit_ids is None:
            unit_ids = ids
        elif not np.array_equal(ids, unit_ids):
            raise ValueError(f"{folder}: other units than the first analyzer")
        sample, unit = z["sorting/spikes/sample_index"][:], z["sorting/spikes/unit_index"][:]
        if len(z["extensions/random_spikes/random_spikes_indices"]) != len(sample):
            raise ValueError(f"{folder}: the templates were not computed from all spikes")
        average = np.asarray(z["extensions/templates/average"][:], dtype=np.float64)
        probes = z["recording_info"].attrs["probegroup"]["probes"]
        electrodes = np.concatenate([np.asarray(p["contact_annotations"]["electrode"]) for p in probes])
        counts = np.bincount(unit, minlength=len(ids))
        trains = [np.sort(sample[unit == k]).astype(np.int64) for k in range(len(ids))]
        F, info = dc.solve_footprints(trains, average * counts[:, None, None], average.shape[1])
        if info["ridge"]:
            warnings.warn(f"{folder}: singular spike train Gram matrix, ridge {info['ridge']:.2g}")
        if grid_avg is None:
            grid_avg = np.zeros((len(ids), ut.N_COLS, ut.N_ROWS, average.shape[1]), np.float32)
            grid_dec = np.zeros_like(grid_avg)
        cols, rows = electrodes % ut.N_COLS, electrodes // ut.N_COLS
        grid_avg[:, cols, rows] = average.transpose(0, 2, 1)
        grid_dec[:, cols, rows] = F.transpose(0, 2, 1)
    if save_folder is not None:
        for name, grids in (("templates", grid_avg), ("templates_deconvolved", grid_dec)):
            os.makedirs(os.path.join(save_folder, name), exist_ok=True)
            for k, u in enumerate(unit_ids):
                np.save(os.path.join(save_folder, name, f"{u}.npy"), grids[k])
    return unit_ids, grid_avg, grid_dec


def convert_to_grid(template_matrix, pos):
    """
    Converts a template matrix into a grid based on provided positions.
    Parameters:
    template_matrix (numpy.ndarray): The template matrix (n_samples, n_electrodes) to be converted.
    pos (dict): A dictionary containing 'x' and 'y' positions. The positions
                should be numpy arrays with the same length as the number of columns
                in the template_matrix.
    Returns:
    numpy.ndarray: A 3D grid where the first two dimensions correspond to the
                   x and y indices derived from the positions, and the third
                   dimension corresponds to the voltage over time.
    """

    # Clean up template matrix and positions to remove NaNs from sparse recordings
    recorded = ~np.isnan(pos["x"])
    clean_template = template_matrix[:, recorded]
    clean_x = pos["x"][recorded]
    clean_y = pos["y"][recorded]

    # Convert positions to the grid
    x_idx = np.rint(clean_x / ut.EL_SPACING).astype(int)
    y_idx = np.rint(clean_y / ut.EL_SPACING).astype(int)
    grid = np.zeros(
        [np.max(x_idx) + 1, np.max(y_idx) + 1, clean_template.shape[0]], dtype="float32"
    )
    grid[x_idx, y_idx, :] = clean_template.T

    return grid


def save_individual_templates(template_matrix, save_folder, unit_ids, pos):
    """
    Saves the template of each unit as a grid (see convert_to_grid) to <save_folder>/<unit_id>.npy.
    Parameters:
    template_matrix (numpy.ndarray): The template matrix (n_units, n_samples, n_electrodes) to be saved.
    save_folder (str): The folder to save the templates to.
    unit_ids (list): Unit IDs corresponding to the first dimension of template_matrix.
    pos (dict): Electrode positions, see convert_to_grid.
    """

    for i, unit_id in enumerate(unit_ids):
        save_path_i = os.path.join(save_folder, f"{unit_id}.npy")
        grid = convert_to_grid(template_matrix[i, :, :], pos)
        np.save(save_path_i, grid)
