"""Spike sorting of MaxWell HD-MEA recordings (axon scans and network recordings) with Kilosort4.

Axon scans: the configurations are concatenated on their common electrodes (spike_sorting.concatenate_recording_
slices: in axon tracking assays, 3x3 blocks around the 30 trigger electrodes). Network recordings (one recording per
well) are sorted as they are. Kilosort4's defaults target Neuropixels probes (30 kHz, a narrow column of contacts,
drift); for the planar 17.5 um grid recorded at 10 kHz:

- no drift correction (nblocks=0): a culture on a chip does not move, and axon scans store concatenated trigger
  snippets, so there is no continuous time axis to estimate drift on;
- template centres on the electrode grid (dmin = dminx = 17.5 um) and x_centers set explicitly, as Kilosort
  recommends for 2D arrays (every ~200 um);
- local whitening and detection within an electrode block (whitening_range = nearest_chans = block size: 9 for the
  3x3 blocks of axon scans, 16 for 4x4 blocks) instead of 32 / 10 channels, which would mix in electrodes of blocks
  hundreds of um away;
- waveform length for 10 kHz: nt = 21 samples (2.1 ms, as Kilosort's 61 samples at 30 kHz), trough at sample 7.
- detection thresholds Th_universal / Th_learned = 7 / 6 instead of 9 / 8: more complete spike trains of the same
  units, hence less noisy templates; the few extra false detections can be flagged with axon_tracking.spike_qc.
  Much lower thresholds start to degrade the clustering.
- batches: 60000 samples (6 s) for axon scans; for network recordings (~1000 channels) 20000 (2 s, Kilosort's
  default duration): the universal-template step scales with templates x batch length, and long batches on many
  channels exceed the GPU memory.

--keep-binary keeps the exported binary (next to the sorting, for reruns), --binary reuses one (same dtype and
channel order as the recording that would be exported).

Usage: python sort_axon_scan_ks4.py --rec <data.raw.h5> --out <root> [--variant hdmea] well [well ...]
Output: <root>/<variant>/<well>/sorter_output (+ spikeinterface_recording.json for template extraction).
Needs the environment with kilosort and torch (CUDA), e.g. E:/envs/ks4.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py

sys.path.append(str(Path(__file__).resolve().parents[1]))
import spikeinterface.full as si  # noqa: E402

from axon_tracking import spike_sorting as ss  # noqa: E402


# the stored samples are written to a binary file for Kilosort; threads share the opened recording (worker
# processes would each reopen and re-centre all configurations of the scan first, which takes far longer)
BINARY_EXPORT = dict(use_binary_file=True, delete_recording_dat=True, n_jobs=8, pool_engine="thread",
                     chunk_duration="1s")


def hdmea_params(block_channels=9, **overrides):
    params = dict(
        nblocks=0,
        do_correction=False,
        dmin=17.5,
        dminx=17.5,
        x_centers=16,
        whitening_range=block_channels,
        nearest_chans=block_channels,
        nt=21,
        nt0min=7,
        Th_universal=7,
        Th_learned=6,
        batch_size=60000,  # 6 s at 10 kHz
        **BINARY_EXPORT,
    )
    return {**params, **overrides}


VARIANTS = {
    "neuropixels_defaults": lambda block: dict(BINARY_EXPORT),
    "hdmea": lambda block: hdmea_params(block),
    # for comparison: Kilosort's default thresholds, and lower ones
    "hdmea_th9_8": lambda block: hdmea_params(block, Th_universal=9, Th_learned=8),
    "hdmea_th6_5": lambda block: hdmea_params(block, Th_universal=6, Th_learned=5),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wells", nargs="+")
    parser.add_argument("--variant", default="hdmea", choices=sorted(VARIANTS))
    parser.add_argument("--rec", required=True, help="MaxWell raw file (data.raw.h5)")
    parser.add_argument("--out", required=True, help="output root (<out>/<variant>/<well>/sorter_output)")
    parser.add_argument("--block", type=int, default=None, help="channels per electrode block (default: 9 for axon "
                                                                  "scans, 16 for network recordings)")
    parser.add_argument("--batch-size", type=int, default=None, help="default: 60000 (axon scans), 20000 (network)")
    parser.add_argument("--keep-binary", action="store_true")
    parser.add_argument("--binary", default=None, help="existing binary of the recording to sort")
    args = parser.parse_args()
    for well in args.wells:
        with h5py.File(args.rec, "r") as f:
            rec_names = list(f["wells"][well].keys())
        if len(rec_names) > 1:  # axon scan
            recording, common_el, _ = ss.concatenate_recording_slices(args.rec, well)
            block = args.block or 9
        else:
            recording = si.read_maxwell(args.rec, stream_id=well, rec_name=rec_names[0])
            # 10-bit samples stored as uint16: signed int16 around 0 (the binary for Kilosort stays int16, half
            # the size of the float32 of a centred recording)
            recording = si.unsigned_to_signed(recording, bit_depth=10)
            block = args.block or 16
        params = VARIANTS[args.variant](block)
        params["batch_size"] = args.batch_size or (60000 if len(rec_names) > 1 else 20000)
        if args.keep_binary:
            params["delete_recording_dat"] = False
        if args.binary:
            source = recording
            recording = si.read_binary(args.binary, sampling_frequency=source.get_sampling_frequency(),
                                       dtype=source.get_dtype(), num_channels=source.get_num_channels(),
                                       channel_ids=source.get_channel_ids())
            recording.set_probe(source.get_probe(), in_place=True)
        print(f"{well}: {recording.get_num_channels()} channels, {recording.get_total_duration():.0f} s, "
              f"{recording.get_num_segments()} segment(s), batch {params['batch_size']}")
        ss.clean_sorting(recording, os.path.join(args.out, args.variant), well, "kilosort4",
                         sorter_params=params, clear_files=not args.keep_binary and not args.binary,
                         verbose=True)
        if args.binary:  # template extraction needs the source recording (MaxWell files), not the binary copy
            json_path = os.path.join(args.out, args.variant, well, "spikeinterface_recording.json")
            os.replace(json_path, json_path.replace(".json", "_binary.json"))
            source.dump_to_json(json_path)
