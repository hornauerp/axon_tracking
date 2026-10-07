"""Helpers to write small synthetic MaxWell (MaxTwo format) recordings and Kilosort outputs for tests."""

import numpy as np
import h5py

FS = 20000
LSB = 6.2e-6  # V per bit
TRIGGER = (20, 40)  # trigger_pre, trigger_post in samples
WAVEFORM_HALF_WIDTH = 15


def electrode_id(col, row):
    return row * 220 + col


def electrode_xy(electrodes):
    electrodes = np.asarray(electrodes)
    return (electrodes % 220) * 17.5, (electrodes // 220) * 17.5


def block(cols, rows):
    """Electrode IDs of a rectangular block of columns x rows."""
    return np.array([electrode_id(c, r) for r in rows for c in cols])


def _waveform(amplitude):
    t = np.arange(-WAVEFORM_HALF_WIDTH, WAVEFORM_HALF_WIDTH + 1)
    return amplitude * np.exp(-0.5 * (t / 3.0) ** 2)


def write_maxwell_file(path, wells, n_samples, units=(), trigger=TRIGGER, noise_bits=2.0, seed=0, stored=None,
                       first_frame=1000000):
    """
    Args:
        path: File to write.
        wells: {well_name: [electrodes of rec0000, electrodes of rec0001, ...]}
        n_samples: Number of samples per recording.
        units: list of {"spike_times": {(well_name, rec_index): sample indices},
                        "footprint": {electrode: (amplitude_bits, delay_samples)}}
        trigger: (trigger_pre, trigger_post) written to groups/routed, or None (network recording).
        stored: optional {(well_name, rec_index): sample indices}: only these samples are stored (triggered
            acquisition), with their frame numbers (first_frame + sample) in groups/routed/frame_nos. Spike times
            and the signal refer to all n_samples.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as h5:
        h5["version"] = np.array([b"20190530"])
        for well, rec_electrodes in wells.items():
            for rec_index, electrodes in enumerate(rec_electrodes):
                electrodes = np.asarray(electrodes)
                group = h5.create_group(f"wells/{well}/rec{rec_index:04d}")
                group["settings/sampling"] = np.array([float(FS)])
                group["settings/lsb"] = np.array([LSB])
                mapping = np.zeros(
                    len(electrodes),
                    dtype=[("channel", "<i4"), ("electrode", "<i4"), ("x", "<f8"), ("y", "<f8")],
                )
                mapping["channel"] = np.arange(len(electrodes))
                mapping["electrode"] = electrodes
                mapping["x"], mapping["y"] = electrode_xy(electrodes)
                group["settings/mapping"] = mapping

                data = 512 + noise_bits * rng.standard_normal((len(electrodes), n_samples))
                for unit in units:
                    spike_times = unit["spike_times"].get((well, rec_index), [])
                    for channel, electrode in enumerate(electrodes):
                        if electrode not in unit["footprint"]:
                            continue
                        amplitude, delay = unit["footprint"][electrode]
                        waveform = _waveform(amplitude)
                        for t in spike_times:
                            start = t + delay - WAVEFORM_HALF_WIDTH
                            data[channel, start : start + len(waveform)] += waveform
                keep = np.arange(n_samples) if stored is None else np.asarray(stored[(well, rec_index)])
                group["groups/routed/raw"] = np.clip(np.rint(data[:, keep]), 0, 1023).astype("uint16")
                group["groups/routed/frame_nos"] = (first_frame + keep).astype(np.int64)
                if trigger is not None:
                    group["groups/routed/trigger_pre"] = np.array([trigger[0]])
                    group["groups/routed/trigger_post"] = np.array([trigger[1]])


def write_kilosort_output(folder, spike_times, spike_clusters, n_channels, ks_labels=None):
    """Minimal Kilosort/phy output folder readable by spikeinterface.read_kilosort."""
    folder.mkdir(parents=True, exist_ok=True)
    order = np.argsort(spike_times)
    np.save(folder / "spike_times.npy", np.asarray(spike_times, dtype="uint64")[order])
    np.save(folder / "spike_clusters.npy", np.asarray(spike_clusters, dtype="int32")[order])
    np.save(folder / "spike_templates.npy", np.asarray(spike_clusters, dtype="int32")[order])
    np.save(folder / "amplitudes.npy", np.ones(len(spike_times)))
    (folder / "params.py").write_text(
        f"dat_path = 'recording.dat'\nn_channels_dat = {n_channels}\ndtype = 'int16'\n"
        f"offset = 0\nsample_rate = {float(FS)}\nhp_filtered = False\n"
    )
    if ks_labels is not None:
        lines = "".join(f"{unit}\t{label}\n" for unit, label in ks_labels.items())
        (folder / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n" + lines)
