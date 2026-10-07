import numpy as np
import pytest
import spikeinterface.full as si

import synthetic as syn
from axon_tracking import spike_sorting as ss

COMMON = syn.block(range(20, 22), range(10, 12))  # recorded in every configuration
REC_ELECTRODES = [
    np.concatenate([syn.block(range(10, 13), range(10, 12)), COMMON]),
    np.concatenate([COMMON, syn.block(range(15, 18), range(10, 12))]),  # different channel order
]


@pytest.fixture
def axon_scan(tmp_path):
    path = tmp_path / "data.raw.h5"
    syn.write_maxwell_file(path, {"well000": REC_ELECTRODES}, n_samples=4000)
    return str(path)


def test_get_electrode_ids(axon_scan):
    rec = si.read_maxwell(axon_scan, stream_id="well000", rec_name="rec0001")
    np.testing.assert_array_equal(ss.get_electrode_ids(rec), REC_ELECTRODES[1])
    x, y = syn.electrode_xy(REC_ELECTRODES[1])
    np.testing.assert_allclose(rec.get_channel_locations(), np.column_stack([x, y]))


def test_find_common_electrodes(axon_scan):
    rec_names, common_el, pos = ss.find_common_electrodes(axon_scan, "well000")
    assert rec_names == ["rec0000", "rec0001"]
    np.testing.assert_array_equal(np.sort(common_el), np.sort(COMMON))
    recorded = np.concatenate(REC_ELECTRODES)
    assert np.count_nonzero(~np.isnan(pos["x"])) == len(np.unique(recorded))
    x, y = syn.electrode_xy(recorded)
    np.testing.assert_allclose(pos["x"][recorded], x)
    np.testing.assert_allclose(pos["y"][recorded], y)


def test_concatenate_recording_slices(axon_scan):
    multirecording, common_el, _ = ss.concatenate_recording_slices(axon_scan, "well000")
    assert isinstance(multirecording, si.ConcatenateSegmentRecording)
    assert multirecording.get_num_channels() == len(COMMON)
    assert multirecording.get_num_samples() == 2 * 4000
    # channel i of every slice is electrode common_el[i]
    for sliced in multirecording.recording_list:
        np.testing.assert_array_equal(ss.get_electrode_ids(sliced), common_el)


def test_concatenate_recording_slices_single_recording(tmp_path):
    path = tmp_path / "network.raw.h5"
    syn.write_maxwell_file(path, {"well000": REC_ELECTRODES[:1]}, n_samples=2000)
    recording, common_el, pos = ss.concatenate_recording_slices(str(path), "well000")
    assert recording.get_num_channels() == len(REC_ELECTRODES[0])
    np.testing.assert_array_equal(common_el, REC_ELECTRODES[0])


def test_intersect_and_concatenate_recording_list(axon_scan):
    recs = [si.read_maxwell(axon_scan, stream_id="well000", rec_name=r) for r in ("rec0000", "rec0001")]
    concatenated = ss.intersect_and_concatenate_recording_list(recs)
    assert concatenated.get_num_channels() == len(COMMON)
    assert concatenated.get_dtype() == np.float32


def test_get_stream_ids_and_recording_path(axon_scan):
    assert ss.get_stream_ids(axon_scan) == ["well000"]
    multirecording, _, _ = ss.concatenate_recording_slices(axon_scan, "well000")
    assert ss.get_recording_path(multirecording).replace("\\", "/").endswith("data.raw.h5")


def test_generate_rec_list(tmp_path, axon_scan):
    path_list, stream_ids = ss.generate_rec_list([str(tmp_path), "*.raw.h5"])
    assert path_list == [axon_scan] and stream_ids == ["well000"]
    with pytest.raises(FileNotFoundError):
        ss.generate_rec_list([str(tmp_path), "missing*.h5"])


def test_find_saturated_channels_short_recording():
    traces = np.full((20000, 3), 512, dtype="uint16")  # 1 s, shorter than the 60 s per random chunk
    traces[:, 1] = 1023  # saturated channel
    rec = si.NumpyRecording([traces], sampling_frequency=20000)
    np.testing.assert_array_equal(ss.find_saturated_channels([rec, rec]), [0, 2, 0])


def test_save_split_sorting(tmp_path):
    phy_folder = tmp_path / "sorter_output"
    phy_folder.mkdir()
    for name in ("templates.npy", "channel_positions.npy"):
        np.save(phy_folder / name, np.zeros(2))
    (phy_folder / "params.py").write_text("sample_rate = 20000.0\n")

    recordings = [si.generate_recording(num_channels=2, durations=[d], seed=0) for d in (1.0, 2.0)]
    multirecording = si.concatenate_recordings(recordings)
    sorting = si.generate_sorting(num_units=3, durations=[3.0], seed=0)
    sorting.register_recording(multirecording)
    split = si.SplitSegmentSorting(sorting, multirecording)

    ss.save_split_sorting(split, keep_unit_ids=[sorting.unit_ids[0]], phy_folder=str(phy_folder))

    for seg_id, n_samples in enumerate(r.get_num_samples() for r in recordings):
        spike_times = np.load(phy_folder / f"segment_{seg_id}" / "spike_times.npy")
        unit_ids = np.load(phy_folder / f"segment_{seg_id}" / "spike_templates.npy")
        assert len(spike_times) > 0 and spike_times.max() < n_samples
        assert set(unit_ids) == {sorting.unit_ids[0]}
        assert (phy_folder / f"segment_{seg_id}" / "params.py").exists()
