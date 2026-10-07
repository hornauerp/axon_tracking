from pathlib import Path

import h5py
import numpy as np
import pytest
import spikeinterface.full as si

from axon_tracking import utils as ut

PARAMS = dict(upsample=[0.5, 0.5, 0.25], el_spacing=17.5 * 0.5)


def write_trigger_file(path, wells):
    """wells: {well_name: [(trigger_pre, trigger_post) or None per recording]}"""
    with h5py.File(path, "w") as h5:
        for well, recs in wells.items():
            for rec_index, trigger in enumerate(recs):
                rec = h5.create_group(f"wells/{well}/rec{rec_index:04d}")
                rec["settings/sampling"] = [20000.0]
                rec.create_group("groups/routed")
                if trigger is not None:
                    rec["groups/routed/trigger_pre"] = [trigger[0]]
                    rec["groups/routed/trigger_post"] = [trigger[1]]


class TestGetCutoutInfo:
    def test_falls_back_to_next_well(self, tmp_path):
        path = tmp_path / "rec.h5"
        write_trigger_file(path, {"well000": [(20, -5)], "well001": [(20, 40)]})
        assert ut.get_cutout_info(path) == ([20, 40], [1.0, 2.0])

    def test_prefers_given_stream(self, tmp_path):
        path = tmp_path / "rec.h5"
        write_trigger_file(path, {"well000": [(20, 40)], "well001": [(30, 60), (30, 60)]})
        assert ut.get_cutout_info(path, stream_id="well001") == ([30, 60], [1.5, 3.0])

    def test_network_recording_default(self, tmp_path):
        path = tmp_path / "rec.h5"
        write_trigger_file(path, {"well000": [None], "well001": [None]})
        assert ut.get_cutout_info(path) == ([30, 100], [1.5, 5.0])


class TestScale:
    def test_um_paths_near_top_of_chip_are_detected(self):
        path = np.array([[1500.0, 50.0, 0], [2500.0, 300.0, 10], [3000.0, 350.0, 20]])
        assert ut.check_if_scaled([path], PARAMS)
        assert ut.convert_coor_scale([path], PARAMS, "um")[0] is path

    def test_electrode_unit_paths(self):
        path = np.array([[10.0, 5.0, 0], [430.0, 230.0, 10]])
        assert not ut.check_if_scaled([path], PARAMS)
        scaled = ut.convert_coor_scale([path], PARAMS, "um")[0]
        np.testing.assert_allclose(scaled[:, :2], path[:, :2] * 8.75)
        np.testing.assert_allclose(scaled[:, 2], path[:, 2])

    def test_explicit_current_scale(self):
        path = np.array([[10.0, 5.0, 0], [20.0, 10.0, 10]])  # um, ambiguous for the heuristic
        assert ut.convert_coor_scale([path], PARAMS, "um", current_scale="um")[0] is path
        np.testing.assert_allclose(
            ut.convert_coor_scale([path], PARAMS, "el", current_scale="um")[0][:, :2], path[:, :2] / 8.75
        )


def test_infer_stream_id():
    assert ut.infer_stream_id("/data/exp/AxonTracking/well012/sorter_output") == "well012"
    with pytest.raises(ValueError):
        ut.infer_stream_id("/data/exp/sorter_output")


def test_find_successful_sortings(tmp_path):
    rec_path = "/recordings/exp1/data.raw.h5"
    save_path_changes = {"pos": [0, 1], "vals": [str(tmp_path), ""]}  # /recordings -> tmp_path
    sorting_dir = tmp_path / "exp1" / "data.raw.h5" / "well000" / "sorter_output"
    sorting_dir.mkdir(parents=True)
    np.save(sorting_dir / "templates.npy", np.zeros(1))
    result = ut.find_successful_sortings([rec_path], save_path_changes)
    assert [Path(p) for p in result[rec_path]] == [sorting_dir]


def test_find_source_recording_through_processing_chain():
    source = si.generate_recording(num_channels=4, durations=[1.0, 1.0], seed=0)
    source._kwargs.update(file_path="/data/rec.raw.h5", stream_id="well003")  # mimic a MaxWell recording
    segments = [
        si.select_segment_recording(source, segment_indices=[i]) for i in range(2)
    ]
    sliced = [
        si.center(seg).select_channels(seg.channel_ids[:3]).rename_channels([0, 1, 2]) for seg in segments
    ]
    multirecording = si.concatenate_recordings(sliced)

    sorting = si.generate_sorting(num_units=3, durations=[2.0], seed=0)
    sorting = si.remove_excess_spikes(sorting.select_units(sorting.unit_ids[:2]), multirecording)
    sorting.register_recording(multirecording)
    split = si.SplitSegmentSorting(sorting, multirecording)

    assert ut.find_source_recording(split) is source
    assert ut.get_sorting_stream_id(split) == "well003"
    with pytest.raises(ValueError):
        ut.find_source_recording(si.generate_sorting(num_units=1, durations=[1.0]))
