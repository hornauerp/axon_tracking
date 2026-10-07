"""End-to-end template extraction on synthetic MaxWell recordings with known spikes."""

import numpy as np
import pytest
import spikeinterface.full as si

import synthetic as syn
from axon_tracking import spike_sorting as ss
from axon_tracking import template_extraction as te

N_SAMPLES = 30000  # 1.5 s per recording

SOMA_0 = syn.electrode_id(20, 10)
SOMA_1 = syn.electrode_id(21, 11)
COMMON = syn.block(range(20, 22), range(10, 12))
AXON_REC0 = syn.block(range(10, 15), [12])  # only recorded in rec0000
AXON_REC1 = syn.block(range(15, 20), [12])  # only recorded in rec0001
REC_ELECTRODES = [
    np.concatenate([COMMON, syn.block(range(10, 15), range(11, 14))]),
    np.concatenate([syn.block(range(15, 20), range(11, 14)), COMMON]),
    np.concatenate([COMMON, syn.block(range(10, 15), range(16, 19))]),  # no axon here
]


def spike_times(seed, n_recs):
    rng = np.random.default_rng(seed)
    base = np.arange(500, N_SAMPLES - 500, 1000)
    return {("well000", r): base + rng.integers(-100, 100, len(base)) for r in range(n_recs)}


UNIT_0 = {  # propagating unit: soma plus axon along row 12, delays increasing towards the soma's left
    "spike_times": spike_times(0, 3),
    "footprint": {
        SOMA_0: (-60, 0),
        **{el: (-20, 2 + i) for i, el in enumerate(AXON_REC0[::-1])},
        **{el: (-20, 2 + i) for i, el in enumerate(AXON_REC1)},
    },
}
UNIT_1 = {"spike_times": spike_times(1, 3), "footprint": {SOMA_1: (-40, 0)}}
UNIT_MUA = {"spike_times": spike_times(2, 3), "footprint": {SOMA_1: (-10, 0)}}


def concatenated_spikes(units, n_recs):
    times, clusters = [], []
    for unit_id, unit in enumerate(units):
        for r in range(n_recs):
            t = unit["spike_times"][("well000", r)] + r * N_SAMPLES
            times.append(t)
            clusters.append(np.full(len(t), unit_id))
    return np.concatenate(times), np.concatenate(clusters)


@pytest.fixture
def axon_scan_sorting(tmp_path):
    """Synthetic axon scan + Kilosort-like output, as produced by spike_sorting.sort_recording_list."""
    rec_path = tmp_path / "data.raw.h5"
    units = [UNIT_0, UNIT_1, UNIT_MUA]
    syn.write_maxwell_file(rec_path, {"well000": REC_ELECTRODES}, N_SAMPLES, units=units)

    multirecording, common_el, _ = ss.concatenate_recording_slices(str(rec_path), "well000")
    sorting_root = tmp_path / "sorting" / "well000"
    sorter_output = sorting_root / "sorter_output"
    times, clusters = concatenated_spikes(units, 3)
    syn.write_kilosort_output(
        sorter_output, times, clusters, len(common_el), ks_labels={0: "good", 1: "good", 2: "mua"}
    )
    multirecording.dump_to_json(sorting_root / "spikeinterface_recording.json")
    return sorter_output


QC = dict(min_n_spikes=5, exclude_mua=True, use_bc=False, use_si=False)
TE = dict(n_jobs=1, filter_band=150, overwrite=True, max_spikes_per_unit=100)


def trough(grid, electrode):
    col, row = electrode % 220, electrode // 220
    return grid[col, row].min(), grid[col, row].argmin()


def test_extract_templates_axon_scan(axon_scan_sorting):
    te.extract_templates_from_sorting_list([str(axon_scan_sorting)], QC, TE, raise_errors=True)

    template_folder = axon_scan_sorting / "templates"
    assert sorted(p.name for p in template_folder.iterdir()) == ["0.npy", "1.npy"]  # MUA excluded

    grid = np.load(template_folder / "0.npy")
    assert grid.shape[2] == sum(syn.TRIGGER)  # cutout from trigger_pre/post

    # strongest signal at the soma; axon visible on electrodes recorded in only one configuration
    min_map = grid.min(axis=2)
    assert np.unravel_index(np.argmin(min_map), min_map.shape) == (SOMA_0 % 220, SOMA_0 // 220)
    soma_amp, soma_t = trough(grid, SOMA_0)
    for el in (AXON_REC0[0], AXON_REC1[-1]):
        amp, t = trough(grid, el)
        assert amp < 0.15 * soma_amp  # clearly above noise (soma_amp < 0)
        assert t > soma_t  # axonal signal arrives later
    # electrodes of rec0002 carry no signal of unit 0
    assert trough(grid, REC_ELECTRODES[2][-1])[0] > 0.1 * soma_amp

    grid_1 = np.load(template_folder / "1.npy")
    min_map_1 = grid_1.min(axis=2)
    assert np.unravel_index(np.argmin(min_map_1), min_map_1.shape) == (SOMA_1 % 220, SOMA_1 // 220)


PARTNER_EL = syn.electrode_id(12, 13)  # recorded only in rec0000, no signal of unit 0


@pytest.fixture
def locked_partner_sorting(tmp_path):
    """Unit 0 plus a 'MUA' unit that fires 1 ms after 90% of unit 0's spikes (as in a network burst)."""
    rng = np.random.default_rng(5)
    partner_times = {k: np.sort(t[rng.random(len(t)) < 0.9] + 20) for k, t in UNIT_0["spike_times"].items()}
    partner = {"spike_times": partner_times, "footprint": {PARTNER_EL: (-40, 0)}}
    units = [UNIT_0, partner]
    rec_path = tmp_path / "data.raw.h5"
    syn.write_maxwell_file(rec_path, {"well000": REC_ELECTRODES}, N_SAMPLES, units=units)
    multirecording, common_el, _ = ss.concatenate_recording_slices(str(rec_path), "well000")
    sorting_root = tmp_path / "sorting" / "well000"
    times, clusters = concatenated_spikes(units, 3)
    syn.write_kilosort_output(sorting_root / "sorter_output", times, clusters, len(common_el),
                              ks_labels={0: "good", 1: "mua"})
    multirecording.dump_to_json(sorting_root / "spikeinterface_recording.json")
    return sorting_root / "sorter_output"


def test_deconvolved_templates_remove_a_locked_partner(locked_partner_sorting):
    out = locked_partner_sorting
    te.extract_templates_from_sorting_list([str(out)], QC, TE, raise_errors=True)
    te.extract_templates_from_sorting_list([str(out)], QC, dict(TE, deconvolve=True), raise_errors=True)
    average = np.load(out / "templates" / "0.npy")
    deconvolved = np.load(out / "templates_deconvolved" / "0.npy")
    assert sorted(p.name for p in (out / "templates_deconvolved").iterdir()) == ["0.npy"]  # MUA: regressor only
    assert average.shape == deconvolved.shape

    soma_avg, soma_dec = trough(average, SOMA_0)[0], trough(deconvolved, SOMA_0)[0]
    assert abs(soma_dec - soma_avg) < 0.05 * abs(soma_avg)  # own footprint unchanged
    partner_avg, partner_dec = trough(average, PARTNER_EL)[0], trough(deconvolved, PARTNER_EL)[0]
    assert partner_avg < 0.4 * soma_avg  # the average contains the partner (~90% of its -40 bits)
    assert partner_dec > 0.1 * partner_avg  # removed down to the noise
    # with only the good units as regressors the partner cannot be removed
    te.extract_templates_from_sorting_list([str(out)], QC, dict(TE, deconvolve=True, deconvolve_units="good"),
                                           raise_errors=True)
    good_only = np.load(out / "templates_deconvolved" / "0.npy")
    assert trough(good_only, PARTNER_EL)[0] < 0.4 * soma_avg


def test_deconvolution_uses_rejected_units_with_spikeinterface_qc(locked_partner_sorting):
    """With the spikeinterface QC (merges / redundant units), units rejected before it remain regressors."""
    out = locked_partner_sorting
    qc = dict(QC, use_si=True, auto_merge=True, remove_redundant=True)
    te.extract_templates_from_sorting_list([str(out)], qc, dict(TE, deconvolve=True), raise_errors=True)
    deconvolved = np.load(out / "templates_deconvolved" / "0.npy")
    soma = trough(deconvolved, SOMA_0)[0]
    assert trough(deconvolved, PARTNER_EL)[0] > 0.1 * soma


@pytest.fixture
def unsorted_partner_with_network(tmp_path):
    """The partner fires with unit 0 but is not in the axon scan sorting; a network sorting of the same well has
    templates of the partner and of unit 0 (a duplicate of the sorted unit)."""
    rng = np.random.default_rng(5)
    partner_times = {k: np.sort(t[rng.random(len(t)) < 0.9] + 20) for k, t in UNIT_0["spike_times"].items()}
    partner = {"spike_times": partner_times, "footprint": {PARTNER_EL: (-40, 0)}}
    rec_path = tmp_path / "data.raw.h5"
    syn.write_maxwell_file(rec_path, {"well000": REC_ELECTRODES}, N_SAMPLES, units=[UNIT_0, partner])
    multirecording, common_el, _ = ss.concatenate_recording_slices(str(rec_path), "well000")
    sorting_root = tmp_path / "sorting" / "well000"
    times, clusters = concatenated_spikes([UNIT_0], 3)
    syn.write_kilosort_output(sorting_root / "sorter_output", times, clusters, len(common_el), ks_labels={0: "good"})
    multirecording.dump_to_json(sorting_root / "spikeinterface_recording.json")

    network_electrodes = np.array([PARTNER_EL, PARTNER_EL - 1, PARTNER_EL + 1, SOMA_0, SOMA_0 + 1,
                                   syn.electrode_id(30, 30)])
    t = np.arange(30) - 10
    shape = np.exp(-0.5 * (t / 3.0) ** 2) * syn.LSB * 1e6  # uV per bit
    templates = np.zeros((2, 30, len(network_electrodes)))
    templates[0, :, 0] = -40 * shape  # partner
    templates[1, :, 3] = -60 * shape  # unit 0
    qc = tmp_path / "network" / "well000" / "sorter_output" / "qc_output"
    qc.mkdir(parents=True)
    np.save(qc / "raw_templates.npy", templates)
    np.save(qc / "phy_ids.npy", np.array([10, 11]))
    np.save(qc / "channel_positions.npy", np.column_stack(syn.electrode_xy(network_electrodes)))
    (qc / "params.py").write_text(f"sample_rate = {float(syn.FS)}\n")
    return sorting_root / "sorter_output", qc.parent


def test_network_units_remove_an_unsorted_partner(unsorted_partner_with_network, tmp_path):
    out, network = unsorted_partner_with_network
    te.extract_templates_from_sorting_list([str(out)], QC, TE, raise_errors=True)
    te.extract_templates_from_sorting_list([str(out)], QC, dict(TE, deconvolve=True), raise_errors=True,
                                           save_folders=[tmp_path / "without"])
    te.extract_templates_from_sorting_list([str(out)], QC, dict(TE, deconvolve=True), raise_errors=True,
                                           save_folders=[tmp_path / "with"], network_sortings=[str(network)])
    average = np.load(out / "templates" / "0.npy")
    without = np.load(tmp_path / "without" / "0.npy")
    with_network = np.load(tmp_path / "with" / "0.npy")
    soma = trough(average, SOMA_0)[0]
    assert trough(average, PARTNER_EL)[0] < 0.4 * soma  # contaminated
    assert trough(without, PARTNER_EL)[0] < 0.4 * soma  # not sorted: cannot be removed
    assert trough(with_network, PARTNER_EL)[0] > 0.1 * soma  # detected by its network template and removed
    # the network duplicate of unit 0 was not used (unit 0's own footprint unchanged)
    assert abs(trough(with_network, SOMA_0)[0] - soma) < 0.05 * abs(soma)


def test_deconvolution_from_saved_analyzers_matches_raw_path(locked_partner_sorting, tmp_path):
    """Per-configuration analyzers with the averages of all spikes give the same deconvolved templates as the raw
    data path (same window, same regressors)."""
    out = locked_partner_sorting
    qc = dict(QC, exclude_mua=False)  # both units are outputs and regressors
    te.extract_templates_from_sorting_list([str(out)], qc, dict(TE, deconvolve=True, deconvolve_pad_ms=(0.0, 0.0),
                                                                deconvolve_units="good"), raise_errors=True)
    segment_sorting = te.preprocess_sorting(str(out), qc)
    source = te.ut.find_source_recording(segment_sorting)
    _, cutout_ms = te.ut.get_cutout_info(source._kwargs["file_path"], stream_id="well000")
    rec_names, _, _ = ss.find_common_electrodes(source._kwargs["file_path"], "well000")
    folders = []
    for k, rec_name in enumerate(rec_names):
        rec = te.filter_recording(si.read_maxwell(source._kwargs["file_path"], stream_id="well000", rec_name=rec_name), 150)
        seg = si.SelectSegmentSorting(segment_sorting, k)
        seg = si.remove_excess_spikes(seg, rec)
        folder = tmp_path / "spikes" / f"seg_{k:02d}.zarr"
        analyzer = si.create_sorting_analyzer(seg, rec, format="zarr", folder=folder, sparse=False)
        analyzer.compute({"random_spikes": {"method": "all"},
                          "templates": {"ms_before": cutout_ms[0], "ms_after": cutout_ms[1]}})
        folders.append(folder)
    unit_ids, _, dec = te.deconvolve_saved_analyzers(folders, save_folder=tmp_path / "from_analyzers")
    for k, u in enumerate(unit_ids):
        raw_path = np.load(out / "templates_deconvolved" / f"{u}.npy")  # cropped to the recorded electrodes
        nx, ny, _ = raw_path.shape
        assert np.abs(dec[k][:nx, :ny] - raw_path).max() < 0.02 * np.abs(raw_path).max()
        assert not dec[k][nx:].any() and not dec[k][:, ny:].any()
    partner = np.load(tmp_path / "from_analyzers" / "templates_deconvolved" / "0.npy")
    assert trough(partner, PARTNER_EL)[0] > 0.1 * trough(partner, SOMA_0)[0]


def test_remapped_raw_path_and_separate_output_folder(axon_scan_sorting, tmp_path):
    """Raw data moved (e.g. sorted on Linux, extracted on Windows) and output written outside the sorting."""
    import json

    from axon_tracking import utils as ut

    json_path = axon_scan_sorting.parent / "spikeinterface_recording.json"
    d = json.loads(json_path.read_text())
    moved = {str(tmp_path): "/net/server/export/group/recordings"}
    def move(o):  # store the raw data path as it would appear on another machine
        if isinstance(o, dict):
            return {k: ut.remap_path(v, moved) if k == "file_path" else move(v) for k, v in o.items()}
        return [move(v) for v in o] if isinstance(o, list) else o
    json_path.write_text(json.dumps(move(d)))
    assert "/net/server/export/group/recordings/data.raw.h5" in json_path.read_text()
    before = sorted(p.name for p in axon_scan_sorting.iterdir())
    out = tmp_path / "elsewhere" / "well000"
    te.extract_templates_from_sorting_list([str(axon_scan_sorting)], QC, dict(TE, deconvolve=True), raise_errors=True,
                                           save_folders=[out], path_map={"/net/server/export/group/recordings": tmp_path})
    assert sorted(p.name for p in out.iterdir()) == ["0.npy", "1.npy"]
    assert sorted(p.name for p in axon_scan_sorting.iterdir()) == before  # sorting folder untouched


def test_existing_templates_are_skipped(axon_scan_sorting, capsys):
    te.extract_templates_from_sorting_list([str(axon_scan_sorting)], QC, TE, raise_errors=True)
    te.extract_templates_from_sorting_list(
        [str(axon_scan_sorting)], QC, dict(TE, overwrite=False), raise_errors=True
    )
    assert "Templates already extracted" in capsys.readouterr().out


def test_extract_templates_network_recording(tmp_path):
    """Single (non-concatenated) recording without trigger information."""
    rec_path = tmp_path / "network.raw.h5"
    electrodes = np.concatenate([COMMON, syn.block(range(10, 15), range(11, 14))])
    units = [UNIT_0, UNIT_1]
    syn.write_maxwell_file(rec_path, {"well000": [electrodes]}, N_SAMPLES, units=units, trigger=None)

    sorting_root = tmp_path / "sorting" / "well000"
    times, clusters = concatenated_spikes(units, 1)
    syn.write_kilosort_output(sorting_root / "sorter_output", times, clusters, len(electrodes), {0: "good", 1: "good"})
    si.read_maxwell(str(rec_path), stream_id="well000").dump_to_json(sorting_root / "spikeinterface_recording.json")

    te.extract_templates_from_sorting_list([str(sorting_root / "sorter_output")], QC, TE, raise_errors=True)

    grid = np.load(sorting_root / "sorter_output" / "templates" / "0.npy")
    assert grid.shape[2] == 130  # default cutout of 1.5 + 5 ms at 20 kHz
    min_map = grid.min(axis=2)
    assert np.unravel_index(np.argmin(min_map), min_map.shape) == (SOMA_0 % 220, SOMA_0 // 220)


def test_errors_are_reported_and_skipped(tmp_path, capsys):
    te.extract_templates_from_sorting_list([str(tmp_path / "missing" / "sorter_output")], QC, TE)
    assert "Error in" in capsys.readouterr().out


class TestSelectGoodUnits:
    @staticmethod
    def sorting():
        sorting = si.generate_sorting(num_units=3, durations=[10.0], firing_rates=10.0, seed=0)
        sorting.set_property("KSLabel", np.array(["good", "mua", "good"]))
        return sorting

    def test_ks_labels_and_spike_count(self):
        cleaned = te.select_good_units(self.sorting(), None, dict(QC, min_n_spikes=5))
        assert list(cleaned.unit_ids) == list(self.sorting().unit_ids[[0, 2]])

    def test_missing_bombcell_output_warns_and_keeps_units(self):
        with pytest.warns(UserWarning, match="bombcell"):
            cleaned = te.select_good_units(self.sorting(), None, dict(QC, use_bc=True, exclude_mua=False))
        assert cleaned.get_num_units() == 3

    def test_bombcell_labels(self):
        sorting = self.sorting()
        sorting.set_property("bc_unitType", np.array(["GOOD", "GOOD", "MUA"]))
        cleaned = te.select_good_units(sorting, None, dict(QC, use_bc=True, exclude_mua=False))
        assert list(cleaned.unit_ids) == list(sorting.unit_ids[:2])


class TestSpikeinterfaceQc:
    @staticmethod
    def data():
        recording, sorting = si.generate_ground_truth_recording(
            durations=[10.0], num_channels=8, num_units=4, seed=0
        )
        return recording, sorting

    def test_runs_with_real_curation(self):
        recording, sorting = self.data()
        cleaned = te.perform_si_qc(sorting, recording, dict(auto_merge=True, remove_redundant=True))
        assert 0 < cleaned.get_num_units() <= sorting.get_num_units()

    def test_merges_are_kept_without_redundant_unit_removal(self, monkeypatch):
        recording, sorting = self.data()
        merged = sorting.select_units(sorting.unit_ids[:2])

        class MergedAnalyzer:
            pass

        def fake_auto_merge(analyzer, presets):
            result = MergedAnalyzer()
            result.sorting = merged
            return result

        monkeypatch.setattr(te.si, "auto_merge_units", fake_auto_merge)
        cleaned = te.perform_si_qc(sorting, recording, dict(auto_merge=True, remove_redundant=False))
        assert cleaned is merged


def test_averaged_templates_ignore_main_channel_of_the_sorted_recording(tmp_path):
    """Kilosort4 sortings carry main_channel_id for the concatenated recording (other channel ids)."""
    rec_path = tmp_path / "data.raw.h5"
    syn.write_maxwell_file(rec_path, {"well000": REC_ELECTRODES[:1]}, N_SAMPLES, units=[UNIT_0])
    rec = si.read_maxwell(str(rec_path), stream_id="well000", rec_name="rec0000")
    sorting = si.NumpySorting.from_unit_dict([{0: UNIT_0["spike_times"][("well000", 0)]}], syn.FS)
    sorting.set_property("main_channel_id", np.array([9999]))
    templates = te.extract_templates(rec, sorting, dict(te.DEFAULT_TE_PARAMS, **TE), (1.0, 2.0))
    assert templates.shape[0] == 1 and np.isfinite(templates).all()


def test_network_templates_from_a_kilosort_sorting(tmp_path):
    from axon_tracking import network_units as nu

    rec_path = tmp_path / "network.raw.h5"
    syn.write_maxwell_file(rec_path, {"well000": REC_ELECTRODES[:1]}, N_SAMPLES, units=[UNIT_0, UNIT_1], trigger=None)
    rec = si.read_maxwell(str(rec_path), stream_id="well000", rec_name="rec0000")
    root = tmp_path / "sorting" / "well000"
    times = np.concatenate([UNIT_0["spike_times"][("well000", 0)], UNIT_1["spike_times"][("well000", 0)]])
    clusters = np.r_[np.zeros(len(UNIT_0["spike_times"][("well000", 0)]), int),
                     np.ones(len(UNIT_1["spike_times"][("well000", 0)]), int)]
    syn.write_kilosort_output(root / "sorter_output", times, clusters, rec.get_num_channels(),
                              ks_labels={0: "good", 1: "mua"})
    rec.dump_to_json(root / "spikeinterface_recording.json")
    net = nu.network_templates_from_sorting(str(root / "sorter_output"), ms_before=1.0, ms_after=1.5)
    assert list(net.unit_ids) == [0]  # good units only
    assert net.soma_electrodes[0] == SOMA_0
    assert (root / "sorter_output" / "network_templates.npz").exists()
    again = nu.network_templates_from_sorting(str(root / "sorter_output"))
    np.testing.assert_array_equal(again.templates, net.templates)
