import numpy as np
import osteoid
import pytest

from axon_tracking import skeletonization as skel

# Parameters as used in notebooks/part3_skeletonization.ipynb
PARAMS = dict(
    upsample=[0.5, 0.5, 0.25],
    sampling_rate=10000 / 0.25,
    el_spacing=17.5 * 0.5,
    ms_cutout=[3.0, 5.0],
    ais_detection="dev",
    buffer_frames=10,
)


def line_path(x0, y0, velocity, n_points):
    """Straight path in um (z in samples) with the given velocity in m/s (== mm/ms)."""
    step_um = velocity * 1e6 / PARAMS["sampling_rate"]
    i = np.arange(n_points)
    return np.column_stack([x0 + step_um * i, np.full(n_points, y0), i]).astype(float)


def index_of(path, paths):
    return next(i for i, p in enumerate(paths) if np.array_equal(p, path))


class TestPerformPathQc:
    def test_keeps_paths_within_velocity_range(self):
        paths = [
            line_path(500, 600, 0.5, 20),  # good
            line_path(500, 900, 5.0, 20),  # too fast
            line_path(900, 600, 0.5, 20),  # good
            line_path(900, 900, 0.5, 3),  # too short for a velocity fit, kept
        ]
        qc_list, r2s, vels, lengths = skel.perform_path_qc(
            paths, PARAMS, window_size=7, min_r2=0.9, vel_range=[0.3, 0.8], min_length=1
        )
        assert [index_of(p, paths) for p in qc_list] == [0, 2, 3]
        assert np.allclose(vels[:2], 0.5)
        assert np.isnan(r2s[2])
        assert lengths == [20, 20, 3]

    def test_selection_does_not_depend_on_last_path(self):
        paths = [line_path(500, 600, 0.5, 20), line_path(500, 900, 5.0, 20)]
        qc_list, _, _, _ = skel.perform_path_qc(paths, PARAMS, vel_range=[0.3, 0.8], min_length=1)
        assert [index_of(p, paths) for p in qc_list] == [0]

    def test_all_paths_filtered(self):
        paths = [line_path(500, 600, 0.5, 20)]
        assert skel.perform_path_qc(paths, PARAMS, window_size=7, min_length=50) == ([], [], [], [])

    def test_no_paths(self):
        assert skel.perform_path_qc([], PARAMS) == ([], [], [], [])


def make_skeleton(vertices, edges):
    return osteoid.Skeleton(np.asarray(vertices, float), np.asarray(edges, np.uint32))


class TestBranchesFromPaths:
    def test_unbranched_skeleton_is_one_branch(self):
        skeleton = make_skeleton([[i, 0, i] for i in range(10)], [[i, i + 1] for i in range(9)])
        branches = skel.branches_from_paths(skeleton)
        assert len(branches) == 1
        assert len(branches[0]) == 10

    def test_branches_split_at_every_branch_point_without_duplicates(self):
        # trunk 0..6, side branch off vertex 2 (7, 8, 9) and off vertex 4 (10, 11, 12)
        vertices = [[i, 0, i] for i in range(7)]
        vertices += [[2, 1, 3], [2, 2, 4], [2, 3, 5], [4, -1, 5], [4, -2, 6], [4, -3, 7]]
        edges = [[i, i + 1] for i in range(6)] + [[2, 7], [7, 8], [8, 9], [4, 10], [10, 11], [11, 12]]
        skeleton = make_skeleton(vertices, edges)
        lookup = {tuple(map(float, v)): i for i, v in enumerate(vertices)}

        branches = skel.branches_from_paths(skeleton)
        segments = {frozenset(lookup[tuple(p)] for p in branch) for branch in branches}
        assert segments == {
            frozenset({0, 1, 2}),
            frozenset({2, 7, 8, 9}),
            frozenset({2, 3, 4}),
            frozenset({4, 5, 6}),
            frozenset({4, 10, 11, 12}),
        }


class TestLocalizeAis:
    @pytest.mark.parametrize("mode, expected", [("time", (5, 5)), ("dev", (20, 20)), ("amp", (20, 20))])
    def test_modes(self, mode, expected):
        template = np.zeros((30, 30, 200))
        template[5, 5, 10] = -1.0  # early, small peak
        template[20, 20, 120] = -2.0  # peak at the expected time (3 ms * 40 kHz)
        _, ais = skel.localize_ais(template, dict(PARAMS, ais_detection=mode))
        assert tuple(int(a) for a in ais[:2]) == expected

    def test_neuron_at_edge_of_array(self):
        template = np.zeros((30, 30, 200))
        template[12, 0, 120] = -2.0  # strongest signal on the first row of electrodes
        template[20, 20, 100] = -0.1  # weak (< 10%) interior peak
        _, ais = skel.localize_ais(template, PARAMS)
        assert tuple(int(a) for a in ais[:2]) == (12, 0)

    def test_no_peak_raises(self):
        with pytest.raises(ValueError, match="No peak"):
            skel.localize_ais(np.zeros((30, 30, 200)), PARAMS)

    def test_dev_prefers_expected_time_over_amplitude(self):
        template = np.zeros((30, 30, 200))
        template[5, 5, 30] = -3.0  # largest, but early
        template[20, 20, 115] = -1.0  # close to the expected time (120)
        _, ais = skel.localize_ais(template, PARAMS)
        assert tuple(int(a) for a in ais[:2]) == (20, 20)


def reference_latency_map(template, start, params):
    """Original (memory hungry) implementation of valid_latency_map."""
    indices_array = np.indices(template.shape) * params["el_spacing"]
    distances = (
        np.sqrt(
            (indices_array[0] - start[0] * params["el_spacing"]) ** 2
            + (indices_array[1] - start[1] * params["el_spacing"]) ** 2
        )
        / 1000000
    )
    delay_mat = np.zeros(distances.shape)
    for z in range(distances.shape[2]):
        delay_mat[:, :, z] = (1 / params["sampling_rate"]) * np.abs((z - start[2]))
    delay_mat[:, :, start[2]] = 1 / params["sampling_rate"]
    return (distances / delay_mat) < params["max_velocity"]


def test_valid_latency_map_matches_reference():
    template = np.zeros((25, 15, 40))
    params = dict(PARAMS, max_velocity=0.8)
    start = np.array([7, 4, 6])
    np.testing.assert_array_equal(
        skel.valid_latency_map(template, start, params),
        reference_latency_map(template, start, params),
    )


class TestSkeletonize:
    @staticmethod
    def bar():
        volume = np.zeros((40, 40, 40), bool)
        volume[5:35, 18:22, 18:22] = True
        return volume

    def test_anisotropy_is_forwarded_and_output_in_voxels(self, monkeypatch):
        calls = []
        original = skel.kimimaro.skeletonize

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(skel.kimimaro, "skeletonize", spy)
        skeleton = skel.skeletonize(self.bar(), dust_threshold=0, n_jobs=1, anisotropy=(10, 10, 10))[1]

        assert calls[0]["anisotropy"] == (10, 10, 10)
        assert skeleton.space == "voxel"
        assert skeleton.vertices[:, 0].min() >= 5 and skeleton.vertices[:, 0].max() < 35

    def test_ais_used_as_extra_target_only_if_inside(self, monkeypatch):
        calls = []
        monkeypatch.setattr(skel.kimimaro, "skeletonize", lambda *a, **k: calls.append(k) or {})
        skel.skeletonize(self.bar(), ais=np.array([10, 20, 20]))
        skel.skeletonize(self.bar(), ais=np.array([0, 0, 0]))
        skel.skeletonize(self.bar())
        assert [c["extra_targets_before"] for c in calls] == [[(10, 20, 20)], [], []]

    def test_tick_threshold_warns(self, monkeypatch):
        monkeypatch.setattr(skel.kimimaro, "skeletonize", lambda *a, **k: {})
        with pytest.warns(UserWarning, match="tick_threshold"):
            skel.skeletonize(self.bar(), tick_threshold=10)


def propagating_template(shape=(40, 20, 60), t0=20, samples_per_electrode=0.5):
    """Template grid (x, y, t) with a negative pulse travelling along x from x=5 (0.7 m/s at 20 kHz)."""
    rng = np.random.default_rng(0)
    x, y, t = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    arrival = t0 + (x - 5) * samples_per_electrode
    pulse = -5 * np.exp(-0.5 * ((t - arrival) / 3) ** 2) * np.exp(-0.5 * ((y - 10) / 2.5) ** 2)
    pulse[x < 5] = 0
    return pulse + 0.02 * rng.standard_normal(shape)


def test_full_skeletonization_runs(tmp_path):
    np.save(tmp_path / "7.npy", propagating_template())
    params = dict(
        upsample=[0.5, 0.5, 0.25],
        sampling_rate=20000 / 0.25,
        el_spacing=17.5 * 0.5,
        ms_cutout=[1.0, 2.0],
        ais_detection="dev",
        buffer_frames=10,
        noise_threshold=None,
        abs_threshold=-0.1,
        max_velocity=1.0,
        max_noise_level=None,
    )
    skel_params = dict(scale=2, const=10, dust_threshold=10, n_jobs=1)
    qc_params = dict(window_size=5, max_duplicate_ratio=0.3, min_r2=0.9, vel_range=[0.3, 1.2], min_length=1)

    qc_skeleton, qc_paths = skel.full_skeletonization(str(tmp_path), 7, params, skel_params, qc_params)

    assert isinstance(qc_skeleton, osteoid.Skeleton)
    assert len(qc_paths) > 0
    x_extent_um = np.ptp(qc_skeleton.vertices[:, 0])
    assert x_extent_um > 10 * 17.5  # the axon spans 35 electrodes along x


def test_qc_paths_of_smoothed_skeleton_can_be_postprocessed():
    """part3 workflow: average_smoothing returns float64 vertices, which kimimaro cannot postprocess."""
    vertices = [[i, 10 + 0.3 * (i % 2), i] for i in range(30)] + [[10, 11 + i, 10 + i] for i in range(1, 15)]
    edges = [[i, i + 1] for i in range(29)] + [[10, 30]] + [[30 + i, 31 + i] for i in range(13)]
    skeleton = make_skeleton(vertices, edges).average_smoothing(n=4, check_boundary=False)
    assert skeleton.vertices.dtype == np.float64

    params = dict(PARAMS, sampling_rate=20000 / 0.25)  # ~0.7 m/s for one voxel per sample
    qc_list, _, _, _ = skel.perform_path_qc(
        skel.branches_from_paths(skeleton), params, window_size=5, min_r2=0.5, vel_range=[0.1, 2], min_length=1
    )
    assert qc_list and all(p.dtype == np.float32 for p in qc_list)

    # as in notebooks/part3_skeletonization.ipynb (cells 18-19)
    qc_skeleton = osteoid.Skeleton.simple_merge([osteoid.Skeleton.from_path(x) for x in qc_list])
    qc_skeleton = skel.kimimaro.join_close_components(qc_skeleton, radius=100)
    qc_skeleton = skel.kimimaro.postprocess(qc_skeleton, dust_threshold=0, tick_threshold=0)
    assert len(qc_skeleton.vertices) > 0
    assert len(skel.paths_to_skeleton(qc_list).vertices) > 0


def test_restore_sparse_template_removes_empty_electrodes():
    template = np.zeros((6, 6, 4))
    template[1, 2, :] = 1
    template[3, 2, :] = 2
    template[1, 4, :] = 3
    restored = skel.restore_sparse_template(template, spacing=(1, 1, 1))
    assert restored.shape == (2, 2, 4)
