import numpy as np

from axon_tracking import graph_reconstruction as gr


def _template(rng, noise_uv=1.0):
    """Soma at (50, 60), an axon along row 60 to column 90 (0.25 m/s), and a foreign soma blob at (120, 30)."""
    fs, t0, n = 10000.0, 30, 80
    t = np.arange(n)
    grid = rng.normal(0, noise_uv, (220, 120, n))
    spike = lambda amp, at: -amp * np.exp(-0.5 * ((t - at) / 1.5) ** 2) + 0.3 * amp * np.exp(-0.5 * ((t - at - 5) / 3) ** 2)
    grid[50, 60] += spike(200, t0)
    for k, col in enumerate(range(51, 91)):
        grid[col, 60] += spike(25, t0 + 1 + k * 0.7)  # 17.5 um per 0.07 ms = 0.25 m/s
    for dc in range(-2, 3):
        for dr in range(-2, 3):
            grid[120 + dc, 30 + dr] += spike(60 / (1 + dc ** 2 + dr ** 2), t0 + 20)
    return grid, fs, t0


def test_tree_follows_the_propagating_axon_and_ignores_static_blobs():
    grid, fs, t0 = _template(np.random.default_rng(0))
    paths = gr.reconstruct(grid, fs, t0, dendrites="none")
    pts = np.concatenate([p[:, :2] for p in paths])
    axon_y = 60 * 17.5
    # stays on the axon (long edges may attach an occasional noise electrode)
    assert np.mean(np.abs(pts[:, 1] - axon_y) <= 2 * 17.5) >= 0.95
    assert pts[:, 0].max() >= 85 * 17.5  # reaches (nearly) its end
    assert not np.any(np.linalg.norm(pts - [120 * 17.5, 30 * 17.5], axis=1) < 100)  # the blob is not reached
    lat = np.concatenate([p[:, 2] for p in paths])
    assert np.corrcoef(pts[:, 0], lat)[0, 1] > 0.9  # latencies increase along the axon


def test_empty_template_gives_no_paths():
    grid = np.random.default_rng(1).normal(0, 1.0, (220, 120, 80))
    assert gr.reconstruct(grid, 10000.0, 30, z_min=8.0) == []


def test_decoy_threshold_rises_with_heavy_tailed_noise_and_keeps_the_axon():
    rng = np.random.default_rng(2)
    grid, fs, t0 = _template(rng)
    assert gr.decoy_threshold(grid, fs, t0, fdr=0.1) <= 6.5  # white noise: ~5 for this sparse axon
    # residual spikes of other neurons at random times and places (heavy tails, before and after the trigger)
    t = np.arange(grid.shape[2])
    for _ in range(3000):
        c, r, at = rng.integers(0, 220), rng.integers(0, 120), rng.uniform(0, grid.shape[2])
        grid[c, r] -= rng.uniform(4, 15) * np.exp(-0.5 * ((t - at) / 1.5) ** 2)
    tau = gr.decoy_threshold(grid, fs, t0, fdr=0.1)
    assert tau > 8.0
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", node_fdr=0.1)
    pts = np.concatenate([p[:, :2] for p in paths])
    assert np.mean(np.abs(pts[:, 1] - 60 * 17.5) <= 2 * 17.5) >= 0.95
    assert pts[:, 0].max() >= 85 * 17.5


def test_null_template_as_decoy():
    rng = np.random.default_rng(3)
    grid, fs, t0 = _template(rng)
    t = np.arange(grid.shape[2])
    null = rng.normal(0, 1.0, grid.shape)
    for g in (grid, null):  # the same heavy-tailed background in the template and its null
        for _ in range(3000):
            c, r, at = rng.integers(0, 220), rng.integers(0, 120), rng.uniform(0, grid.shape[2])
            g[c, r] -= rng.uniform(4, 15) * np.exp(-0.5 * ((t - at) / 1.5) ** 2)
    tau = gr.decoy_threshold(grid, fs, t0, fdr=0.1, null_grid=null)
    assert tau > 8.0
    assert gr.decoy_threshold(grid, fs, t0, fdr=0.1, null_grid=rng.normal(0, 1.0, grid.shape)) < tau
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", null_grid=null, node_fdr=0.5)
    pts = np.concatenate([p[:, :2] for p in paths])
    assert np.mean(np.abs(pts[:, 1] - 60 * 17.5) <= 2 * 17.5) >= 0.95 and pts[:, 0].max() >= 85 * 17.5


def test_zero_lag_copies_are_removed_far_from_the_soma():
    rng = np.random.default_rng(4)
    grid, fs, t0 = _template(rng)
    soma = grid[50, 60].copy()
    for c, r in ((50, 90), (80, 20), (60, 75)):  # 525, 700 and 320 um from the soma
        grid[c, r] += 0.05 * soma
    clean = gr.remove_zero_lag_copies(grid, t0, fs, min_distance_um=250)
    for c, r in ((50, 90), (80, 20), (60, 75)):
        assert np.abs(clean[c, r]).max() < 4.5  # back to noise (sd 1)
    np.testing.assert_array_equal(clean[50, 60], grid[50, 60])  # soma untouched
    axon = clean[85, 60]  # 612 um away, delayed by ~24 samples: kept
    assert axon.min() < -20


def test_spatial_smoothing_keeps_unrecorded_electrodes_empty():
    grid = np.random.default_rng(5).normal(0, 1, (220, 120, 20))
    grid[:10] = 0
    out = gr.spatial_smooth(grid, 0.7)
    assert np.all(out[:10] == 0) and out[10:].std() < grid[10:].std()


def test_path_linearity_pruning_keeps_the_axon():
    grid, fs, t0 = _template(np.random.default_rng(6))
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", min_path_r2=0.8)
    pts = np.concatenate([p[:, :2] for p in paths])
    assert pts[:, 0].max() >= 85 * 17.5 and np.mean(np.abs(pts[:, 1] - 60 * 17.5) <= 2 * 17.5) >= 0.95


def test_decoy_threshold_with_hysteresis_follows_the_axon():
    grid, fs, t0 = _template(np.random.default_rng(9))
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", z_min=4.0, node_fdr=0.5, weak_edge_um=25.0,
                           smooth_sigma=None, ridge_refine=False)
    pts = np.concatenate([p[:, :2] for p in paths])
    assert pts[:, 0].max() >= 85 * 17.5 and np.mean(np.abs(pts[:, 1] - 60 * 17.5) <= 2 * 17.5) >= 0.9


def test_soma_region_is_not_returned_as_axon():
    grid, fs, t0 = _template(np.random.default_rng(10))
    t = np.arange(grid.shape[2])
    for dc in range(-4, 5):  # a large somatic/dendritic footprint around the soma (50, 60), firing with it
        for dr in range(-4, 5):
            if dc or dr:
                grid[50 + dc, 60 + dr] += -80 / (1 + 0.3 * (dc ** 2 + dr ** 2)) * np.exp(-0.5 * ((t - t0) / 1.5) ** 2)
    plain = gr.reconstruct(grid, fs, t0, dendrites="none")
    region = gr.reconstruct(grid, fs, t0, dendrites="none", soma_region_ms=0.15)
    def inside(paths):
        pts = np.concatenate([p[:, :2] for p in paths])
        return np.mean(np.hypot(pts[:, 0] - 50 * 17.5, pts[:, 1] - 60 * 17.5) <= 4 * 17.5)
    assert inside(region) < inside(plain)
    pts = np.concatenate([p[:, :2] for p in region])
    assert pts[:, 0].max() >= 85 * 17.5  # the axon is still followed


def test_spur_pruning_and_centerline_cable():
    grid, fs, t0 = _template(np.random.default_rng(11))
    plain = gr.reconstruct(grid, fs, t0, dendrites="none", z_min=3.5, node_fdr=None, weak_edge_um=None)
    pruned = gr.reconstruct(grid, fs, t0, dendrites="none", z_min=3.5, node_fdr=None, weak_edge_um=None, min_spur_nodes=3)
    assert len(pruned) <= len(plain)
    pts = np.concatenate([p[:, :2] for p in pruned])
    assert pts[:, 0].max() >= 85 * 17.5
    # a zigzag along a straight line: centre-line cable close to the straight length
    zig = np.column_stack([np.arange(0, 41) * 17.5, (np.arange(41) % 2) * 17.5])
    assert gr.centerline_cable([zig]) < 0.8 * np.linalg.norm(np.diff(zig, axis=0), axis=1).sum()
    assert abs(gr.centerline_cable([zig]) - 40 * 17.5) < 0.1 * 40 * 17.5


def test_auto_spur_length_grows_with_noise():
    assert gr.auto_spur_nodes(500, 0.001) < gr.auto_spur_nodes(500, 0.02) <= gr.auto_spur_nodes(500, 0.05)
    grid, fs, t0 = _template(np.random.default_rng(12))
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", min_spur_nodes="auto")
    assert np.concatenate([p[:, :2] for p in paths])[:, 0].max() >= 85 * 17.5


def test_width_spurs_keep_a_propagating_collateral():
    rng = np.random.default_rng(13)
    grid, fs, t0 = _template(rng)
    t = np.arange(grid.shape[2])
    spike = lambda amp, at: -amp * np.exp(-0.5 * ((t - at) / 1.5) ** 2)
    # a thick axon: the footprint along row 60 spreads over +-2 rows
    for k, col in enumerate(range(51, 91)):
        for dr in (-2, -1, 1, 2):
            grid[col, 60 + dr] += spike(25 / (1 + dr ** 2), t0 + 1 + k * 0.7)
    # a short propagating collateral leaving at column 70 upwards (rows 59..54), later than the main axon
    for k, row in enumerate(range(57, 51, -1)):
        grid[70, row] += spike(20, t0 + 1 + 19 * 0.7 + (k + 1) * 0.7)
    paths = gr.reconstruct(grid, fs, t0, dendrites="none", width_spurs=1.5)
    pts = np.concatenate([p[:, :2] for p in paths])
    assert np.any((np.abs(pts[:, 0] - 70 * 17.5) <= 17.5) & (pts[:, 1] <= 53 * 17.5))  # the collateral is kept
    plain = gr.reconstruct(grid, fs, t0, dendrites="none")
    assert len(paths) <= len(plain)  # spurs through the footprint's width are removed


def test_ridge_refinement_centres_paths_on_a_wide_footprint():
    rng = np.random.default_rng(14)
    grid, fs, t0 = _template(rng)
    t = np.arange(grid.shape[2])
    spike = lambda amp, at: -amp * np.exp(-0.5 * ((t - at) / 1.5) ** 2)
    for k, col in enumerate(range(51, 91)):  # a wide footprint centred on row 60 (+-3 rows)
        for dr in (-3, -2, -1, 1, 2, 3):
            grid[col, 60 + dr] += spike(25 * np.exp(-0.5 * (dr / 1.6) ** 2), t0 + 1 + k * 0.7)
    def mean_offset(paths):
        pts = np.concatenate([p[:, :2] for p in paths])
        on = pts[:, 0] > 55 * 17.5
        return np.mean(np.abs(pts[on, 1] - 60 * 17.5))
    plain = gr.reconstruct(grid, fs, t0, dendrites="none", ridge_refine=False)
    ridge = gr.reconstruct(grid, fs, t0, dendrites="none", ridge_refine=True)
    assert mean_offset(ridge) < mean_offset(plain) and mean_offset(ridge) < 10.0


def test_bridging_reaches_a_distal_fragment_beyond_a_gap():
    rng = np.random.default_rng(15)
    fs, t0, n = 10000.0, 30, 80
    t = np.arange(n)
    grid = rng.normal(0, 1.0, (220, 120, n))
    spike = lambda amp, at: -amp * np.exp(-0.5 * ((t - at) / 1.5) ** 2)
    grid[50, 60] += spike(200, t0)
    for k, col in enumerate(range(51, 71)):  # axon to column 70 at 0.25 m/s (0.7 samples per electrode)
        grid[col, 60] += spike(25, t0 + 1 + k * 0.7)
    for k, col in enumerate(range(85, 100)):  # gap of 14 electrodes (245 um), then the axon continues
        grid[col, 60] += spike(25, t0 + 1 + (col - 51) * 0.7)
    plain = gr.reconstruct(grid, fs, t0, dendrites="none")
    bridged = gr.reconstruct(grid, fs, t0, dendrites="none", bridge_um=300)
    reach = lambda paths: np.concatenate([p[:, 0] for p in paths]).max()
    assert reach(plain) < 80 * 17.5 and reach(bridged) >= 95 * 17.5
    # a fragment that propagates towards the tree (later at the near end) is not attached
    grid2 = grid.copy()
    for col in range(85, 100):
        grid2[col, 60] -= spike(25, t0 + 1 + (col - 51) * 0.7)
        grid2[col, 60] += spike(25, t0 + 1 + (133 - col - 51) * 0.7)
    assert reach(gr.reconstruct(grid2, fs, t0, dendrites="none", bridge_um=300)) < 80 * 17.5
