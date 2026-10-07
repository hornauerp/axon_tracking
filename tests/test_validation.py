import json

import numpy as np

from axon_tracking.validation import dandi_000294 as dandi


def test_register_morphology_recovers_rotation_mirror_and_translation(tmp_path):
    rng = np.random.default_rng(0)
    # recorded block (MEA frame) and a smooth amplitude bump around the AIS
    cols, rows = np.meshgrid(np.arange(100, 130), np.arange(60, 90), indexing="ij")
    ids = (rows * 220 + cols).ravel()
    mea = np.column_stack([ids % 220, ids // 220]) * 17.5
    soma_mea = mea[len(mea) // 2] + [3.0, -2.0]
    p2p = 100 * np.exp(-np.sum((mea - soma_mea - [20, -35]) ** 2, axis=1) / (2 * 60 ** 2)) + rng.random(len(mea))
    template = np.zeros((len(ids), 20), np.float32)
    template[:, 5], template[:, 8] = -p2p, 0
    t = dandi.PatchTemplate(template, template * 0, 100, ids, 20000.0, 5, np.zeros(0))
    # cell frame: MEA rotated by 30 deg and mirrored, soma at the origin
    th = np.deg2rad(30)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([-1.0, 1.0])
    strong = p2p > 30
    cell = (mea[strong] - soma_mea) @ R.T
    json.dump({"pos": np.column_stack([cell, np.zeros(len(cell))]).tolist()}, open(tmp_path / "probe.json", "w"))
    np.save(tmp_path / "template.npy", np.column_stack([np.zeros(strong.sum()), -p2p[strong]]))
    (tmp_path / "cell.swc").write_text("1 1 0 0 0 5 -1\n2 4 0 -20 0 0.5 1\n3 2 0 -60 0 0.3 2\n4 3 30 0 0 1 1\n")
    reg = dandi.register_morphology(t, tmp_path / "probe.json", tmp_path / "template.npy", tmp_path / "cell.swc")
    assert reg.mirrored and reg.amplitude_correlation > 0.99 and reg.matched_fraction == 1.0
    soma = reg.morphology.mid[reg.morphology.kind == "soma", :2].mean(axis=0)
    np.testing.assert_allclose(soma, soma_mea, atol=1.0)
    axon_end = reg.morphology.end[reg.morphology.kind == "axon"][-1, :2]
    np.testing.assert_allclose(axon_end, soma_mea + np.linalg.inv(R) @ [0, -60], atol=1.0)
