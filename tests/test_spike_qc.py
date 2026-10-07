import numpy as np

from axon_tracking import spike_qc as qc


def test_false_positive_spikes_are_flagged():
    rng = np.random.default_rng(0)
    n, pre, post = 200000, 10, 20
    template = np.zeros((pre + post, 3))
    template[pre - 2 : pre + 3, 0] = [-10, -40, -60, -40, -10]
    template[pre - 2 : pre + 3, 1] = [-5, -20, -30, -20, -5]
    traces = rng.normal(0, 5.0, (n, 3))
    real = np.sort(rng.choice(np.arange(100, n - 100, 50), 800, replace=False))
    for t in real:
        traces[t - pre : t + post] += template
    false = np.sort(rng.choice(np.setdiff1d(np.arange(100, n - 100), np.r_[real, real + 1, real - 1]), 200,
                               replace=False))
    train = np.sort(np.r_[real, false])
    get = lambda a, b: traces[a:b]
    scores = qc.projection_scores(get, n, [train], template[None], [np.array([0, 1])], pre, post, chunk=30000)[0]
    decoys = qc.decoy_times(train, n, 2000, pre, post, rng)
    null = qc.projection_scores(get, n, [decoys], template[None], [np.array([0, 1])], pre, post)[0]
    flagged = qc.false_positive_mask(scores, null, fpr=0.01)
    is_false = np.isin(train, false)
    assert flagged[is_false].mean() > 0.9  # false detections are recognised (unless next to a real spike)
    assert flagged[~is_false].mean() < 0.02  # real spikes are kept
    assert abs(np.median(scores[~is_false]) - 1) < 0.1


def test_unit_channels_around_the_peak():
    locations = np.array([[0, 0], [17.5, 0], [35, 0], [200, 0]], float)
    templates = np.zeros((1, 5, 4))
    templates[0, 2, 1] = -50
    templates[0, 2, 0] = -10
    assert list(qc.unit_channels(templates, locations)[0]) == [0, 1, 2]
