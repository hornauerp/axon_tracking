import numpy as np

from axon_tracking import quantification as quant


def test_template_size_counts_unique_positions():
    b1 = np.array([[100.0, 200.0, 0], [117.5, 200.0, 1], [135.0, 200.0, 2]])
    b2 = np.array([[135.0, 200.0, 2], [135.0, 217.5, 3]])
    assert quant.get_simple_template_size([b1, b2]) == 4


def test_template_size_on_electrode_grid():
    smoothed = np.array([[100.0, 200.0, 0], [101.0, 201.0, 1], [117.0, 200.0, 2]])
    assert quant.get_simple_template_size([smoothed]) == 3
    assert quant.get_simple_template_size([smoothed], el_spacing=17.5) == 2


def test_sliding_window_velocity():
    params = {"sampling_rate": 20000}
    i = np.arange(30)
    path = np.column_stack([i * 35.0, np.zeros(30), i])  # 35 um per sample at 20 kHz = 0.7 m/s
    mean_vel, std_vel = quant.get_sliding_window_velocity([path], params, window_size=5)
    assert np.isclose(mean_vel, 0.7)
    assert np.isclose(std_vel, 0)


def test_sliding_window_velocity_without_valid_paths():
    params = {"sampling_rate": 20000}
    assert np.all(np.isnan(quant.get_sliding_window_velocity([np.zeros((2, 3))], params)))
