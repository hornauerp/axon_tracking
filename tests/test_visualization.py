import matplotlib.pyplot as plt
import numpy as np
import osteoid

from axon_tracking import visualization as vis


def test_plot_skeleton_applies_limits():
    vertices = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 4.0], [20.0, 12.0, 8.0]])
    skeleton = osteoid.Skeleton(vertices, np.array([[0, 1], [1, 2]], np.uint32))
    vis.plot_skeleton(skeleton, x_lim=[-5, 30], y_lim=[-2, 20])
    ax = plt.gca()
    assert ax.get_xlim() == (-5, 30)
    assert ax.get_ylim() == (-2, 20)
    plt.close("all")
