import os
import warnings

import matplotlib

matplotlib.use("Agg")

import pytest


@pytest.fixture(autouse=True, scope="session")
def no_maxwell_plugin_download(tmp_path_factory):
    """si.read_maxwell downloads MaxWell's HDF5 compression plugin unless it is already present.
    The synthetic test files are uncompressed, so point HDF5_PLUGIN_PATH to placeholder files."""
    plugin_dir = tmp_path_factory.mktemp("hdf5_plugin")
    for name in ("libcompression.so", "libcompression.dylib", "compression.dll"):
        (plugin_dir / name).write_bytes(b"")
    previous = os.environ.get("HDF5_PLUGIN_PATH")
    os.environ["HDF5_PLUGIN_PATH"] = str(plugin_dir)
    yield
    if previous is None:
        os.environ.pop("HDF5_PLUGIN_PATH", None)
    else:
        os.environ["HDF5_PLUGIN_PATH"] = previous


@pytest.fixture(autouse=True)
def quiet_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield
