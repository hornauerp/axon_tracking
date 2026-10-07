"""Electrical images of cortical (and motor) neurons on MaxOne (Radivojevic & Punga 2023, eLife 86512; Dryad
doi:10.5061/dryad.gxd2547r1 or Zenodo 8251984, CC0). Templates of 26400 electrodes at 20 kHz (the README's
200 kHz refers to their upsampled analysis), no morphology."""

import io
import zipfile

import numpy as np


def neuron_names(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n for n in z.namelist() if n.endswith(".mat") and not n.startswith("__MACOSX"))


def load_template(zip_path, name, pre=30, post=120):
    """Template (220, 120, pre + post) around the somatic trough (largest negative sample)."""
    import scipy.io as sio

    with zipfile.ZipFile(zip_path) as z:
        d = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True)["myNeuron"]
    temp = np.asarray(d["temp"].item(), float)
    elcs = np.asarray(d["elcs"].item(), int)
    _, t0 = np.unravel_index(np.argmin(temp), temp.shape)
    crop = np.pad(temp, ((0, 0), (pre, post)))[:, t0 : t0 + pre + post]
    grid = np.zeros((220, 120, pre + post))
    grid[elcs % 220, elcs // 220] = crop
    return grid
