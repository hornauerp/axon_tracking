# axon_tracking
Python package to reconstruct the functional/electrophysiological axonal morphology from high-density microelectrode array (HD-MEA) recordings.

Gain intuitive insights into the morphology of your neurons:

<img src="https://github.com/hornauerp/axon_tracking/assets/57408359/317f2577-1e07-4e76-ad48-f42855d4ad6a" width=50% height=50%>

Evaluate the action potential propagation:

<img src="https://github.com/hornauerp/axon_tracking/assets/57408359/1e99a55a-aaf0-49ce-9774-be95fc817af0" width=50% height=50%>

This repo is still a WIP, but will soon be available as a full package to install via pip.

## Installation

Requires Python 3.10 or newer; tested with Python 3.12 (on 3.14, installing SpikeInterface 0.105 failed because its
`numcodecs` dependency could not be built). From the repository root:

```bash
pip install -e .
```

Use `pip install -e ".[notebooks]"` for the extra packages used in the notebooks.
The package targets SpikeInterface 0.105 (`spikeinterface>=0.105,<0.106`).

## Workflow

1. `notebooks/part1_spike-sorting.ipynb`: concatenate and spike sort axon scans (`axon_tracking.spike_sorting`).
   `scripts/sort_axon_scan_ks4.py` runs Kilosort4 with settings for the planar 17.5 um grid (no drift correction,
   whitening and detection within electrode blocks, waveforms for 10 kHz, detection thresholds 7/6);
   `axon_tracking.spike_qc` flags spikes that are indistinguishable from noise.
2. `notebooks/part2_template-extraction.ipynb`: QC units and extract templates (`axon_tracking.template_extraction`).
   With `te_params={"deconvolve": True}` the footprints of all sorted units are estimated jointly per configuration
   (least squares on the spike-triggered sums and the cross-correlograms of all units, also those removed by the
   QC) instead of averaged, which removes co-active units (e.g. during network bursts) from the templates without
   discarding any spikes. Output: `sorter_output/templates_deconvolved` (about 10 s and 1-2 GB per configuration
   for ~100 units at 10 kHz).
3. `notebooks/part3_skeletonization.ipynb`: reconstruct the axonal arbor (`axon_tracking.skeletonization`).
   Experimental alternative: `axon_tracking.graph_reconstruction.reconstruct`, a maximum-weight tree over the
   electrodes in latency order (spatially smoothed template, fixed node threshold, paths centred on the footprint).
4. `notebooks/part4_quantification.ipynb`: quantify the skeletons (`axon_tracking.quantification`)

## Simulated ground truth

`axon_tracking.simulation` generates axon scan templates with known morphology (`pip install -e ".[simulation]"`):
random branching neurons or SWC morphologies, membrane currents from a propagating action potential (or NEURON,
following [axon_velocity](https://github.com/alejoe91/axon_velocity)), extracellular footprints on the MaxWell
grid (line-source model above a non-conducting chip, validated against LFPykit), network bursts with cell-type
dependent participation, and spike-triggered averages per recording configuration with limited spikes and noise.

```python
from axon_tracking.simulation import evaluation, scenarios

culture = scenarios.build_culture(n_neurons=80, seed=0, cache_dir="sim_cache")
for result in scenarios.run_scenario("bursting", culture, n_targets=10):
    paths, branch_points = scenarios.reconstruct_part3(result["template"])
    print(evaluation.evaluate(paths, result["ground_truth"], branch_points=branch_points))
```

Reconstructed culture morphologies (e.g. from NeuroMorpho.Org) can be used instead of random neurons
(`build_culture(morphologies={"excitatory": [...swc], "inhibitory": [...]})`). Calibrated against patched neurons
with imaged morphology (below) and the Radivojevic & Punga (2023) templates: asymmetric AP
(`ap_kwargs=dict(width_ms=0.15, decay_width_ms=0.3)`), axon diameter 0.7 um at 0.4 m/s with v ~ sqrt(d), dendritic
back-propagation (`dendrite_velocity_m_s=0.3, dendrite_length_constant_um=50, return_length_constant_um=50`).
`axon_tracking.deconvolution` estimates the footprints of all sorted units jointly (least squares on the
spike-triggered sums and cross-correlograms), which removes the contamination by co-active sorted units.

## Validation with real ground truth

`axon_tracking.validation.dandi_000294` builds templates of two patched rat cortical neurons on a MaxOne array
([DANDI 000294](https://dandiarchive.org/dandiset/000294)), triggered on their intracellular spikes, and registers
their imaged morphologies (NeuroMorpho `Buccino_Hierlemann_Cell1/2`, aligned with the files of
[multimodalfitting](https://github.com/alejoe91/multimodalfitting)) onto the electrodes, so reconstructions can be
scored against the real axon. `axon_tracking.validation.radivojevic` loads the electrical images of
[Radivojevic & Punga (2023)](https://doi.org/10.5061/dryad.gxd2547r1).

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests run on small synthetic MaxWell recordings and do not need any data.
