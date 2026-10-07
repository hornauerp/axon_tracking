"""Simulated ground truth for axon tracking on MaxWell HD-MEAs.

Layers:
    morphology  -- neuron morphologies (SWC files or random branching axons) as line segments
    currents    -- transmembrane currents: NEURON simulation or a prescribed propagation model
    forward     -- extracellular footprints on the MaxWell electrode grid (LFPykit, non-conducting chip)
    activity    -- spike trains with tonic firing and network bursts
    recording   -- axon scan configurations and spike-triggered average templates
    evaluation  -- comparison of reconstructions with the ground-truth axon
    scenarios   -- predefined culture/recording scenarios

Optional dependencies: LFPykit (forward model) and NEURON (biophysical currents).
"""
