# Hippocampal Electrophysiology Analysis

Python pipeline for analyzing neural population dynamics during a spatial memory task, modeled after in-vivo rat hippocampal recordings (CircleTrack paradigm).

## What this project does
- Generates and processes spike train data from 60 simulated neurons (Poisson, 2–30 Hz)
- Builds population raster plots and firing rate histograms
- Performs PCA on neural population activity to identify low-dimensional state structure
- Maps population dynamics to behavioral states (rearing, locomotion, head scanning, turnaround, rest)
- Computes pairwise functional connectivity via spike-count correlation matrices

## Figures
| Analysis | Figure |
|---|---|
| Spike Raster Plot | figures/raster_plot.png |
| Population Firing Rate | figures/firing_rate_histogram.png |
| PCA Variance Explained | figures/pca_variance.png |
| Firing Rate per Behavioral State | figures/firing_rate_heatmap.png |
| Functional Connectivity | figures/functiol_connectivity.png |

## Tools
Python · NumPy · SciPy · scikit-learn · matplotlib

## Background
Pipeline mirrors analysis performed on real Kilosort/Neuropixels data in a memory lab setting, including spike sorting output structure, behavioral state alignment, and dimensionality reduction approaches used in systems neuroscience.
