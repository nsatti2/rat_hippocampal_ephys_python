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
| Spike Raster Plot | <img width="513" height="287" alt="Screenshot 2026-05-16 at 6 11 04 PM" src="https://github.com/user-attachments/assets/dca75069-60fa-49d6-bb26-d29303f89633" />|
| Population Firing Rate | <img width="2404" height="929" alt="image" src="https://github.com/user-attachments/assets/b2b01ab5-0c27-43bd-a0ed-97e77cf4282f" />|
| PCA Variance Explained | <img width="1982" height="1448" alt="image" src="https://github.com/user-attachments/assets/4083a86b-f9ef-4f2e-80a8-5ce97595850d" /> <img width="1588" height="1107" alt="image" src="https://github.com/user-attachments/assets/87abd97b-5257-45ad-b6ec-4fb70a944f09" />|
| Firing Rate per Behavioral State | <img width="1999" height="1325" alt="image" src="https://github.com/user-attachments/assets/c1347c77-a232-485a-877a-2b777f52b7c6" />|
| Functional Connectivity | <img width="1584" height="1514" alt="image" src="https://github.com/user-attachments/assets/4d5c9446-2a59-4587-baad-417a7172ca38" />|

## Tools
Python · NumPy · SciPy · scikit-learn · matplotlib

## Background
Pipeline mirrors analysis performed on real Kilosort/Neuropixels data in a memory lab setting, including spike sorting output structure, behavioral state alignment, and dimensionality reduction approaches used in systems neuroscience.
