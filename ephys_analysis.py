import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

print("Libraries loaded. Starting analysis...")

# ─────────────────────────────────────────
# 2. SIMULATE SPIKE TRAIN DATA
# (mirrors your MATLAB pipeline structure)
# ─────────────────────────────────────────

np.random.seed(42)

# Simulate 60 neurons, 300 seconds recording
n_neurons = 60
duration = 300  # seconds
fs = 30000      # sampling rate (matches your Kilosort data)

print(f"\nSimulating {n_neurons} neurons over {duration}s at {fs}Hz...")

# Each neuron fires at a different base rate (2-30 Hz)
base_rates = np.random.uniform(2, 30, n_neurons)

# Generate spike times for each neuron
spike_times_list = []
spike_clusters_list = []

for i, rate in enumerate(base_rates):
    # Poisson spike train
    n_spikes = np.random.poisson(rate * duration)
    times = np.sort(np.random.uniform(0, duration, n_spikes))
    spike_times_list.append(times)
    spike_clusters_list.append(np.full(n_spikes, i))

spike_times_sec = np.concatenate(spike_times_list)
spike_clusters  = np.concatenate(spike_clusters_list)

# Sort by time
sort_idx       = np.argsort(spike_times_sec)
spike_times_sec = spike_times_sec[sort_idx]
spike_clusters  = spike_clusters[sort_idx]

print(f"Total spikes generated: {len(spike_times_sec)}")

# ─────────────────────────────────────────
# 3. RASTER PLOT
# ─────────────────────────────────────────

print("\nPlotting raster...")
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(n_neurons):
    idx = spike_clusters == i
    t   = spike_times_sec[idx]
    ax.plot(t, np.ones_like(t) * i, '.k', markersize=0.5)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Neuron #')
ax.set_title('Spike Raster Plot — 60 Simulated Neurons')
ax.set_xlim(0, duration)
plt.tight_layout()
plt.savefig('figures/raster_plot.png', dpi=150)
plt.show()
print("Saved: figures/raster_plot.png")

# ─────────────────────────────────────────
# 4. POPULATION FIRING RATE HISTOGRAM
# ─────────────────────────────────────────

bin_size = 0.130  # 130ms bins (matches your MATLAB)
edges    = np.arange(0, duration + bin_size, bin_size)

counts, _ = np.histogram(spike_times_sec, bins=edges)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(edges[:-1], counts, width=bin_size, color='steelblue', edgecolor='none')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spike Count')
ax.set_title('Population Firing Rate Histogram')
plt.tight_layout()
plt.savefig('figures/firing_rate_histogram.png', dpi=150)
plt.show()
print("Saved: figures/firing_rate_histogram.png")

# ─────────────────────────────────────────
# 5. RASTER MATRIX + PCA
# ─────────────────────────────────────────

print("\nBuilding raster matrix and running PCA...")

n_bins = len(edges) - 1
raster_matrix = np.zeros((n_neurons, n_bins))

for i in range(n_neurons):
    idx = spike_clusters == i
    raster_matrix[i], _ = np.histogram(spike_times_sec[idx], bins=edges)

# PCA on population activity (bins x neurons)
pca = PCA()
score = pca.fit_transform(raster_matrix.T)  # shape: n_bins x n_components
explained = pca.explained_variance_ratio_ * 100

n_for_80 = np.argmax(np.cumsum(explained) >= 80) + 1
n_for_90 = np.argmax(np.cumsum(explained) >= 90) + 1
print(f"{n_for_80} PCs explain 80% of variance")
print(f"{n_for_90} PCs explain 90% of variance")
print(f"{n_for_80/n_neurons*100:.1f}% of neurons needed for 80% variance")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.cumsum(explained), 'o-', linewidth=2, markersize=3)
ax.axhline(80, color='r', linestyle='--', label='80%')
ax.axhline(90, color='g', linestyle='--', label='90%')
ax.set_xlabel('Number of PCs')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_title('PCA — How many dimensions capture population activity?')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('figures/pca_variance.png', dpi=150)
plt.show()
print("Saved: figures/pca_variance.png")

# ─────────────────────────────────────────
# 6. SIMULATED BEHAVIORAL STATES + NEURAL MODULATION
# ─────────────────────────────────────────

print("\nMapping behavioral states to neural activity...")

behavior_names = ['Rearing-wall', 'Rearing-unsupported', 'Turnaround',
                  'Head scan', 'Locomotion', 'Rest']

# Assign behavioral states to time bins
bin_centers    = edges[:-1] + bin_size / 2
behavior_label = np.zeros(n_bins, dtype=int)

# Simulate behavioral epochs across the session
np.random.seed(7)
t = 0
while t < duration:
    state    = np.random.randint(1, 7)
    length   = np.random.uniform(2, 15)
    idx      = (bin_centers >= t) & (bin_centers < t + length)
    behavior_label[idx] = state
    t += length

# Firing rate per neuron per behavioral state
firing_rates = np.zeros((n_neurons, 6))
for b in range(1, 7):
    bins_in_state = behavior_label == b
    if bins_in_state.sum() > 0:
        firing_rates[:, b-1] = raster_matrix[:, bins_in_state].mean(axis=1) / bin_size

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(firing_rates, aspect='auto', cmap='hot')
plt.colorbar(im, ax=ax, label='Firing Rate (Hz)')
ax.set_xlabel('Behavioral State')
ax.set_ylabel('Neuron #')
ax.set_xticks(range(6))
ax.set_xticklabels(behavior_names, rotation=45, ha='right')
ax.set_title('Firing Rate per Neuron per Behavioral State')
plt.tight_layout()
plt.savefig('figures/firing_rate_heatmap.png', dpi=150)
plt.show()
print("Saved: figures/firing_rate_heatmap.png")

# ─────────────────────────────────────────
# 7. 3D POPULATION STATE BY BEHAVIOR
# ─────────────────────────────────────────

fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
colors = ['blue', 'green', 'red', 'magenta', 'black', 'cyan']

for b in range(1, 7):
    bidx = behavior_label == b
    ax.scatter(score[bidx, 0], score[bidx, 1], score[bidx, 2],
               s=1, c=colors[b-1], label=behavior_names[b-1], alpha=0.5)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Population State by Behavioral State')
ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig('figures/population_state_3d.png', dpi=150)
plt.show()
print("Saved: figures/population_state_3d.png")

# ─────────────────────────────────────────
# 8. FUNCTIONAL CONNECTIVITY
# ─────────────────────────────────────────

print("\nComputing functional connectivity...")

# Smooth raster slightly before correlating
from scipy.ndimage import gaussian_filter1d
raster_smooth = gaussian_filter1d(raster_matrix.astype(float), sigma=2, axis=1)

corr_matrix = np.corrcoef(raster_smooth)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_matrix, cmap='jet', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xlabel('Neuron #')
ax.set_ylabel('Neuron #')
ax.set_title('Functional Connectivity — Spike Count Correlation')
plt.tight_layout()
plt.savefig('figures/functional_connectivity.png', dpi=150)
plt.show()
print("Saved: figures/functional_connectivity.png")

# Top correlated pairs
np.fill_diagonal(corr_matrix, np.nan)
flat       = corr_matrix.flatten()
sorted_idx = np.argsort(flat)[::-1]

print("\nTop 5 most correlated neuron pairs:")
seen = set()
count = 0
for idx in sorted_idx:
    r, c = divmod(idx, n_neurons)
    if r != c and (c, r) not in seen:
        print(f"  Neuron {r} & Neuron {c}: r = {flat[idx]:.4f}")
        seen.add((r, c))
        count += 1
        if count == 5:
            break

print("\nDone! All figures saved to figures/")