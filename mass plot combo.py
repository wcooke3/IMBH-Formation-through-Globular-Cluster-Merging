import numpy as np
import matplotlib.pyplot as plt
import csv

# Set global font size for all text elements
plt.rcParams.update({'font.size': 12})  # Increase font size globally

AU = 1.496e11              # astronomical unit (m)
kpc = 2.063e8 * AU         # kpc (m)
pc = kpc / 1000            # pc (m)
M_sun = 1.988475e30        # Mass of Sun (kg)

def sample_imf(n_stars, m_min, m_max, alpha=-2.35):
    # Sample masses from a power-law IMF
    u = np.random.uniform(0, 1, n_stars)
    m_stars = ((m_max**(alpha + 1) - m_min**(alpha + 1)) * u + m_min**(alpha + 1))**(1 / (alpha + 1))
    return m_stars

def sample_radii(n_stars, r_core, r_tidal):
    # Sample radii from a King-like profile
    u = np.random.uniform(0, 1, n_stars)
    r_stars = r_core * np.sqrt((1 + (r_tidal / r_core)**2)**u - 1)
    return r_stars

def assign_masses(n_stars, m_min, m_max, r_core, r_tidal, alpha=-2.35):
    # Sample masses from IMF
    m_stars = sample_imf(n_stars, m_min, m_max, alpha)
    # Sample radii from King-like profile
    r_stars = sample_radii(n_stars, r_core, r_tidal)
    # Sort masses in descending order (most massive first)
    sorted_masses = np.sort(m_stars)[::-1]
    # Sort radii in ascending order (smallest radii first)
    sorted_radii = np.sort(r_stars)
    # Assign the most massive stars to the smallest radii
    return sorted_masses, sorted_radii

# Parameters
n_stars = int(1e3)                          # Number of stars
m_min, m_max = 0.5, 2.5                     # Minimum and maximum stellar masses (solar masses)
r_core, r_tidal = 1.0, 10.0                 # Core and tidal radii (parsecs)

# Generate masses and radii
masses, radii = assign_masses(n_stars, m_min, m_max, r_core, r_tidal)

# Function to load data from a CSV file
def load_final_state(filename):
    x, y, z, m = [], [], [], []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
            m.append(float(row[6]))
    return np.array(x), np.array(y), np.array(z), np.array(m)

# List of merger files
merger_files = ["merger 1.csv", "merger 2.csv", "merger 3.csv", "merger 4.csv", "merger 5.csv"]
colors = ['b', 'g', 'r', 'c', 'm']  # Colors for different mergers
labels = [f"Merger {i+1}" for i in range(len(merger_files))]

# Store data for plotting
r_all = []
m_all = []
fin = []

# Load all mergers
for file in merger_files:
    x, y, z, m = load_final_state(file)
    r = np.sqrt(x**2 + y**2 + z**2)  # Compute radius
    sorted_indices = np.argsort(r)   # Sort by radius
    r_sorted = r[sorted_indices]
    m_sorted = m[sorted_indices]
    r_all.append(r_sorted)
    m_all.append(m_sorted)
    fin.append(r_sorted[-1])


# Create a figure with two subplots sharing the same x-axis
# Adjust the height ratio to make the histogram smaller
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})

# ---------------------------
# 1. Cumulative Mass Distribution (CDF)
# ---------------------------
for i in range(len(merger_files)):
    cumulative_mass = np.cumsum(m_all[i]) / np.sum(m_all[i])  # Normalize to total mass
    ax1.plot(r_all[i] / pc, cumulative_mass, color=colors[i], label=labels[i])
ax1.set_ylabel("Cumulative Mass Fraction", fontsize=16)  # Increase font size for y-label
ax1.set_title("Cluster Mass Distribution over Subsequent Mergers", fontsize=16)  # Increase font size for title
for i in range(len(fin)):
    ax1.axvline(x=fin[i]/pc, color=colors[i], linestyle='--', linewidth=1.5) # vertical line

# Overlay mass-radius relationship on the CDF plot
#ax1_twin = ax1.twinx()  # Create a twin axis for the initial mass distribution

# Calculate cumulative mass for the initial mass distribution
cumulative_masses = np.cumsum(masses) / np.sum(masses)  # Normalize to total mass
ax1.plot(radii, cumulative_masses, 'k-', label='Initial Distribution')
ax1.legend(loc='lower right', fontsize=13)  # Increase font size for twin legend
ax1.axvline(x=radii[-1], color='k', linestyle='--', linewidth=1.5) # Draw a vertical line
ax1.legend(loc='right', fontsize=13)  # Increase font size for legend

# ---------------------------
# 2. Mass Histogram (Density Distribution)
# ---------------------------
bins = np.linspace(0, np.max(r_all[-1]) / pc, 30)  # Create radial bins
for i in range(len(merger_files)):
    ax2.hist(r_all[i] / pc, bins=bins, weights=m_all[i] / M_sun, alpha=0.5, color=colors[i], label=labels[i], histtype='stepfilled')
ax2.set_xlabel("Distance from cluster center [pc]", fontsize=16)
ax2.set_ylabel("Total Mass per bin [M☉]", fontsize=16)
ax2.legend(loc='right', fontsize=13)
for i in range(len(fin)):
    ax2.axvline(x=fin[i]/pc, color=colors[i], linestyle='--', linewidth=1.5)
ax2.axvline(x=radii[-1], color='k', linestyle='--', linewidth=1.5)

# Remove the gap between the two subplots
plt.subplots_adjust(hspace=0)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()