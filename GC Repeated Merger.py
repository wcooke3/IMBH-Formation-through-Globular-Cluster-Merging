import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from numba import njit, prange
import csv

# Constants
G = 6.67430e-11            # gravitational constant (N m^2 kg^-2)
AU = 1.496e11              # astronomical unit (m)
kpc = 2.063e8 * AU         # kpc (m)
pc = kpc / 1000            # pc (m)
M_sun = 1.988475e30        # Mass of Sun (kg)

day = 24 * 60 * 60         # day in seconds (s)
year = 365 * day           # one year (s)
total_time = 5e6 * year    # total simulation time (s) [5 milliion years]
dt = 1e3 * year            # smaller timestep (s)
n = int(total_time / dt)   # number of timesteps
print(n, "time steps")

central_bh = 20 * M_sun  
bh_marker_size = 10 # adjust size of black hole in plot

R = 10 * pc                # radius of the sphere particle positions are generated in
schw = 2 * G * central_bh / (3e8)**2

stick_distance = 100 * AU   # !!!!! CHANGE FOR TESTING !!!!!------------------------------------
epsilon = 0.001 * R        # reduced softening parameter

# Parameters for the globular clusters
M_cluster = 1e4 * M_sun    # Total mass of each globular cluster in M_sun 
N_stars = int(1e3)         # Number of star particles per cluster
r_c = 1.0 * pc             # Core radius of the King model (m)
r_t = 10.0 * pc            # Tidal radius of the King model (m)
rho_0 = 1.0                # Central density (arbitrary units, will normalize)

# Calculate the required velocity dispersion for virial equilibrium (excluding BH mass)
sigma = np.sqrt(G * (M_cluster - central_bh) / (5*R)) # A more accurate formula for the velocity dispersion that accounts for the King model as mass not distributed in uniform sphere

# King model density profile
def king_model(r, r_c, r_t, rho_0):
    term1 = 1 / np.sqrt(1 + (r / r_c)**2)
    term2 = 1 / np.sqrt(1 + (r_t / r_c)**2)
    return rho_0 * (term1 - term2)**2

# Generate radial distances using the King model
def sample_king_model(N, r_c, r_t, rho_0):
    r = np.linspace(0, r_t, 1000)
    density = king_model(r, r_c, r_t, rho_0)
    cdf = cumulative_trapezoid(density, r, initial=0)
    cdf /= cdf[-1]  # Normalize CDF
    inv_cdf = interp1d(cdf, r, bounds_error=False, fill_value='extrapolate')
    u = np.random.rand(N)
    return inv_cdf(u)

# Distribute particles in 3D space
def distribute_particles(r_values, N, R):
    x, y, z = [], [], []
    for i in range(N):
        while True:
            r = r_values[i]
            if r > R:
                r = np.random.uniform(0, R)
           
            # Sample spherical coordinates
            theta = np.arccos(2 * np.random.uniform(0, 1) - 1)
            phi = np.random.uniform(0, 2 * np.pi)

            # Convert to Cartesian coordinates
            x_new = r * np.sin(theta) * np.cos(phi)
            y_new = r * np.sin(theta) * np.sin(phi)
            z_new = r * np.cos(theta)

            if np.sqrt(x_new**2 + y_new**2 + z_new**2) <= R:
                x.append(x_new)
                y.append(y_new)
                z.append(z_new)
                break
    return np.array(x), np.array(y), np.array(z)

# Define a mass range
m_min = 0.5 * M_sun  # Minimum mass
m_max = 2.25 * M_sun   # Maximum mass
bh_threshold = 3 * M_sun  # Black holes are particles with mass > 3 solar masses
bh_masses = []

# Assign particle masses based on radial distance [distributed linearly]
def assign_masses(r_values, m_min, m_max):
    # Normalize radial distances to [0, 1]
    r_norm = r_values / np.max(r_values)
    # Assign masses such that larger masses are closer to the center
    m_stars = m_max - (m_max - m_min) * r_norm
    return m_stars

# Import cluster state from previous merger
def load_final_state(filename):
    x, y, z, vx, vy, vz, m = [], [], [], [], [], [], []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))
            vx.append(float(row[3]))
            vy.append(float(row[4]))
            vz.append(float(row[5]))
            m.append(float(row[6]))
    return np.array(x), np.array(y), np.array(z), np.array(vx), np.array(vy), np.array(vz), np.array(m)

# Load saved state from CSV file
x1_stars, y1_stars, z1_stars, vx1_stars, vy1_stars, vz1_stars, m1_stars = load_final_state('merger 4.csv') # ---------------------change file name-------------------------

# Generate second cluster as usual
r_stars = sample_king_model(N_stars, r_c, r_t, rho_0) # generates distribution layout which is applied to both GCs
m_stars = assign_masses(r_stars, m_min, m_max)
m_stars = m_stars[:-1]

x2_stars, y2_stars, z2_stars = distribute_particles(r_stars, N_stars, R)
offset = 5 * pc
x2_stars += offset  # Offset cluster 2's positions

# Combine star particles for both clusters (include BHs)
x0 = np.concatenate([x1_stars, x2_stars])
y0 = np.concatenate([y1_stars, y2_stars])
z0 = np.concatenate([z1_stars, z2_stars])
m = np.concatenate([m1_stars, m_stars, [central_bh]])  # Combine masses
print(len(m), "particles")
print(len(m1_stars), (len(m_stars)+1))
breakpoint

# Center of Mass Calculation (re-center including BHs)
Rcom_x = np.sum(m * x0) / np.sum(m)
Rcom_y = np.sum(m * y0) / np.sum(m)
Rcom_z = np.sum(m * z0) / np.sum(m)
x0 -= Rcom_x
y0 -= Rcom_y
z0 -= Rcom_z

# Assign velocities with BH sphere of influence adjustment
m_star = np.mean(m_stars) # I dont want a mean because I want a mass range
r_influence = G * central_bh / sigma**2  # Sphere of influence
vx0, vy0, vz0 = [], [], []

# Add bound orbital velocity for the second cluster
M_total = 2 * M_cluster  # Total mass of both clusters
d_initial = offset       # Initial separation between clusters
v_circular = np.sqrt(G * M_total / d_initial)  # ~1.3e3 m/s for your parameters

# GCs have same no. of stars

for i in range(len(m)):
    r = np.sqrt(x0[i]**2 + y0[i]**2 + z0[i]**2)
    if r < r_influence and m[i] < 3 * m_star:  # Stars within BH influence
        v_mag = np.sqrt(G * (central_bh + m[i]) / r)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        vx0.append(v_mag * direction[0])
        vy0.append(v_mag * direction[1])
        vz0.append(v_mag * direction[2])
    else:
        # Assign velocities for the second cluster with bound orbital velocity
        if i >= N_stars:
            vx0.append(-v_circular)
        else:
            vx0.append(np.random.normal(0, sigma))
        vy0.append(np.random.normal(0, sigma))
        vz0.append(np.random.normal(0, sigma))

vx0 = np.array(vx0)
vy0 = np.array(vy0)
vz0 = np.array(vz0)

# Subtract center of mass velocity
Vcom_x = np.sum(m * vx0) / np.sum(m)
Vcom_y = np.sum(m * vy0) / np.sum(m)
Vcom_z = np.sum(m * vz0) / np.sum(m)
vx0 -= Vcom_x
vy0 -= Vcom_y
vz0 -= Vcom_z

# Arrays to save states
x = [x0]
y = [y0]
z = [z0]
ux = [vx0]
uy = [vy0]
uz = [vz0]
t = [0]

# JIT-compiled function for acceleration calculation
@njit(parallel=True)
def compute_accelerations(x, y, z, m, G, epsilon):
    n = len(m)
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.zeros(n)
    for i in prange(n):
        for j in prange(n):
            if i != j:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]
                dist2 = dx**2 + dy**2 + dz**2
                dist = np.sqrt(dist2 + epsilon**2)
                dist3 = dist**3
                ax[i] += G * m[j] * dx / dist3
                ay[i] += G * m[j] * dy / dist3
                az[i] += G * m[j] * dz / dist3
    return ax, ay, az

# MAIN SIMULATION LOOP (LEAPFROG INTEGRATOR)
measurement_points = 400
measurement_interval = int(n / measurement_points)
stored_states = []
bh_count = []

# Initialize half-step velocities
ax, ay, az = compute_accelerations(x0, y0, z0, m, G, epsilon)
ux_half = vx0 + ax * (dt / 2)
uy_half = vy0 + ay * (dt / 2)
uz_half = vz0 + az * (dt / 2)

#-------------------------- LEAPFROG ---------------------------
for i in range(n):
    # Leapfrog integrator steps
    x_new = x[-1] + ux_half * dt
    y_new = y[-1] + uy_half * dt
    z_new = z[-1] + uz_half * dt

    ax_new, ay_new, az_new = compute_accelerations(x_new, y_new, z_new, m, G, epsilon)
    
    ux_half += ax_new * dt
    uy_half += ay_new * dt
    uz_half += az_new * dt

    ux_new = ux_half - ax_new * (dt / 2)
    uy_new = uy_half - ay_new * (dt / 2)
    uz_new = uz_half - az_new * (dt / 2)

    # Collision detection and merger handling
    tree = cKDTree(np.column_stack((x_new, y_new, z_new)))
    pairs = tree.query_pairs(stick_distance)
    to_remove = set()

    for j, l in pairs:
        if j not in to_remove and l not in to_remove:
            is_bh_j = (m[j] > bh_threshold)     
            is_bh_l = (m[l] > bh_threshold)
            
            if is_bh_j and is_bh_l:
                # Merge two black holes
                bh_idx, other_bh_idx = j, l
                ux_new[bh_idx] = (m[bh_idx] * ux_new[bh_idx] + m[other_bh_idx] * ux_new[other_bh_idx]) / (m[bh_idx] + m[other_bh_idx])
                uy_new[bh_idx] = (m[bh_idx] * uy_new[bh_idx] + m[other_bh_idx] * uy_new[other_bh_idx]) / (m[bh_idx] + m[other_bh_idx])
                uz_new[bh_idx] = (m[bh_idx] * uz_new[bh_idx] + m[other_bh_idx] * uz_new[other_bh_idx]) / (m[bh_idx] + m[other_bh_idx])
                m[bh_idx] += m[other_bh_idx]

                to_remove.add(other_bh_idx)
            elif is_bh_j or is_bh_l:
                # Merge star into BH
                if is_bh_j:
                    bh_idx, star_idx = j, l
                else:
                    bh_idx, star_idx = l, j
                
                # Update BH mass and velocity (momentum conservation)
                ux_new[bh_idx] = (m[bh_idx] * ux_new[bh_idx] + m[star_idx] * ux_new[star_idx]) / m[bh_idx]
                uy_new[bh_idx] = (m[bh_idx] * uy_new[bh_idx] + m[star_idx] * uy_new[star_idx]) / m[bh_idx]
                uz_new[bh_idx] = (m[bh_idx] * uz_new[bh_idx] + m[star_idx] * uz_new[star_idx]) / m[bh_idx]
                m[bh_idx] += m[star_idx]
                to_remove.add(star_idx)

    # Remove merged particles
    to_remove = sorted(to_remove, reverse=True)
    for idx in to_remove:
        m = np.delete(m, idx)
        x_new = np.delete(x_new, idx)
        y_new = np.delete(y_new, idx)
        z_new = np.delete(z_new, idx)
        ux_new = np.delete(ux_new, idx)
        uy_new = np.delete(uy_new, idx)
        uz_new = np.delete(uz_new, idx)
        ux_half = np.delete(ux_half, idx)
        uy_half = np.delete(uy_half, idx)
        uz_half = np.delete(uz_half, idx)
    

    # Remove unbound particles
    if len(m) > 0:  # Only remove particles if there are particles left
        v_mag = np.sqrt(ux_new**2 + uy_new**2 + uz_new**2)
        r = np.sqrt(x_new**2 + y_new**2 + z_new**2)
        escape_velocity = np.sqrt(2 * G * np.sum(m) / r)
        safety_margin = 1.5  # Adjust as needed
        unbound = (v_mag > safety_margin * escape_velocity) & (m < bh_threshold) # only removes stars, not BHs
        m = np.delete(m, unbound)
        x_new = np.delete(x_new, unbound)
        y_new = np.delete(y_new, unbound)
        z_new = np.delete(z_new, unbound)
        ux_new = np.delete(ux_new, unbound)
        uy_new = np.delete(uy_new, unbound)
        uz_new = np.delete(uz_new, unbound)
        ux_half = np.delete(ux_half, unbound)
        uy_half = np.delete(uy_half, unbound)
        uz_half = np.delete(uz_half, unbound)

    if len(m) == 0:
        print("All particles have been removed. Ending simulation.")
        break

    # Track BH masses
    bh_indices = np.where(m > bh_threshold)[0]
    bh_masses_current = [m[idx] for idx in bh_indices] # masses of all BHs
    
    # Append BH masses to the tracking list
    bh_masses.append(bh_masses_current)
    number_bh = len(bh_indices)
    bh_count.append(number_bh)

    # Update state lists
    x.append(x_new)
    y.append(y_new)
    z.append(z_new)
    ux.append(ux_new)
    uy.append(uy_new)
    uz.append(uz_new)

    if i % measurement_interval == 0:
        stored_states.append((x_new, y_new, z_new, ux_new, uy_new, uz_new, m.copy()))

    print(i)

# Save the final state to a CSV file
def save_final_state(filename, x, y, z, vx, vy, vz, m):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['x (m)', 'y (m)', 'z (m)', 'vx (m/s)', 'vy (m/s)', 'vz (m/s)', 'mass (kg)'])
        # Write the data
        for i in range(len(m)):
            writer.writerow([x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i]])

# Call the function to save the final state
final_state_filename = 'merger 5.csv' #----------------------------------------------------------------------------change file name-------------------------------
save_final_state(final_state_filename, x[-1], y[-1], z[-1], ux[-1], uy[-1], uz[-1], m)

#--------------------------------------------------------------------------
# POST-PROCESSING: Calculate momentum and energies for stored states
total_momentum = []
total_kinetic = []
total_gravitational = []

# Function to calculate gravitational potential energy (Numba-optimized)
@njit(parallel=True)
def compute_gravitational_potential_energy(m, positions, G, epsilon):
    n = len(m)
    system_gravitational = 0.0
    for j in prange(n):
        for l in prange(j + 1, n):
            dx = positions[j, 0] - positions[l, 0]
            dy = positions[j, 1] - positions[l, 1]
            dz = positions[j, 2] - positions[l, 2]
            dist = np.sqrt(dx**2 + dy**2 + dz**2 + epsilon**2)  # Include softening
            system_gravitational += -G * m[j] * m[l] / dist
    return system_gravitational

# Main loop for processing stored states
for state in stored_states:
    x_new, y_new, z_new, ux_new, uy_new, uz_new, m = state
    velocities = np.array([ux_new, uy_new, uz_new]).T  # Shape: (k, 3)
    positions = np.array([x_new, y_new, z_new]).T      # Shape: (k, 3)

    # Vectorized momentum and kinetic energy calculations
    system_momentum = np.sum(m[:, np.newaxis] * velocities, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    system_kinetic = np.sum(0.5 * m * speeds**2)

    # Optimized gravitational potential energy calculation
    if len(m) >= 2:
        system_gravitational = compute_gravitational_potential_energy(m, positions, G, epsilon)
    else:
        system_gravitational = 0.0  # No gravitational potential if fewer than 2 particles

    # Append results to lists
    total_momentum.append(system_momentum)
    total_kinetic.append(system_kinetic)
    total_gravitational.append(system_gravitational)

# Calculate virial ratio
virial_ratio = []
virial_theorem = []
total_energy = []
for i in range(len(total_kinetic)):
    if abs(total_gravitational[i]) > 0:  # Avoid division by zero
        virial_ratio.append((2 * total_kinetic[i]) / abs(total_gravitational[i]))
    else:
        virial_ratio.append(0.0)
    total_energy.append(total_kinetic[i] + total_gravitational[i])
    virial_theorem.append(2*total_kinetic[i] + total_gravitational[i])

print(np.mean(virial_ratio))

# Convert bh_masses to a NumPy array for easier analysis
bh_masses = np.array(bh_masses)

#--------------------------------------------------------------------------
# GRAPH PLOTTING
# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colours for each cluster and merged particles
colors = ['green', 'blue']
merged_color = 'red'
bh_color = 'black'

# Create lines with different colours
lines = [ax.plot([], [], [], '.', color=colors[i // N_stars])[0] for i in range(2 * N_stars)]

# Set dynamic limits for the axes
def set_dynamic_limits(ax, x_data, y_data, z_data):
    x_flat = np.concatenate(x_data) / pc
    y_flat = np.concatenate(y_data) / pc
    z_flat = np.concatenate(z_data) / pc
    ax.set_xlim(np.min(x_flat), np.max(x_flat))
    ax.set_ylim(np.min(y_flat), np.max(y_flat))
    ax.set_zlim(np.min(z_flat), np.max(z_flat))

set_dynamic_limits(ax, x, y, z)
ax.set_xlabel('X (pc)')
ax.set_ylabel('Y (pc)')
ax.set_zlabel('Z (pc)')

# Initialization function
def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

# Update function over time
def update(frame, m):
    # Get the current state
    x_frame = x[frame] / pc
    y_frame = y[frame] / pc
    z_frame = z[frame] / pc
    m_frame = m[:len(x_frame)]  # Ensure m_frame matches the current number of particles

    # Update the number of lines to match the current number of particles
    while len(lines) < len(x_frame):
        # Add new lines for new particles
        new_line = ax.plot([], [], [], '.', color='blue')[0]  # Default color for stars
        lines.append(new_line)
    while len(lines) > len(x_frame):
        # Remove lines for deleted particles
        lines[-1].remove()
        lines.pop()

    # Update particle positions and colors
    for i in range(len(x_frame)):  # Iterate only over valid indices
        if i < len(m_frame):  # Ensure m_frame has a mass for this particle
            lines[i].set_data([x_frame[i]], [y_frame[i]])
            lines[i].set_3d_properties([z_frame[i]])
            
            # Check if the particle is a black hole
            if m_frame[i] > bh_threshold:
                lines[i].set_color('black')  # Set color to black for black holes
                lines[i].set_markersize(bh_marker_size)  # Increased size for BHs
            else:
                # Assign color based on cluster or other criteria
                if i < len(x1_stars):  # First cluster
                    lines[i].set_color('green')
                else:  # Second cluster
                    lines[i].set_color('blue')
                lines[i].set_markersize(5)  # Default dot size for stars
        else:
            # Hide the line if the particle no longer exists
            lines[i].set_data([], [])
            lines[i].set_3d_properties([])
        
    return lines

# Create animation
ani = FuncAnimation(fig, update, frames=n, init_func=init, fargs=(m,), blit=True, interval=10)
plt.show()

# Plot all energies on one graph
plt.figure()
plt.plot(total_energy, label='Total Energy')
plt.plot(total_kinetic, label='Kinetic Energy')
plt.plot(total_gravitational, label='Gravitational Potential Energy')
plt.title('Total System Energy (J)')
plt.xlabel('Time Step')
plt.ylabel('Energy (J)')
plt.legend()
plt.show()

# Plot momentum
plt.figure()
plt.plot([np.linalg.norm(p) for p in total_momentum])
plt.title('Total System Momentum Time Evolution')
plt.xlabel('Time Step')
plt.ylabel('Momentum (kg m/s)')
plt.show()

# Plot virial ratio
plt.figure()
plt.plot(virial_ratio, label='Virial Ratio')
plt.axhline(y=1, color='r', linestyle='-', label='Equilibrium')
plt.title("Virial Ratio (2K/|U|) Over Simulation")
plt.ylabel("Virial Ratio")
plt.xlabel("Sampling Step")
plt.show()

# Track number of black holes during simulation
plt.figure()
plt.plot(bh_count)
plt.title("Number of black holes in simulation")
plt.ylabel("Black holes")
plt.xlabel("Time Step")
plt.show()

# Plot BH masses over time
time_steps = np.arange(len(bh_masses)) * dt / year  # Convert time steps to years

plt.figure(figsize=(8, 5))
for i in range(bh_masses.shape[1]):  # Iterate over BHs
    plt.plot(time_steps, (bh_masses[:, i]/M_sun), label=f"BH {i+1}")
plt.xlabel("Time (Years)")
plt.ylabel("Black Hole Mass (M_sun)")
plt.title("Black Hole Mass Evolution Over Time")
plt.legend()
plt.grid()
plt.show()
