import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
from numba import njit, prange

# Constants
G = 6.67430e-11      # gravitational constant (N m^2 kg^-2)
AU = 1.496e11        # astronomical unit (m)
kpc = 2.063e8 * AU   # kpc (m)
M_sun = 1.989e30     # Mass of Sun (kg)

day = 24 * 60 * 60        # day in seconds (s)
year = 365 * day          # one year (s)
total_time = 5e6 * year   # total simulation time (s)
dt = 5e4 * year           # timestep (s)
n = int(total_time / dt)  # number of timesteps [n=365]
print(n, "time steps")

R = 10 * kpc              # radius of the sphere particle positions are generated in
stick_distance = 400 * AU # distance within which particles will 'stick'

#--------------------------------------------------------------------------
# INITIAL CONDITIONS
# Parameters for the galaxies
M_stars = 1e11 * M_sun      # Total stellar mass in M_sun
M_dark = 5e12 * M_sun       # Total dark matter mass in M_sun
a = 2.0 * kpc               # Scale radius for stellar component in kpc
r_s = 8.0 * kpc             # Scale radius for dark matter halo in kpc
N_stars = int(1e3)          # Number of star particles per galaxy
N_dark = int(8e3)           # Number of dark matter particles per galaxy
ellipticity = 1           # Flattening parameter (0 = spherical, 1 = highly flattened)

# Number of bodies in the simulation
k = 2 * (N_stars + N_dark)

# Hernquist profile: Probability density function (PDF) for sampling
def sample_hernquist(N, a):
    u = np.random.uniform(0, 1, N)
    return a * np.sqrt(u) / (1 - np.sqrt(u))

# NFW profile: Probability density function (PDF) for sampling
def sample_nfw(N, r_s):
    u = np.random.uniform(0, 1, N)
    return r_s * (1 / (1 - u) - 1)

# Sample particle radii
r_stars = sample_hernquist(N_stars, a) / 200  # Radii for star particles
r_dark = sample_nfw(N_dark, r_s) / 200        # Radii for dark matter particles

# Assign particle masses
m_star = M_stars / N_stars  # Mass per star particle
m_dark = M_dark / N_dark    # Mass per dark matter particle

# Central black hole
central_bh = 2e39           # black hole mass (kg) [10^9 M_sun]

# Distribute particles in 3D space with ellipticity and containment check
def distribute_particles(r_values, ellipticity, N, R):
    x, y, z = [], [], []
    for i in range(N):
        while True:
            r = r_values[i]
            if r > R:
                r = np.random.uniform(0, R)
           
            # Sample spherical coordinates
            theta = np.arccos(2 * np.random.uniform(0, 1) - 1)  # Polar angle
            phi = np.random.uniform(0, 2 * np.pi)               # Azimuthal angle

            # Apply ellipticity (flatten along the z-axis)
            z_new = r * np.cos(theta) * np.sqrt(1 - ellipticity**2)
            x_new = r * np.sin(theta) * np.cos(phi)
            y_new = r * np.sin(theta) * np.sin(phi)

            # Check if the particle is within the sphere of radius R
            if np.sqrt(x_new**2 + y_new**2 + z_new**2) <= R:
                x.append(x_new)
                y.append(y_new)
                z.append(z_new)
                break
               
    return np.array(x), np.array(y), np.array(z)

# Generate star and dark matter particle positions
x1_stars, y1_stars, z1_stars = distribute_particles(r_stars, ellipticity, N_stars, R)
x1_dark, y1_dark, z1_dark = distribute_particles(r_dark, ellipticity, N_dark, R)

# Generate positions for galaxy 2
x2_stars, y2_stars, z2_stars = distribute_particles(r_stars, ellipticity, N_stars, R)
x2_dark, y2_dark, z2_dark = distribute_particles(r_dark, ellipticity, N_dark, R)

# Offset galaxy 2's positions to ensure they are separate initially
x2_stars += 10 * kpc  # shift to the right
x2_dark += 10 * kpc   # shift to the right

# Combine star and dark matter particles for both galaxies
x0 = np.concatenate([x1_stars, x1_dark, [0], x2_stars, x2_dark, [0]])
y0 = np.concatenate([y1_stars, y1_dark, [0], y2_stars, y2_dark, [0]])
z0 = np.concatenate([z1_stars, z1_dark, [0], z2_stars, z2_dark, [0]])
m = np.concatenate([np.full(N_stars, m_star), np.full(N_dark, m_dark), [central_bh],
                    np.full(N_stars, m_star), np.full(N_dark, m_dark), [central_bh]])

# Center of Mass Calculation
Rcom_x = np.sum(m * x0) / np.sum(m)
Rcom_y = np.sum(m * y0) / np.sum(m)
Rcom_z = np.sum(m * z0) / np.sum(m)
x0 -= Rcom_x
y0 -= Rcom_y
z0 -= Rcom_z

# Initial velocities (m/s)
vx0 = np.concatenate([np.full(N_stars + N_dark, 2e5), [0], np.full(N_stars + N_dark, -2e5), [0]])  # Galaxy 1 to the left, Galaxy 2 to the right [200 km/s]
vy0 = np.zeros(len(m))
vz0 = np.zeros(len(m))

# Calculate the center of mass velocity
Vcom_x = np.sum(m * vx0) / np.sum(m)
Vcom_y = np.sum(m * vy0) / np.sum(m)
Vcom_z = np.sum(m * vz0) / np.sum(m)
vx0 = vx0 - Vcom_x
vy0 = vy0 - Vcom_y
vz0 = vz0 - Vcom_z

# Arrays to save states
x = [x0]
y = [y0]
z = [z0]
ux = [vx0]
uy = [vy0]
uz = [vz0]
t = [0]

# Softening parameter
epsilon = 0.1 * R  # example value, adjust as needed (recommended 0.1R)

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
measurement_points = 10
measurement_interval = n // measurement_points
stored_states = []

# Initialize half-step velocities
ax, ay, az = compute_accelerations(x0, y0, z0, m, G, epsilon)
ux_half = vx0 + ax * (dt / 2)
uy_half = vy0 + ay * (dt / 2)
uz_half = vz0 + az * (dt / 2)

for i in range(n):
    # Implement Leapfrog integrator:
    # Update positions using half-step velocities
    x_new = x[-1] + ux_half * dt
    y_new = y[-1] + uy_half * dt
    z_new = z[-1] + uz_half * dt

    # Compute new accelerations
    ax_new, ay_new, az_new = compute_accelerations(x_new, y_new, z_new, m, G, epsilon)

    # Update velocities by another half step
    ux_half = ux_half + ax_new * dt
    uy_half = uy_half + ay_new * dt
    uz_half = uz_half + az_new * dt

    # Store full-step velocities for output
    ux_new = ux_half - ax_new * (dt / 2)
    uy_new = uy_half - ay_new * (dt / 2)
    uz_new = uz_half - az_new * (dt / 2)

    # Collision detection using a kd-tree
    tree = cKDTree(np.column_stack((x_new, y_new, z_new)))
    pairs = tree.query_pairs(stick_distance)
    to_remove = set()
    for j, l in pairs:
        if j not in to_remove and l not in to_remove:
            # Combine masses
            m[j] += m[l]
            # Combine momenta
            ux_new[j] = (m[j] * ux_new[j] + m[l] * ux_new[l]) / m[j]
            uy_new[j] = (m[j] * uy_new[j] + m[l] * uy_new[l]) / m[j]
            uz_new[j] = (m[j] * uz_new[j] + m[l] * uz_new[l]) / m[j]
            # Mark the particle for removal
            to_remove.add(l)

    # Remove the stuck particles
    to_remove = sorted(to_remove, reverse=True)
    for idx in to_remove:
        m = np.delete(m, idx)
        x_new = np.delete(x_new, idx)
        y_new = np.delete(y_new, idx)
        z_new = np.delete(z_new, idx)
        ux_new = np.delete(ux_new, idx)
        uy_new = np.delete(uy_new, idx)
        uz_new = np.delete(uz_new, idx)
        # Also remove from ux_half, uy_half, uz_half
        ux_half = np.delete(ux_half, idx)
        uy_half = np.delete(uy_half, idx)
        uz_half = np.delete(uz_half, idx)

    # Update lists
    x.append(x_new)
    y.append(y_new)
    z.append(z_new)
    ux.append(ux_new)
    uy.append(uy_new)
    uz.append(uz_new)

    # Store system state at measurement intervals
    if i % measurement_interval == 0:
        stored_states.append((x_new, y_new, z_new, ux_new, uy_new, uz_new, m.copy()))

    # Track progress of code as it runs
    print(i)

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
    system_gravitational = compute_gravitational_potential_energy(m, positions, G, epsilon)

    # Append results to lists
    total_momentum.append(system_momentum)
    total_kinetic.append(system_kinetic)
    total_gravitational.append(system_gravitational)

# Calculate virial ratio
virial_ratio = []
virial_theorem = []
total_energy = []
for i in range(len(total_kinetic)):
    virial_ratio.append((2 * total_kinetic[i]) / abs(total_gravitational[i]))
    total_energy.append(total_kinetic[i] + total_gravitational[i])
    virial_theorem.append(2*total_kinetic[i] + total_gravitational[i])
#print("Virial Ratio (2K/|U|) [should tend to 1]:", virial_ratio)
print("Virial theorem is", virial_theorem)

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
plt.plot([np.linalg.norm(p) for p in total_momentum])
plt.title('Total System Momentum Time Evolution')
plt.xlabel('Time Step')
plt.ylabel('Momentum (kg m/s)')
plt.show()

plt.plot(virial_ratio)
plt.title("Virial Ratio (2K/|U|) Over Simulation")
plt.ylabel("Virial Ratio")
plt.xlabel("Sampling Step")
plt.show()


#--------------------------------------------------------------------------
# GRAPH PLOTTING
# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define colors for each galaxy
colors = ['purple', 'blue']

# Create lines with different colors
lines = [ax.plot([], [], [], '.', color=colors[i % len(colors)])[0] for i in range(k)]

# Set dynamic limits for the axes
def set_dynamic_limits(ax, x_data, y_data, z_data):
    x_flat = np.concatenate(x_data)
    y_flat = np.concatenate(y_data)
    z_flat = np.concatenate(z_data)
    ax.set_xlim(np.min(x_flat), np.max(x_flat))
    ax.set_ylim(np.min(y_flat), np.max(y_flat))
    ax.set_zlim(np.min(z_flat), np.max(z_flat))

set_dynamic_limits(ax, x, y, z)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Initialization function
def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

# Update function over time
def update(frame):
    for i, line in enumerate(lines):
        if i < len(x[frame]):  # Check if the particle exists at this frame
            line.set_data([x[frame][i]], [y[frame][i]])
            line.set_3d_properties([z[frame][i]])
        else:
            # Hide the line if the particle no longer exists
            line.set_data([], [])
            line.set_3d_properties([])
    return lines

# Create animation
ani = FuncAnimation(fig, update, frames=n, init_func=init, blit=True, interval=0.1)
plt.show()