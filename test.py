import numpy as np

spins = np.random.choice([-1, 1], size=(8,8))  # Random initial configuration of spins
J = 1.0
T = 1.0


def total_energy(spin_grid):
    """Compute the 2D Ising Hamiltonian with periodic boundaries."""
    energy = 0.0
    for i in range(spin_grid.shape[0]):
        for j in range(spin_grid.shape[1]):
            energy += -J * spin_grid[i, j] * (
                spin_grid[i, (j + 1) % spin_grid.shape[1]]
                + spin_grid[(i + 1) % spin_grid.shape[0], j]
            )
    return energy


def metropolis_update(spin_grid, temperature):
    """Perform one Metropolis update: pick one random spin and attempt a flip."""
    i = np.random.randint(spin_grid.shape[0])
    j = np.random.randint(spin_grid.shape[1])

    s_old = spin_grid[i, j]
    neighbors = (
        spin_grid[i, (j + 1) % spin_grid.shape[1]]
        + spin_grid[i, (j - 1) % spin_grid.shape[1]]
        + spin_grid[(i + 1) % spin_grid.shape[0], j]
        + spin_grid[(i - 1) % spin_grid.shape[0], j]
    )

    # For J=1, flipping a spin changes the energy by ΔE = 2 s_i * sum(neighbors).
    delta_E = 2 * s_old * neighbors

    if delta_E <= 0:
        spin_grid[i, j] = -s_old
    else:
        accept_prob = np.exp(-delta_E / temperature)
        if np.random.random() < accept_prob:
            spin_grid[i, j] = -s_old

    return spin_grid


def sweep(spin_grid, temperature):
    """Perform roughly one random update per spin in the lattice."""
    n_updates = spin_grid.size
    for _ in range(n_updates):
        spin_grid = metropolis_update(spin_grid, temperature)
    return spin_grid

def simulate(spin_grid, temperature, n_sweeps):
    """Run the simulation for a given number of sweeps."""
    for _ in range(n_sweeps):
        spin_grid = sweep(spin_grid, temperature)
    return spin_grid


H = total_energy(spins)
M = np.sum(spins) / spins.size

print("Initial spins:")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")

spins = simulate(spins, T, n_sweeps=1000)
H = total_energy(spins)
M = np.sum(spins) / spins.size

print("\nAfter simulation (1000 sweeps):")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")
