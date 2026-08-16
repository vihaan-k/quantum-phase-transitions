import numpy as np

Lx, Ly = 8, 8
spins = np.random.choice([-1, 1], size=(Lx, Ly))
J = 1.0
T = 2.0


def total_energy(spin_grid):
    """Compute the 2D Ising Hamiltonian with periodic boundaries."""
    energy = 0.0
    for i in range(Lx):
        for j in range(Ly):
            energy += -J * spin_grid[i, j] * (
                spin_grid[i, (j + 1) % Ly] + spin_grid[(i + 1) % Lx, j]
            )
    return energy


def metropolis_single_update(spin_grid, temperature):
    """Perform one Metropolis update: pick one random spin and attempt a flip."""
    i = np.random.randint(Lx)
    j = np.random.randint(Ly)

    old_spin = spin_grid[i, j]
    new_spin = -old_spin

    # Local energy change from flipping one spin
    # ΔE = E_new - E_old = 2 * s_i * (sum of neighbor spins) for J=1 in this convention
    neighbors = (
        spin_grid[i, (j + 1) % Ly]
        + spin_grid[i, (j - 1) % Ly]
        + spin_grid[(i + 1) % Lx, j]
        + spin_grid[(i - 1) % Lx, j]
    )
    delta_E = 2 * old_spin * neighbors

    if delta_E <= 0:
        spin_grid[i, j] = new_spin
    else:
        accept_prob = np.exp(-delta_E / temperature)
        if np.random.random() < accept_prob:
            spin_grid[i, j] = new_spin

    return spin_grid


H = total_energy(spins)
M = np.sum(spins) / spins.size

print("Initial spins:")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")

spins = metropolis_single_update(spins, T)

H = total_energy(spins)
M = np.sum(spins) / spins.size

print("\nAfter one Metropolis update:")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")
