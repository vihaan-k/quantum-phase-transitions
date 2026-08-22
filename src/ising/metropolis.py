import numpy as np
from numba import njit

@njit
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