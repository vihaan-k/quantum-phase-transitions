import numpy as np

def create_lattice(L=8):
    """Create a 2D square lattice of spins with random initial configuration."""
    return np.random.choice([-1, 1], size=(L, L))

def total_energy(spin_grid, J=1.0):
    """Compute the 2D Ising Hamiltonian with periodic boundaries."""
    energy = 0.0
    for i in range(spin_grid.shape[0]):
        for j in range(spin_grid.shape[1]):
            energy += -J * spin_grid[i, j] * (
                spin_grid[i, (j + 1) % spin_grid.shape[1]]
                + spin_grid[(i + 1) % spin_grid.shape[0], j]
            )
    return energy

def magnetization(spin_grid):
    """Compute the magnetization of the spin grid."""
    return np.sum(spin_grid) / spin_grid.size   