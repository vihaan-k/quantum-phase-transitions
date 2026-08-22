import numpy as np

def create_lattice(L=8):
    """Create a 2D square lattice of spins with random initial configuration."""
    return np.random.choice([-1, 1], size=(L, L))

def total_energy(spin_grid, J=1.0):
    """Compute the 2D Ising Hamiltonian with periodic boundaries."""
    # Sum interactions with right and down neighbors (handles periodic boundaries automatically)
    right_neighbors = np.roll(spin_grid, -1, axis=1)
    down_neighbors = np.roll(spin_grid, -1, axis=0)
    return -J * np.sum(spin_grid * (right_neighbors + down_neighbors))

def magnetization(spin_grid):
    """Compute the magnetization of the spin grid."""
    return np.sum(spin_grid) / spin_grid.size   