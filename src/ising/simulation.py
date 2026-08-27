from .metropolis import metropolis_update
from .lattice import total_energy, magnetization
from numba import njit
import numpy as np

@njit
def sweep(spin_grid: np.ndarray, temperature: float) -> np.ndarray:
    """Perform roughly one random update per spin in the lattice."""
    n_updates = spin_grid.size
    for _ in range(n_updates):
        spin_grid = metropolis_update(spin_grid, temperature)
    return spin_grid

def simulate(
        spin_grid: np.ndarray, 
        temperature: float, 
        n_equilibrations: int = 1000, 
        n_measurements: int = 1000
) -> tuple[np.ndarray, list[float], list[float]]:
    """Run an Ising simulation and record equilibrium measurements.
        Returns spin grid, energies, magnetizations.
    """

    # Equilibration
    for _ in range(n_equilibrations):
        spin_grid = sweep(spin_grid, temperature)

    # Measurements
    energies = []
    magnetizations = []

    for _ in range(n_measurements):
        spin_grid = sweep(spin_grid, temperature)

        energies.append(total_energy(spin_grid))
        magnetizations.append(magnetization(spin_grid))

    return spin_grid, energies, magnetizations