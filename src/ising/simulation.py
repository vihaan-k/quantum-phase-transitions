from .metropolis import metropolis_update
from .lattice import total_energy, magnetization
from numba import njit

@njit
def sweep(spin_grid, temperature):
    """Perform roughly one random update per spin in the lattice."""
    n_updates = spin_grid.size
    for _ in range(n_updates):
        spin_grid = metropolis_update(spin_grid, temperature)
    return spin_grid

def simulate(spin_grid, temperature, n_equilibration=1000, n_measurements=1000):
    """Run an Ising simulation and record equilibrium measurements."""

    # Equilibration
    for _ in range(n_equilibration):
        spin_grid = sweep(spin_grid, temperature)

    # Measurements
    energies = []
    magnetizations = []

    for _ in range(n_measurements):
        spin_grid = sweep(spin_grid, temperature)

        energies.append(total_energy(spin_grid))
        magnetizations.append(magnetization(spin_grid))

    return spin_grid, energies, magnetizations