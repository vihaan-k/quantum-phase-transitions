from .metropolis import metropolis_update
from .lattice import total_energy, magnetization

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
    return spin_grid, total_energy(spin_grid), magnetization(spin_grid)