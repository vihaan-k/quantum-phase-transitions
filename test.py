import numpy as np
from ising import simulate, create_lattice

spins = create_lattice(L=8)
spins, H, M = simulate(spins, temperature=5.0, n_sweeps=0)
print("Initial spins:")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")

spins, H, M = simulate(spins, temperature=5.0, n_sweeps=1000)

print("\nAfter simulation (1000 sweeps):")
print(spins)
print(f"Magnetization M = {M}")
print(f"Energy H = {H}")
