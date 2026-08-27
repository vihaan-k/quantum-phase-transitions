import numpy as np
import random
from numba import njit

@njit
def downsample_lattice(
    lattice: np.ndarray, 
    target_x: int = 4, 
    target_y: int = 4
) -> np.ndarray:
    L_y, L_x = lattice.shape
    if L_y % target_y != 0 or L_x % target_x != 0:
        raise ValueError(
            f"Lattice dimensions ({L_y}x{L_x}) must be evenly",
            f"divisible by target_size ({target_x}x{target_y})."
        )

    new_x = L_x // target_x
    new_y = L_y // target_y

    out = np.zeros((target_y, target_x))
    for y in range(target_y):
        for x in range(target_x):
            block_sum = 0
            for i in range(new_y):
                for j in range(new_x):
                    block_sum += lattice[y * new_y + i, x * new_x + j]

            if block_sum > 0:
                out[y,x] = 1
            elif block_sum < 0:
                out[y,x] = -1
            else:
                out[y,x] = np.random.choice(np.array([-1,1]))
    return out

def batch_downsample(
    dataset: np.ndarray, target_x: int = 4, target_y: int = 4
) -> np.ndarray:
    return np.array(
        [downsample_lattice(grid, target_x, target_y) for grid in dataset]
    )