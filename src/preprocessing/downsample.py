import numpy as np

def downsample_lattice(lattice: np.ndarray, block_size: int = 4) -> np.ndarray:
    """Downsamples an (L x L) Ising spin lattice into a 

    (L/block_size x L/block_size) lattice via block averaging.

    Parameters
    ----------
    lattice : np.ndarray
        Input 2D spin grid with values in {-1, +1}. Shape must be divisible by
        block_size.
    block_size : int, optional
        Linear size of the non-overlapping square blocks (default is 4).

    Returns
    -------
    np.ndarray
        Downsampled 2D spin grid with values in {-1, +1}.
    """
    L_y, L_x = lattice.shape
    if L_y % block_size != 0 or L_x % block_size != 0:
        raise ValueError(
            f"Lattice dimensions ({L_y}x{L_x}) must be evenly divisible by block_size ({block_size})."
        )

    new_y = L_y // block_size
    new_x = L_x // block_size

    # Reshape into a 4D array isolating non-overlapping spatial blocks
    blocks = lattice.reshape(new_y, block_size, new_x, block_size)

    # Calculate average spin value per block
    block_means = blocks.mean(axis=(1, 3))

    # Resolve tie-breaks (mean == 0) deterministically to +1
    downsampled = np.where(block_means >= 0, 1, -1)

    return downsampled


def batch_downsample(
    dataset: np.ndarray, block_size: int = 4
) -> np.ndarray:
    """Applies downsampling across a batch of spin configurations.

    Parameters
    ----------
    dataset : np.ndarray
        3D array of shape (N_samples, L, L) representing spin matrices.
    block_size : int, optional
        Linear size of the downsampling block (default is 4).

    Returns
    -------
    np.ndarray
        Downsampled dataset of shape (N_samples, L//block_size, L//block_size).
    """
    return np.array(
        [downsample_lattice(grid, block_size) for grid in dataset]
    )   