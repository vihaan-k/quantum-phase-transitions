import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Step back 2 levels: 01_visualize_downsampling.py -> experiments -> project_root
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing import batch_downsample
from src.ising import simulate 

def run_and_visualize_downsampling(
    lattice_size: int = 16, block_size: int = 4
) -> None:
    """Generates Ising lattices across temperature regimes, applies 

    block-downsampling, and visualizes the shape transformations.
    """
    temperatures = [1.5, 2.27, 3.5]  # Low T, Near-Critical T, High T

    original_lattice = np.random.choice([-1, 1], size=(lattice_size, lattice_size))
    original_lattices = []

    print("--- Running Ising Simulations ---")
    for T in temperatures:
        simulation = simulate(
            spin_grid=original_lattice.copy(), 
            temperature=T
        )
        original_lattices.append(simulation[0])

    original_batch = np.array(original_lattices)

    # Apply batch downsampling
    downsampled_batch = batch_downsample(
        original_batch, target_x=block_size, target_y=block_size
    )

    print(f"Original batch shape:    {original_batch.shape}")
    print(f"Downsampled batch shape: {downsampled_batch.shape}\n\n")
    print(f"Original lattices:\n{original_batch}\n\n\n")
    print(f"Downsampled lattices:\n{downsampled_batch}\n\n\n")


    num_samples = len(original_batch)
    _, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        # Top Row: Original Lattices
        axes[0, i].imshow(original_batch[i], cmap='binary', vmin=-1, vmax=1, interpolation='none')
        axes[0, i].set_title(f"Original #{i+1}")
        axes[0, i].axis('off')

        # Bottom Row: Downsampled Lattices
        axes[1, i].imshow(
            downsampled_batch[i], 
            cmap='binary', 
            vmin=-1, 
            vmax=1, 
            interpolation='none'
        )
        axes[1, i].set_title(f"Downsampled #{i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.xticks(np.arange(-0.5, 4, 1), labels=[])
    plt.yticks(np.arange(-0.5, 4, 1), labels=[])
    plt.grid(color='gray', linestyle='-', linewidth=1)

    # Save and render
    out_dir = Path("results/figures/downsampling")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "downsampling_visualization.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_and_visualize_downsampling()