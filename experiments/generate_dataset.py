import os
import time
import numpy as np
from ising.simulation import simulate
from ising.lattice import create_lattice
from preprocessing.downsample import batch_downsample

def generate_ising_dataset(
    temperatures: np.ndarray,
    samples_per_temp: int,
    lattice_size: int = 16,
    downsampled_size: int = 4,
    n_equilibrations: int = 1000,
    n_measurements: int = 20,
    tc: float = 2.269,
    seed: int = 42
):
    """
    Generates thermalized 2D Ising lattice configurations and downsampled pairs
    across a range of temperatures.
    """
    np.random.seed(seed)
    num_temps = len(temperatures)
    total_samples = num_temps * samples_per_temp

    print(f"--- Generating Ising Dataset ---")
    print(f"Temperatures: {temperatures.min():.1f} to {temperatures.max():.1f} (Step: {temperatures[1]-temperatures[0]:.1f})")
    print(f"Samples per T: {samples_per_temp} | Total Samples: {total_samples}")
    print(f"Lattice sizes: Full ({lattice_size}x{lattice_size}) -> Downsampled ({downsampled_size}x{downsampled_size})")

    # Allocate memory using int8 to optimize storage space
    x_full_list = []
    x_down_list = []
    temp_list = []
    label_list = []

    start_time = time.time()

    for T in temperatures:
        # 1. Initialize random lattice at +1 / -1
        lattice = create_lattice(L=lattice_size).astype(np.int8)

        lattice, _, _ = simulate(
            spin_grid=lattice,
            temperature=T,
            n_equilibrations=n_equilibrations,
            n_measurements=0
        )
        # 2. Collect decorrelated samples
        temp_samples = []
        for _ in range(samples_per_temp):
            lattice, _, _ = simulate(
                spin_grid=lattice,
                temperature=T,
                n_equilibrations=0,
                n_measurements=n_measurements
            )
            temp_samples.append(lattice.copy())

        # 3. Create full batch and downsampled batch as arrays
        full_batch = np.array(temp_samples, dtype=np.int8)
        down_batch = batch_downsample(
            full_batch, 
            target_x=downsampled_size, 
            target_y=downsampled_size
        ).astype(np.int8)

        # 4. Append to overall dataset lists
        x_full_list.append(full_batch)
        x_down_list.append(down_batch)
        temp_list.extend([T] * samples_per_temp)
        
        # Binary label: 0 for Ferromagnetic (T < Tc), 1 for Paramagnetic (T > Tc)
        binary_label = 0 if T < tc else 1
        label_list.extend([binary_label] * samples_per_temp)

    # Concatenate all temperature batches into single dataset arrays
    X_full = np.concatenate(x_full_list, axis=0)
    X_down = np.concatenate(x_down_list, axis=0)
    temperatures_arr = np.array(temp_list, dtype=np.float32)
    labels_arr = np.array(label_list, dtype=np.int8)

    elapsed_time = time.time() - start_time
    print(f"\nGeneration completed in {elapsed_time:.2f} seconds.")
    print(f"X_full shape: {X_full.shape} | X_down shape: {X_down.shape}")

    return X_full, X_down, temperatures_arr, labels_arr


if __name__ == "__main__":
    # --- Configuration ---
    TEMPERATURES = np.arange(0.5, 4.05, 0.1)  # 0.5 to 4.0 (36 temperature steps)
    SAMPLES_PER_TEMP = 500  # Set to 10 for testing execution speed; change to 500 for production
    
    # Save directory setup
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Output path based on run scale
    output_filename = os.path.join(DATA_DIR, f"ising_dataset.npz")

    # --- Run Generation ---
    X_full, X_down, temps, labels = generate_ising_dataset(
        temperatures=TEMPERATURES,
        samples_per_temp=SAMPLES_PER_TEMP,
        lattice_size=16,
        downsampled_size=4,
        n_equilibrations=1000,
        n_measurements=20
    )

    # --- Save Compressed Dataset ---
    np.savez_compressed(
        output_filename,
        X_full=X_full,
        X_down=X_down,
        temperatures=temps,
        labels=labels
    )
    
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Dataset successfully saved to: {output_filename}")
    print(f"Saved file size: {file_size_mb:.3f} MB")