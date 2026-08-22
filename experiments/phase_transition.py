import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ising import create_lattice, simulate

SIZE_COLORS_EQUILIBRATIONS = {
    8: ["C0", 1000],
    16: ["C1", 3000],
    32: ["C2", 5000],
}


def temperature_scan(temperatures, L=8, n_equilibration=1000, n_measurements=1000):
    """Run equilibrium + measurement sweeps at a fixed lattice size."""
    avg_abs_magnetization = []
    avg_energy = []
    susceptibility = []

    lattice = create_lattice(L=L)
    for temperature in temperatures:
        lattice, energies, magnetizations = simulate(
            lattice,
            temperature=temperature,
            n_equilibration=n_equilibration,
            n_measurements=n_measurements,
        )

        avg_abs_magnetization.append(np.mean(np.abs(magnetizations)))
        avg_energy.append(np.mean(energies))
        susceptibility.append(np.var(np.abs(magnetizations)) * (L * L / temperature))

    return (
        np.asarray(temperatures),
        np.asarray(avg_abs_magnetization),
        np.asarray(avg_energy),
        np.asarray(susceptibility),
    )

def scan_all_sizes(temperatures, sizes=(8, 16, 32), n_measurements=1000):
    """Run the temperature scan for several lattice sizes."""
    results = {}
    for L in sizes:
        equilibrations = SIZE_COLORS_EQUILIBRATIONS[L][1]
        results[L] = temperature_scan(
            temperatures,
            L=L,
            n_equilibration=equilibrations,
            n_measurements=n_measurements,
        )
    return results

def plot_energy_vs_temperature(results, output_path):
    """Plot average energy vs temperature for all lattice sizes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for L, (temps, _, avg_energy, _) in results.items():
        ax.plot(
            temps,
            avg_energy,
            marker="o",
            linewidth=2,
            color=SIZE_COLORS_EQUILIBRATIONS[L][0],
            label=f"L={L}",
        )

    ax.axvline(2.269, color="gray", linestyle="--", linewidth=1.0, label=r"$T_c \approx 2.269$")
    ax.set_title(r"Average energy $\langle H \rangle$ vs temperature")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle H \rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_energy_per_spin_vs_temperature(results, output_path):
    """Plot average energy per spin vs temperature for all lattice sizes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for L, (temps, _, avg_energy, _) in results.items():
        energy_per_spin = avg_energy / (L * L)
        ax.plot(
            temps,
            energy_per_spin,
            marker="o",
            linewidth=2,
            color=SIZE_COLORS_EQUILIBRATIONS[L][0],
            label=f"L={L}",
        )

    ax.axvline(2.269, color="gray", linestyle="--", linewidth=1.0, label=r"$T_c \approx 2.269$")
    ax.set_title(r"Average energy per spin $\langle h \rangle$ vs temperature")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle h \rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_magnetization_vs_temperature(results, output_path):
    """Plot average absolute magnetization vs temperature for all lattice sizes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for L, (temps, avg_abs_magnetization, _, _) in results.items():
        ax.plot(
            temps,
            avg_abs_magnetization,
            marker="o",
            linewidth=2,
            color=SIZE_COLORS_EQUILIBRATIONS[L][0],
            label=f"L={L}",
        )

    ax.axvline(2.269, color="gray", linestyle="--", linewidth=1.0, label=r"$T_c \approx 2.269$")
    ax.set_title(r"Average absolute magnetization $\langle |M| \rangle$ vs temperature")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle |M| \rangle$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def plot_susceptibility_vs_temperature(results, output_path):
    """Plot susceptibility vs temperature for all lattice sizes."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for L, (temps, _, _, susceptibility) in results.items():
        ax.plot(
            temps,
            susceptibility,
            marker="o",
            linewidth=2,
            color=SIZE_COLORS_EQUILIBRATIONS[L][0],
            label=f"L={L}",
        )

    ax.axvline(2.269, color="gray", linestyle="--", linewidth=1.0, label=r"$T_c \approx 2.269$")
    ax.set_title(r"Susceptibility $\chi$ vs temperature")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\chi$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    temperatures = np.arange(0.5, 4.1, 0.1)
    sizes = (8,16,32)
    results = scan_all_sizes(
        temperatures,
        sizes=sizes,
        n_measurements=1000,
    )

    output_dir = "results/figures"
    os.makedirs(output_dir, exist_ok=True)
    plot_energy_vs_temperature(results, os.path.join(output_dir, "energy_vs_temperature.png"))
    plot_energy_per_spin_vs_temperature(results, os.path.join(output_dir, "energy_per_spin_vs_temperature.png"))
    plot_magnetization_vs_temperature(results, os.path.join(output_dir, "magnetization_vs_temperature.png"))
    plot_susceptibility_vs_temperature(results, os.path.join(output_dir, "susceptibility_vs_temperature.png"))
    print("Temperature scan complete.")
    for L in sizes:
        temps, avg_abs_m, avg_h, susceptibility = results[L]
        print(f"L={L} -> T={temps[0]} to {temps[-1]}")
        print(f"  <|M|> = {avg_abs_m}")
        print(f"  <H> = {avg_h}")
        print(f"  <h> = {avg_h / (L * L)}")
        print(f"  χ = {susceptibility}\n\n")

if __name__ == "__main__":
    main()