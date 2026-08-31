import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from preprocessing.dataset import load_and_prep_data

def coarse_to_fine_search(X, y, dataset_label="Dataset"):
    """
    Executes a two-stage hyperparameter search:
    1. Coarse log-scale sweep.
    2. Fine log-scale sweep centered around the winning coarse parameter.
    """
    print(f"\n--- Starting Search for {dataset_label} ---")
    
    # Stage 1: Coarse Search
    coarse_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    gs_coarse = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=coarse_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    gs_coarse.fit(X, y)
    
    best_coarse_c = gs_coarse.best_params_['C']
    print(f"[Stage 1] Coarse Best C: {best_coarse_c:<8} | Val Accuracy: {gs_coarse.best_score_ * 100:.2f}%")

    # Stage 2: Fine Log-Scale Search (0.1x to 10x of coarse winner)
    fine_min_log = np.log10(max(0.0001, best_coarse_c * 0.1))
    fine_max_log = np.log10(min(best_coarse_c * 10.0, 10000.0))
    fine_grid = {'C': np.logspace(fine_min_log, fine_max_log, 10)}

    gs_fine = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=fine_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    gs_fine.fit(X, y)

    best_fine_c = gs_fine.best_params_['C']
    print(f"[Stage 2] Refined Best C: {best_fine_c:<8.4f} | Val Accuracy: {gs_fine.best_score_ * 100:.2f}%")

    return gs_fine.best_estimator_, best_fine_c, gs_fine.best_score_

def train_logistic_regression():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "data", "ising_dataset.npz")
    
    results_dir = os.path.join(base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    baselines_res_dir = os.path.join(results_dir, "baselines")
    baselines_fig_dir = os.path.join(figures_dir, "baselines")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(baselines_res_dir, exist_ok=True)
    os.makedirs(baselines_fig_dir, exist_ok=True)

    data = load_and_prep_data(data_path)

    # Calculate absolute magnetization |M| features
    M_train = np.abs(data["X_full_train"].mean(axis=1, keepdims=True))
    M_test = np.abs(data["X_full_test"].mean(axis=1, keepdims=True))

    M_down_train = np.abs(data["X_down_train"].mean(axis=1, keepdims=True))
    M_down_test = np.abs(data["X_down_test"].mean(axis=1, keepdims=True))

    # Define feature dictionary
    feature_sets = {
        "16x16 Raw Spins (Option 1)": (data["X_full_train"], data["X_full_test"]),
        "16x16 |M| Only (Option 2)": (M_train, M_test),
        "16x16 Spins + |M| (Option 3)": (np.hstack([data["X_full_train"], M_train]), np.hstack([data["X_full_test"], M_test])),
        "4x4 Raw Spins (Option 1)": (data["X_down_train"], data["X_down_test"]),
        "4x4 |M| Only (Option 2)": (M_down_train, M_down_test),
        "4x4 Spins + |M| (Option 3)": (np.hstack([data["X_down_train"], M_down_train]), np.hstack([data["X_down_test"], M_down_test]))
    }

    print("=== Feature Representation Comparison for Logistic Regression ===")

    summary_results = {}
    best_overall_model = None
    best_overall_acc = 0.0

    # 1. Evaluate all feature configurations
    for label, (X_tr, X_te) in feature_sets.items():
        model, best_c, cv_acc = coarse_to_fine_search(X_tr, data["y_train"], dataset_label=label)
        test_acc = model.score(X_te, data["y_test"])
        summary_results[label] = (best_c, cv_acc, test_acc)
        
        # Track highest accuracy model for visualization
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_overall_model = (model, X_te, label, best_c)

    # 2. Save complete summary to text file
    summary_path = os.path.join(baselines_res_dir, "logistic_regression_results.txt")
    with open(summary_path, "w") as f:
        f.write("=== LOGISTIC REGRESSION FEATURE COMPARISON RESULTS ===\n\n")
        for label, (c_val, cv_acc, test_acc) in summary_results.items():
            f.write(f"{label}:\n")
            f.write(f"  Best Hyperparameter C:   {c_val:.4f}\n")
            f.write(f"  Mean 5-Fold CV Accuracy: {cv_acc * 100:.2f}%\n")
            f.write(f"  Test Set Accuracy:       {test_acc * 100:.2f}%\n\n")

    print(f"\n[Saved] Metrics output saved to: {summary_path}")

    # 3. Generate phase prediction curve for the top model
    best_model, X_te_best, best_label, best_c = best_overall_model
    probs = best_model.predict_proba(X_te_best)[:, 1]
    unique_temps = np.unique(data["temp_test"])
    avg_probs = [probs[data["temp_test"] == t].mean() for t in unique_temps]

    plt.figure(figsize=(8, 5))
    plt.plot(unique_temps, avg_probs, 'o-', label=f"{best_label} (C={best_c:.2f})")
    plt.axvline(x=2.269, color='r', linestyle='--', label="Critical Temp Tc = 2.269")
    plt.xlabel("Temperature (T)")
    plt.ylabel("Predicted Probability P(High T Phase)")
    plt.title("Phase Prediction Probability vs Temperature (Top Baseline)")
    plt.legend()
    plt.grid(True)
    
    figure_path = os.path.join(baselines_fig_dir, "logistic_regression_phase_pred.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Prediction figure saved to: {figure_path}")

if __name__ == "__main__":
    train_logistic_regression()