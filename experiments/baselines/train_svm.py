import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from preprocessing.dataset import load_and_prep_data

def run_svm_grid_search(X_train, y_train, dataset_label="Dataset"):
    """
    Executes a 5-fold CV hyperparameter search for SVM (C and gamma).
    """
    print(f"\n--- Running LINEAR SVM Search for {dataset_label} ---")

    gs = GridSearchCV(
        estimator = LinearSVC(dual='auto', max_iter=2000),
        param_grid = {'C': [0.0001, 0.01, 1.0, 100.0]},
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    gs.fit(X_train, y_train)

    print(f"Best Params: {gs.best_params_} | Val Accuracy: {gs.best_score_ * 100:.2f}%")
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def train_svm_baselines():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "data", "ising_dataset.npz")
    
    results_dir = os.path.join(base_dir, "results", "baselines")
    os.makedirs(results_dir, exist_ok=True)

    data = load_and_prep_data(data_path)

    # Compute Magnetization |M|
    M_f_train = np.abs(data["X_full_train"].mean(axis=1, keepdims=True))
    M_f_test = np.abs(data["X_full_test"].mean(axis=1, keepdims=True))
    M_d_train = np.abs(data["X_down_train"].mean(axis=1, keepdims=True))
    M_d_test = np.abs(data["X_down_test"].mean(axis=1, keepdims=True))

    # Define side-by-side experiments
    experiments = {
        "16x16 Raw Spins (Linear SVM)": (data["X_full_train"], data["X_full_test"]),
        "16x16 |M| Only (Linear SVM)": (M_f_train, M_f_test),
        "16x16 Spins + |M| (Linear SVM)": (np.hstack([data["X_full_train"], M_f_train]), np.hstack([data["X_full_test"], M_f_test])),
        "4x4 Raw Spins (Linear SVM)": (data["X_down_train"], data["X_down_test"]),
        "4x4 |M| Only (Linear SVM)": (M_d_train, M_d_test),
        "4x4 Spins + |M| (Linear SVM)": (np.hstack([data["X_down_train"], M_d_train]), np.hstack([data["X_down_test"], M_d_test])),
    }

    print("=== SVM Classical Baseline Sweep (Linear vs RBF) ===")

    summary_results = {}
    for label, (X_tr, X_te) in experiments.items():
        model, best_params, cv_acc = run_svm_grid_search(X_tr, data["y_train"], dataset_label=label)
        test_acc = model.score(X_te, data["y_test"])
        summary_results[label] = (best_params, cv_acc, test_acc)

    # Write structured metrics to text file
    summary_path = os.path.join(results_dir, "svm_results.txt")
    with open(summary_path, "w") as f:
        f.write("=== LINEAR SVM FEATURE COMPARISON RESULTS ===\n\n")
        for label, (params, cv_acc, test_acc) in summary_results.items():
            f.write(f"{label}:\n")
            f.write(f"  Best Params:             {params}\n")
            f.write(f"  Mean 5-Fold CV Accuracy: {cv_acc * 100:.2f}%\n")
            f.write(f"  Test Set Accuracy:       {test_acc * 100:.2f}%\n\n")

    print(f"\n[Saved] SVM side-by-side comparison results saved to: {summary_path}")

if __name__ == "__main__":
    train_svm_baselines()