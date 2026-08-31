import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prep_data(filepath: str, seed: int = 42):
    """
    Loads Ising dataset, flattens spatial grid inputs,
    and returns an 85/15 (train/test) split for cross-validation workflows.
    """
    data = np.load(filepath)
    
    X_full = data["X_full"]      # (N, 16, 16)
    X_down = data["X_down"]      # (N, 4, 4)
    temperatures = data["temperatures"]
    labels = data["labels"]      # Binary phase labels (0: Low T, 1: High T)

    N = X_full.shape[0]
    X_full_flat = X_full.reshape(N, -1)
    X_down_flat = X_down.reshape(N, -1)

    # 85% Train (for K-Fold CV) / 15% Held-Out Test
    split_data = train_test_split(
        X_full_flat,
        X_down_flat,
        labels,
        temperatures,
        test_size=0.15,
        stratify=labels,
        random_state=seed
    )
    
    X_f_train, X_f_test, X_d_train, X_d_test, y_train, y_test, temp_train, temp_test = split_data

    return {
        "X_full_train": X_f_train, "X_full_test": X_f_test,
        "X_down_train": X_d_train, "X_down_test": X_d_test,
        "y_train": y_train, "y_test": y_test,
        "temp_train": temp_train, "temp_test": temp_test
    }