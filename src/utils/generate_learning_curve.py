# src/utils/generate_learning_curve.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold

def generate_learning_curve(
    task_type, model_name, model,
    X_train, y_train,
    scoring='r2', random_state=42, n_jobs=-1, output_dir='output'
):
    """
    Generate and save a learning curve plot for classification or regression models. A learning curve shows how model performance varies as a function of training set size. This helps to diagnose underfitting (high bias) or overfitting (high variance).

    The function:
    - Automatically detects classification or regression based on y_train type.
    - Uses StratifiedKFold for classification, KFold for regression.
    - Plots training and cross-validation scores with shaded standard deviations.
    - Saves the plot as a PNG in the specified output directory.

    Args:
        task_type (str): Type of task ('classification' or 'regression').
        model_name (str): Name of the model for labeling.
        model (object): Model object that implements fit and predict.
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target vector for training.
        scoring (str): Scoring metric (e.g., 'f1', 'accuracy' for classification; 'r2', 'neg_mean_squared_error' for regression).
        random_state (int): Seed for reproducibility.
        n_jobs (int): Number of jobs to run in parallel.
        output_dir (str): Output directory for saving the plot.

    Returns:
        None. Saves the learning curve plot as a PNG file.
    """
    try:
        print("      └── Generating and plotting learning curve...")

        # Choose appropriate CV strategy
        if task_type == "classification":
            # Check that all classes have enough samples
            unique, counts = np.unique(y_train, return_counts=True)
            if np.min(counts) < 2:
                print("      └── ⚠️  Not enough samples for some classes. Using KFold instead of StratifiedKFold.")
                cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
            else:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Define training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)

        # Compute learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )

        # Compute mean and std dev
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="r")
        plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="r", label="Cross-validation score")

        plt.title(f"Learning Curves - {model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Training Examples", fontsize=12)
        plt.ylabel(scoring.upper(), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best", fontsize=10)
        plt.tight_layout()

        # Save plot
        os.makedirs(f"{output_dir}/learning_curves", exist_ok=True)
        file_path = f"{output_dir}/learning_curves/learning_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"      └── ⚠️ Failed to plot learning curves for {model_name}: {e}")
