# src/utils/generate_feature_importance.py

import os
import pandas as pd
from sklearn.inspection import permutation_importance


def generate_feature_importance(model_name, best_model, X_train, y_train, feature_names, scoring, output_dir):
    """
    Compute and save feature importance scores for a given regression model.

    The function determines feature importance based on the model type:
    - For linear models, it uses the absolute value of model coefficients.
    - For tree-based models, it uses the model's `feature_importances_` attribute.
    - For other models, it falls back to permutation importance using a specified scoring metric.

    The computed feature importances are saved to a text file in the specified output directory.

    Args:
        model_name (str): Name of the regression model (used for labeling the output file).
        model (object): Trained regression model with a `predict` method.
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training target values.
        feature_names (list): List of feature names corresponding to the columns in X_train.
        scoring (str): Scoring metric for permutation importance (default is 'r2').
        output_dir (str): Directory where the output file will be saved.

    Returns:
        None
    """
    print(f"      └── Generating feature importance scores...")
    try:
        # Check if the model has feature importances or coefficients
        if hasattr(best_model, "coef_"):
            # Logistic Regression
            feature_importances = pd.Series(
                abs(best_model.coef_[0]),
                index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(best_model, "feature_importances_"):
            # Tree-based models
            feature_importances = pd.Series(
                best_model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)
        else:
            # Use permutation importance for other models
            perm_importance = permutation_importance(
                best_model, X_train, y_train, scoring=scoring, n_repeats=10, random_state=42
            )
            feature_importances = pd.Series(
                perm_importance.importances_mean,
                index=feature_names
            ).sort_values(ascending=False)

        # Save feature importance to a file
        os.makedirs(f"{output_dir}/feature_importance", exist_ok=True)
        file_path = f"{output_dir}/feature_importance/feature_importances_{model_name.replace(' ', '_').lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"--- 📊 Feature Importance Scores for {model_name}:---\n")
            f.write(feature_importances.to_string())

    except Exception as e:
        print(f"      └── ⚠️  Could not compute or save feature importance for {model_name}: {str(e)}")