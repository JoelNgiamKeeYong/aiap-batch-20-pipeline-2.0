# src/evaluate_regression_models.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    median_absolute_error, explained_variance_score
)

from utils import generate_feature_importance, generate_learning_curve, save_evaluation_metrics


def evaluate_regression_models(
        task_type,
        trained_models,
        X_train, X_test, y_train, y_test,
        scoring='r2',
        generate_plots=True,
        random_state=42, n_jobs=-1
):
    """
    Evaluate a list of trained regression models and save evaluation results.
    This function evaluates each trained model by computing regression metrics, generating visualizations, and saving results.

    Args:
        task_type (str): Type of task being evaluated (i.e., "classification" or "regression").
        trained_models (list): 
            List of tuples containing trained models to evaluate. Each tuple includes:
            - model_name (str): Name of the model.
            - best_model (object): The trained model after hyperparameter tuning.
            - training_time (float): Time taken to train the model (in seconds).
            - model_size_kb (float): Size of the saved model file (in KB).
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        X_test (pd.DataFrame or np.ndarray): Testing feature matrix.
        y_train (pd.Series or np.ndarray): Training target variable.
        y_test (pd.Series or np.ndarray): Testing target variable.

    Returns:
        list: Updated list of trained models with evaluation metrics and evaluation time appended to each tuple.
    """
    print(f"\n📊 Evaluating best regression models...")

    # Ensure the output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store results
    results = []

    # Get feature names
    feature_names = X_train.columns.tolist()  

    # Loop through each trained model and evaluate
    for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
        print(f"\n   📋 Evaluating {model_name} model...")
        start_time = time.time()

        # Compute evaluation scores
        print(f"      └── Computing regression performance metrics...")
        formatted_metrics = compute_regression_metrics(model_name, best_model, X_train, y_train, X_test, y_test)
        results.append(formatted_metrics)
        
        # Generate feature importance scores
        generate_feature_importance(model_name, best_model, X_train, y_train, feature_names, scoring, output_dir)

        # Plotting charts and diagnostics
        if generate_plots:
            plot_error_diagnostics(model_name, best_model, X_test, y_test, output_dir)    
            generate_learning_curve(task_type, model_name, best_model, X_train, y_train, scoring, random_state, n_jobs, output_dir)

        # Record evaluation time    
        end_time = time.time()
        evaluation_time = end_time - start_time
        print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds")

        # Update model entry with evaluation results
        trained_models[i] = (model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time)

    # Save consolidated results
    save_evaluation_metrics(results, output_dir)
    print(f"\n💾 Saved evaluation metrics and charts to {output_dir} folder")

    return trained_models

#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
# 📀 COMPUTE REGRESSION METRICS
def compute_regression_metrics(model_name, model, X_train, y_train, X_test, y_test):
    """
    Compute comprehensive regression metrics on training and testing data, then format the results for reporting.

    Metrics included:
    - RMSE (Root Mean Squared Error): Measures average magnitude of errors, sensitive to outliers.
    - MAE (Mean Absolute Error): Average absolute errors, interpretable in original units.
    - Median Absolute Error: Robust measure of error less sensitive to outliers.
    - R² (Coefficient of Determination): Proportion of variance explained by the model.
    - Explained Variance Score: Similar to R² but focuses on variance of errors.
    - MAPE (Mean Absolute Percentage Error): Average absolute percent errors, useful for interpretability.
    - RMSLE (Root Mean Squared Logarithmic Error): Penalizes under-predictions and large relative differences (useful when target spans several orders of magnitude).

    Args:
        model_name: String name of the model for labeling.
        model: Trained regression model with a predict method.
        X_train, y_train: Training feature matrix and target array.
        X_test, y_test: Testing feature matrix and target array.
        
    Returns:
        dict: Formatted dictionary of regression metrics with test values and training values in parentheses.
    """
    def _calc_metrics(X, y_true):
        y_pred = model.predict(X)
        y_true_log = np.log1p(np.maximum(0, y_true))
        y_pred_log = np.log1p(np.maximum(0, y_pred))
        rmsle = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))

        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MedianAE": median_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
            "Explained Variance": explained_variance_score(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "RMSLE": rmsle,
        }

    train_metrics = _calc_metrics(X_train, y_train)
    test_metrics = _calc_metrics(X_test, y_test)

    formatted_metrics = {
        "Model": model_name,
        "RMSE": f"{test_metrics['RMSE']:.3f} ({train_metrics['RMSE']:.3f})",
        "MAE": f"{test_metrics['MAE']:.3f} ({train_metrics['MAE']:.3f})",
        "MedianAE": f"{test_metrics['MedianAE']:.3f} ({train_metrics['MedianAE']:.3f})",
        "R2": f"{test_metrics['R2']:.3f} ({train_metrics['R2']:.3f})",
        "Explained Variance": f"{test_metrics['Explained Variance']:.3f} ({train_metrics['Explained Variance']:.3f})",
        "MAPE": f"{test_metrics['MAPE']:.3f} ({train_metrics['MAPE']:.3f})",
        "RMSLE": f"{test_metrics['RMSLE']:.3f} ({train_metrics['RMSLE']:.3f})",
    }

    return formatted_metrics

#################################################################################################################################
#################################################################################################################################
# 📀 PLOT ERROR DIAGNOSTICS
def plot_error_diagnostics(model_name, model, X_test, y_test, output_dir='output'):
    """
    Generate a 2x2 error diagnostics dashboard with key regression error plots:

    1. Predicted vs Actual: Scatter plot comparing predicted values against actual target values to assess overall model fit and identify bias.
    2. Residuals vs Predicted: Scatter plot of residuals against predicted values to check for patterns or heteroscedasticity.
    3. Error Distribution: Histogram with KDE to visualize the distribution and spread of residuals, assessing normality and outliers.
    4. Prediction Error Plot (Residuals vs Actual): Scatter plot of residuals against actual target values to detect biases or trends.

    Saves the combined dashboard plot to the specified output directory.

    Args:
        model_name (str): Name of the regression model (used for titles and saving files).
        model (object): Trained regression model with a predict method.
        X_test (array-like): Feature data for the test set.
        y_test (array-like): True target values for the test set.
        output_dir (str): Directory path to save the generated plot.

    Returns:
        None. Saves the error diagnostics dashboard as a PNG file in the output directory.
    """
    try:
        print(f"      └── Plotting combined error diagnostics dashboard...")

        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{model_name} Diagnostics", fontsize=16)

        # 1. Predicted vs Actual
        axs[0, 0].scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
        axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axs[0, 0].set_xlabel("Actual Values")
        axs[0, 0].set_ylabel("Predicted Values")
        axs[0, 0].set_title("Predicted vs Actual")
        axs[0, 0].grid(True)

        # 2. Residuals vs Predicted
        axs[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolor="k")
        axs[0, 1].axhline(y=0, color="red", linestyle="--", lw=1)
        axs[0, 1].set_xlabel("Predicted Values")
        axs[0, 1].set_ylabel("Residuals")
        axs[0, 1].set_title("Residuals vs Predicted")
        axs[0, 1].grid(True)

        # 3. Error Distribution
        sns.histplot(residuals, kde=True, ax=axs[1, 0], bins=30, color="skyblue")
        axs[1, 0].set_title("Error Distribution")
        axs[1, 0].set_xlabel("Residuals")
        axs[1, 0].set_ylabel("Frequency")

        # 4. Prediction Error Plot
        axs[1, 1].scatter(y_test, residuals, alpha=0.6, edgecolor="k")
        axs[1, 1].axhline(y=0, color="red", linestyle="--", lw=1)
        axs[1, 1].set_xlabel("Actual Values")
        axs[1, 1].set_ylabel("Prediction Error (Residuals)")
        axs[1, 1].set_title("Prediction Error Plot (Residuals vs Actual)")
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        os.makedirs(f"{output_dir}/error_diagnostics", exist_ok=True)
        file_path = f"{output_dir}/error_diagnostics/error_diagnostics_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"      └── ⚠️ Failed to plot error diagnostics dashboard for {model_name}: {e}")
