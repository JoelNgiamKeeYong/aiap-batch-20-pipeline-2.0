# src/evaluate_classification_models.py

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc,
)
from utils import generate_feature_importance, generate_learning_curve, save_evaluation_metrics


def evaluate_classification_models(
        trained_models,
        X_train, X_test, y_train, y_test,
        scoring='f1',
        minimum_acceptable_precision=None,
        minimum_acceptable_recall=None,
        generate_plots=True,
        random_state=42, n_jobs=-1
):
    """
    Evaluate a list of trained models and save evaluation results.

    This function evaluates each trained model by computing various metrics, generating visualizations, and saving the results to the specified output directory. It ensures reproducibility and provides detailed insights into model performance, including general metrics, ROC/PR curves, feature importance, confusion matrices, learning curves, and calibration curves.

    Args:
        trained_models (list): 
            List of tuples containing trained models to evaluate. Each tuple includes:
            - model_name (str): Name of the model.
            - best_model (object): The trained model after hyperparameter tuning.
            - training_time (float): Time taken to train the model (in seconds).
            - model_size_kb (float): Size of the saved model file (in KB).
        X_train (pd.DataFrame or np.ndarray): 
            Training feature matrix.
        X_test (pd.DataFrame or np.ndarray): 
            Testing feature matrix.
        y_train (pd.Series or np.ndarray): 
            Training target variable.
        y_test (pd.Series or np.ndarray): 
            Testing target variable.

    Returns:
        list: 
            Updated list of trained models with evaluation metrics and evaluation time appended to each tuple.
    """
    print(f"\n📊 Evaluating best models...")

    # Ensure the output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model
    roc_data = []  # ROC curve data for all models
    pr_data = []  # Precision-Recall curve data for all models

    # Get feature names
    feature_names = X_train.columns.tolist()  

    # Loop through each trained model and evaluate
    for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
        print(f"\n   📋 Evaluating {model_name} model...")
        start_time = time.time()

        # Compute metrics
        formatted_metrics, roc_data_dict, pr_data_dict = compute_classification_metrics(
            model_name=model_name,
            model=best_model,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            minimum_precision=minimum_acceptable_precision,
            minimum_recall=minimum_acceptable_recall  
        )
        results.append(formatted_metrics)
        roc_data.append(roc_data_dict)
        pr_data.append(pr_data_dict)

        # Process feature importance
        if generate_plots:
            generate_feature_importance(model_name, best_model, X_train, y_train, feature_names, scoring, output_dir) 
            plot_confusion_matrix(
                model_name=model_name, model=best_model,
                X_test=X_test, y_test=y_test,
                negative_class_label="Not Subscribed", positive_class_label="Subscribed",
                output_dir=output_dir
            )
            generate_learning_curve(
                model_name=model_name, model=best_model,
                X_train=X_train, y_train=y_train,
                scoring=scoring,
                random_state=random_state, n_jobs=n_jobs,
                output_dir=output_dir
            )
            plot_calibration_curves(
                model_name=model_name, model=best_model,
                X_test=X_test, y_test=y_test,
                output_dir=output_dir
            )          

        end_time = time.time()
        evaluation_time = end_time - start_time
        print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds!")

        # Add evaluation_time to the trained_models list
        trained_models[i] = (model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time)

    # Save consolidated results
    save_evaluation_metrics(results, output_dir)
    if generate_plots:
        plot_combined_roc_curves(roc_data, output_dir)
        plot_combined_pr_curves(pr_data, output_dir)
    print(f"\n💾 Saved evaluation metrics and charts to {output_dir} folder!")

    return trained_models

#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
# 📀 COMPUTE CLASSIFICATION METRICS
def compute_classification_metrics(
    model_name, model,
    X_train, X_test, y_train, y_test,
    minimum_precision=None,
    minimum_recall=None,
    default_threshold=0.5
):
    """
    Compute classification metrics (with optional threshold optimization), evaluate ROC/PR AUC, and return formatted metrics for reporting.

    Args:
        model_name (str): Name of the model.
        model (object): Trained classifier with predict_proba method.
        X_train, X_test (array-like): Feature matrices.
        y_train, y_test (array-like): True binary labels.
        minimum_precision (float, optional): Constraint for threshold optimization.
        minimum_recall (float, optional): Constraint for threshold optimization.
        default_threshold (float): Fallback threshold if no constraints.

    Returns:
        dict: Formatted evaluation metrics for reporting.
        float: Optimal threshold used on test data.
    """
    print("      └── Computing classification performance metrics...")

    # Step 1: Predict probabilities
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    # Step 2: Determine optimal threshold (if needed)
    def find_threshold(y_true, y_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        precision = precision[:-1]
        recall = recall[:-1]
        thresholds = thresholds

        # Strategy 1: Only minimum precision is specified
        if minimum_precision is not None and minimum_recall is None:
            valid = precision >= minimum_precision
            if not np.any(valid):
                raise ValueError("\n⚠️  No threshold satisfies the minimum precision constraint. Please adjust the constraints in the config.yaml file")
            best_idx = np.argmax(recall * valid)  # Maximize recall under precision constraint
            return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_score(y_true, y_probs >= thresholds[best_idx])

        # Strategy 2: Only minimum recall is specified
        if minimum_recall is not None and minimum_precision is None:
            valid = recall >= minimum_recall
            if not np.any(valid):
                raise ValueError("\n⚠️  No threshold satisfies the minimum recall constraint. Please adjust the constraints in the config.yaml file.")
            best_idx = np.argmax(precision * valid)  # Maximize precision under recall constraint
            return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_score(y_true, y_probs >= thresholds[best_idx])

        # Strategy 3: Both constraints present
        if minimum_precision is not None and minimum_recall is not None:
            valid = (precision >= minimum_precision) & (recall >= minimum_recall)
            if not np.any(valid):
                raise ValueError("\n⚠️  No threshold satisfies both precision and recall constraints. Please adjust the constraints in the config.yaml file")
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores * valid)
            return thresholds[best_idx], precision[best_idx], recall[best_idx], f1_scores[best_idx]

        # No constraints
        return default_threshold, None, None, None

    threshold, best_prec, best_rec, best_f1 = find_threshold(y_train, y_train_probs)

    if best_prec is not None:
        print(f"      └── Optimal threshold = {threshold:.3f} (Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, F1: {best_f1:.3f})")
    else:
        print(f"      └── No constraint set. Using default threshold = {threshold}")

    # Step 3: Threshold predictions
    y_train_pred = (y_train_probs >= threshold).astype(int)
    y_test_pred = (y_test_probs >= threshold).astype(int)

    # Step 4: Compute metrics
    def compute_scores(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
        }

    train_metrics = compute_scores(y_train, y_train_pred)
    test_metrics = compute_scores(y_test, y_test_pred)

    # Step 5: Probability-based metrics
    roc_auc = pr_auc = None
    try:
        # Compute ROC curves
        print(f"      └── Computing ROC AUC data...")
        fpr, tpr, _ = roc_curve(y_test, y_test_probs)
        roc_auc = auc(fpr, tpr)
        roc_data_dict = {"Model": model_name, "FPR": fpr, "TPR": tpr, "AUC": roc_auc}
        
        # Compute Precision-Recall curves
        print(f"      └── Computing Precision-Recall AUC data...")
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_probs)
        pr_auc = auc(recall_curve, precision_curve)
        pr_data_dict = {"Model": model_name, "Precision": precision_curve, "Recall": recall_curve, "AUC-PR": pr_auc}
        
    except Exception as e:
        print(f"      └── ⚠️ Could not compute ROC/PR AUC for {model_name}: {e}")

    # Step 6: Format results for reporting
    metrics = {
        "Model": model_name,
        "Accuracy": f"{test_metrics['Accuracy']:.3f} ({train_metrics['Accuracy']:.3f})",
        "Precision": f"{test_metrics['Precision']:.3f} ({train_metrics['Precision']:.3f})",
        "Recall": f"{test_metrics['Recall']:.3f} ({train_metrics['Recall']:.3f})",
        "F1-Score": f"{test_metrics['F1-Score']:.3f} ({train_metrics['F1-Score']:.3f})",
        "ROC AUC": f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
        "PR AUC": f"{pr_auc:.3f}" if pr_auc is not None else "N/A",
    }

    return metrics, roc_data_dict, pr_data_dict

#################################################################################################################################
#################################################################################################################################
def plot_confusion_matrix(
    model_name, model,
    X_test, y_test,
    negative_class_label="Negative", positive_class_label="Postive", 
    output_dir='output'
):
    """Generate and save confusion matrix visualization."""
    
    try:
        print(f"      └── Generating and plotting confusion matrix heatmap...")

        # Generate normalized confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

        # Ensure output directory exists and define file path
        os.makedirs(f"{output_dir}/confusion_matrix", exist_ok=True)
        file_path = f"{output_dir}/confusion_matrix/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"

        # Plot and save heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_normalized, annot=True, fmt=".3f", cmap="Blues",
            xticklabels=[negative_class_label, positive_class_label],
            yticklabels=[negative_class_label, positive_class_label]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Normalized Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    except Exception as e:
        print(f"      └── ⚠️ Failed to plot confusion matrix for {model_name}: {e}")

#################################################################################################################################
#################################################################################################################################
def plot_calibration_curves(
    model_name, model,
    X_test, y_test,
    output_dir='output'
):
    """Generate and save calibration curves for the model."""

    try:
        print(f"      └── Generating and plotting calibration curve...")

        # Predict probabilities for the positive class
        prob_pos = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred_val = calibration_curve(y_test, prob_pos, n_bins=10)

        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred_val, frac_pos, "s-", label=f"{model_name}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve - {model_name}")
        plt.legend(loc="upper left")
        plt.grid(True)

        # Save the calibration curve plot
        os.makedirs(f"{output_dir}/calibration_curves", exist_ok=True)
        file_path = f"{output_dir}/calibration_curves/calibration_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    except AttributeError:
        print(f"      └── ⚠️ Model {model_name} does not support `predict_proba`. Skipping calibration curve generation.")
    except Exception as e:
        print(f"      └── ⚠️ Failed to plot calibration curve for {model_name}: {e}")

#################################################################################################################################
#################################################################################################################################
def plot_combined_roc_curves(roc_data, output_dir):
    if not roc_data:
        print("      └── ❌ No ROC data available.")
        return

    plt.figure(figsize=(10, 8))
    for roc in roc_data:
        plt.plot(roc["FPR"], roc["TPR"], label=f"{roc['Model']} (AUC = {roc['AUC']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/roc_curves_combined.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

#################################################################################################################################
#################################################################################################################################
def plot_combined_pr_curves(pr_data, output_dir):
    if not pr_data:
        print("      └── ❌ No PR data available.")
        return

    plt.figure(figsize=(10, 8))
    for pr in pr_data:
        plt.plot(pr["Recall"], pr["Precision"], label=f"{pr['Model']} (AUC-PR = {pr['AUC-PR']:.3f})", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/pr_curves_combined.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()