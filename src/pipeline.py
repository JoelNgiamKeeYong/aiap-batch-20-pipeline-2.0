# src/pipeline.py
# This script orchestrates the entire pipeline for the project.

import os
import time
import yaml
import argparse
import matplotlib
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importing custom modules
from utils import compare_dataframes
from load_data import load_data 
from clean_data import clean_data
from preprocess_data import preprocess_data
from train_models import train_models
from evaluate_models import evaluate_models

def main():
    # Start timer
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ML prediction pipeline")
    parser.add_argument("--lite", action="store_true", help="Run the pipeline in lite mode: uses a simpler model.")
    parser.add_argument("--model", nargs="+", choices=["lr", "rf", "xgb", "lgbm"], 
        help="Specify which model(s) to run (lr, rf, xgb, lgbm). If no models are specified, all models will be run.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration values
    LOKY_MAX_CPU_COUNT = config["LOKY_MAX_CPU_COUNT"]
    DB_PATH = config["db_path"]
    DB_TABLE_NAME = config["db_table_name"]
    TARGET = config["target"]
    TEST_SIZE = config["test_size"]
    RANDOM_STATE = config["random_state"]
    N_JOBS = config["n_jobs"]
    GENERATE_FEATURE_IMPORTANCE = config['generate_feature_importance']
    GENERATE_CONFUSION_MATRIX = config['generate_confusion_matrix']
    GENERATE_LEARNING_CURVE = config['generate_learning_curve']
    GENERATE_CALIBRATION_CURVE = config['generate_calibration_curve']
    USE_RANDOMIZED_CV = config["use_randomized_cv"]
    USE_SMOTE_ENN = config["use_smote_enn"]
    RUN_CON_CLEAN_DATA = config["run_on_clean_data"]
    CV_FOLDS = config["cv_folds"]
    SCORING_METRIC = config["scoring_metric"]
    MINIMUM_ACCEPTABLE_PRECISION = config["minimum_acceptable_precision"]
    LR_MODEL = config["model_configuration"]["Logistic Regression"]
    RF_MODEL = config["model_configuration"]["Random Forest"]
    XG_MODEL = config["model_configuration"]["XGBoost"]
    LGBM_MODEL = config["model_configuration"]["LightGBM"]

    # Set environment variables for parallel processing and matplotlib backend
    os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT 
    matplotlib.use('Agg')  # Non-interactive backend for plotting

    # Step 1: Load the dataset
    df = load_data(db_path=DB_PATH, db_table_name=DB_TABLE_NAME)

    # Step 2: Clean the dataset
    df_cleaned = clean_data(df)
    compare_dataframes(df_original=df, df_new=df_cleaned, original_name_string="raw", new_name_string="cleaned")

    # Step 3: Preprocess the data
    X_train, X_test, y_train, y_test, df_preprocessed = preprocess_data(
        df_cleaned=df_cleaned,
        target=TARGET, test_size=TEST_SIZE, run_on_clean_data = RUN_CON_CLEAN_DATA, random_state=RANDOM_STATE
    )
    compare_dataframes(df_original=df_cleaned, df_new=df_preprocessed, original_name_string="cleaned", new_name_string="preprocessed", show_verbose=False)

    # Step 4: Define the models and their respective hyperparameter grids
    all_models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE),
            "params_gscv": LR_MODEL['params_gscv'],
            "params_rscv": LR_MODEL['params_rscv']
        },
        "Random Forest": {
            "model": RandomForestClassifier( random_state=RANDOM_STATE),
            "params_gscv": RF_MODEL['params_gscv'],
            "params_rscv": RF_MODEL['params_rscv']
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=RANDOM_STATE),
            "params_gscv": XG_MODEL['params_gscv'],
            "params_rscv": XG_MODEL['params_rscv']
        },
        "LightGBM": {  
            "model": LGBMClassifier(verbose=-1,  force_row_wise=True, random_state=RANDOM_STATE),
            "params_gscv": LGBM_MODEL['params_gscv'],
            "params_rscv": LGBM_MODEL['params_rscv']
        }
    }

    # Get the models to train based on arguments
    models = get_models_to_train(args, all_models)

    # Step 5: Train the models
    trained_models = train_models(
        models=models, 
        X_train=X_train, y_train=y_train,
        use_randomized_cv=USE_RANDOMIZED_CV,
        n_jobs=N_JOBS, use_smote_enn=USE_SMOTE_ENN, cv_folds=CV_FOLDS, scoring_metric=SCORING_METRIC, random_state=RANDOM_STATE
    )

    # Step 6: Evaluate the models
    trained_models = evaluate_models(
        trained_models=trained_models,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        scoring=SCORING_METRIC, minimum_acceptable_precision=MINIMUM_ACCEPTABLE_PRECISION,
        generate_feature_importance=GENERATE_FEATURE_IMPORTANCE,
        generate_confusion_matrix=GENERATE_CONFUSION_MATRIX,
        generate_learning_curve=GENERATE_LEARNING_CURVE,
        generate_calibration_curve=GENERATE_CALIBRATION_CURVE   
    )

    # Step 7: Log training
    log_training(trained_models)

    # Print a summary table of training times and model sizes
    print("\nPipeline Summary Table:")
    table_data = [
        [model_name, f"{model_size_kb:.2f}", f"{training_time:.2f}", f"{evaluation_time:.2f}"]
        for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models
    ]
    headers = ["Model Name", "Model Size (KB)", "Training Time (s)", "Evaluation Time (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Record time taken for pipeline execution
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")


def get_models_to_train(args, all_models):
    """
    Determines which models to train based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments.
        all_models (dict): Dictionary of all available models and their configurations.
    
    Returns:
        dict: A dictionary of selected models and their configurations.
    """
    # Mapping between shorthand names and full model names
    model_mapping = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgbm": "LightGBM"
    }

    # Filter models based on arguments
    if args.lite:
        # Run only LightGBM in Lite Mode
        return {"LightGBM": all_models["LightGBM"]}

    # Select models based on --model arguments, default to all models if none are specified
    selected_models = args.model or ["lr", "rf", "xgb", "lgbm"]  # Default to all models if no --model is provided
    models = {
        model_mapping[shorthand]: all_models[model_mapping[shorthand]]
        for shorthand in selected_models
        if shorthand in model_mapping  # Ensure the shorthand is valid
    }

    # Handle invalid shorthand names
    invalid_models = set(selected_models) - set(model_mapping.keys())
    if invalid_models:
        print(f"⚠️ Warning: Ignoring invalid model(s): {', '.join(invalid_models)}")

    return models


def log_training(trained_models):
    """
    Logs training details into a file in the archives folder, appending new content at the top of the file.
    """
    archives_dir = "archives"
    log_file_path = os.path.join(archives_dir, "training_logs.txt")
    
    # Ensure the archives directory exists
    os.makedirs(archives_dir, exist_ok=True)

    # Prepare the new content to prepend
    new_content = ""
    for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models:
        new_content += "=" * 135 + "\n"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        new_content += "🕛 " + current_time + "\n"
        metrics_table = tabulate(
            pd.DataFrame([formatted_metrics]).to_dict(orient='records'),
            headers="keys",
            tablefmt="grid",
            floatfmt=".2f"
        )
        new_content += metrics_table + "\n"
        new_content += f"└──── Model Size: {model_size_kb:.2f} KB\n"
        new_content += f"└──── Training Time: {training_time:.2f} seconds\n"
        new_content += f"└──── Evaluation Time: {evaluation_time:.2f} seconds\n"
        new_content += f"└──── Best Parameters: {best_model.get_params()}\n\n"

    # Read the existing content of the file (if it exists)
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    # Write the new content followed by the existing content
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(new_content + existing_content)

    print(f"💾 Saved training logs to archives folder!")


if __name__ == "__main__":
    main()