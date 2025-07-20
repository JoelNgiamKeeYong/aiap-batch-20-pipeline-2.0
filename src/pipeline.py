# src/pipeline.py
# This script orchestrates the entire pipeline for the project.

import os
import time
import yaml
import argparse
import matplotlib
import numpy as np

from sklearn.utils.multiclass import type_of_target
from sklearn.base import is_classifier, is_regressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from utils import compare_dataframes, log_training_summary
from pipeline_debug import run_debug_pipeline
from load_data import load_data 
from clean_data import clean_data
from preprocess_data import preprocess_data
from train_classification_models import train_classification_models
from train_regression_models import train_regression_models
from evaluate_classification_models import evaluate_classification_models
from evaluate_regression_models import evaluate_regression_models


def main():
    # Start timer
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ML prediction pipeline")
    
    parser.add_argument("--debug", nargs="+", choices=["classification", "regression"], help="Run the pipeline in debug mode for fast iteration.")  
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
    GENERATE_PLOTS = config['generate_plots'] 
    USE_RANDOMIZED_CV = config["use_randomized_cv"]
    USE_SMOTE_ENN = config["use_smote_enn"]
    RUN_ON_CLEAN_DATA = config["run_on_clean_data"]
    CV_FOLDS = config["cv_folds"]
    SCORING_METRIC = config["scoring_metric"]
    MINIMUM_ACCEPTABLE_PRECISION = config["minimum_acceptable_precision"]
    MINIMUM_ACCEPTABLE_RECALL = config["minimum_acceptable_recall"]
    LR_MODEL = config["model_configuration"]["Logistic Regression"]
    RF_MODEL = config["model_configuration"]["Random Forest"]
    XG_MODEL = config["model_configuration"]["XGBoost"]
    LGBM_MODEL = config["model_configuration"]["LightGBM"]

    # Set environment variables
    os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT  # Set maximum CPU count for loky
    matplotlib.use('Agg')  # Non-interactive backend for plotting

    #################################################################################################################################
    #################################################################################################################################
    # ☑️ DEBUG MODE
    # Debug short-circuit pipeline for quick testing
    if args.debug:
        run_debug_pipeline(config=config, task_type=args.debug[0])
        return  # Exit after running debug pipeline
    
    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 1: LOAD DATA
    # - Load the dataset into a pandas DataFrame.
    df = load_data(db_path=DB_PATH, db_table_name=DB_TABLE_NAME)

    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 2: CLEAN DATA
    # - Clean the dataset for further exploration and preprocessing,
    # - To avoid data leakage, do not make too many assumptions about the data.
    df_cleaned = clean_data(df=df)
    compare_dataframes(df_original=df, df_new=df_cleaned, original_name_string="raw", new_name_string="cleaned")

    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 3: PREPROCESS DATA
    # - Preprocess the cleaned dataset to prepare it for model training.
    # - Most of the feature engineering is performed here.
    X_train, X_test, y_train, y_test, df_preprocessed = preprocess_data(
        df_cleaned=df_cleaned,
        target=TARGET,
        test_size=TEST_SIZE,
        run_on_clean_data = RUN_ON_CLEAN_DATA,
        random_state=RANDOM_STATE
    )
    compare_dataframes(df_original=df_cleaned, df_new=df_preprocessed, original_name_string="cleaned", new_name_string="preprocessed", show_verbose=False)

    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 4: DEFINE CANDIDATE MODELS
    # - Define the models to be trained and their respective hyperparameter grids.
    # - Validate that the models are appropriate for the task type (classification or regression).
    all_models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=RANDOM_STATE),
            "params_gscv": LR_MODEL['params_gscv'],
            "params_rscv": LR_MODEL['params_rscv']
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
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
    models = select_models_to_train(args=args, all_models=all_models)  # Get the models to train based on arguments
    task_type = detect_task_type_and_validate_models(y=y_train, models=models)  # Determine ML task type based on the target variable

    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 5: TRAIN MODELS
    # - Train the models using the training data.
    if task_type == "regression":
        trained_models = train_regression_models(
            task_type=task_type,
            models=models, 
            X_train=X_train, y_train=y_train,
            use_randomized_cv=USE_RANDOMIZED_CV,
            scoring_metric=SCORING_METRIC,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
    elif task_type == "classification":
        trained_models = train_classification_models(
            task_type=task_type,
            models=models, 
            X_train=X_train, y_train=y_train,
            use_randomized_cv=USE_RANDOMIZED_CV,
            use_smote_enn=USE_SMOTE_ENN,
            cv_folds=CV_FOLDS, scoring_metric=SCORING_METRIC,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 6: EVALUATE MODELS
    # - Evaluate the models using the test data.
    if task_type == "regression":
        trained_models = evaluate_regression_models(
            task_type=task_type,
            trained_models=trained_models,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            scoring=SCORING_METRIC,
            generate_plots=GENERATE_PLOTS,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
    elif task_type == "classification":
        trained_models = evaluate_classification_models(
            task_type=task_type,
            trained_models=trained_models,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            scoring=SCORING_METRIC,
            minimum_acceptable_precision=MINIMUM_ACCEPTABLE_PRECISION,
            minimum_acceptable_recall=MINIMUM_ACCEPTABLE_RECALL,  
            generate_plots=GENERATE_PLOTS,
            n_jobs=N_JOBS, random_state=RANDOM_STATE,
        )   
    
    #################################################################################################################################
    #################################################################################################################################
    # ✅ STEP 7: LOG TRAINING SUMMARY
    log_training_summary(trained_models=trained_models, start_time=start_time)


#####################################################################################################################################
#####################################################################################################################################
# HELPER FUNCTIONS

#####################################################################################################################################
#####################################################################################################################################
# 📀 SELECT MODELS TO TRAIN
def select_models_to_train(args, all_models):
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
        return {"LightGBM": all_models["LightGBM"]}  # Run only LightGBM in Lite Mode

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
        print(f"⚠️  Warning: Ignoring invalid model(s): {', '.join(invalid_models)}")

    return models

#####################################################################################################################################
#####################################################################################################################################
# 📀 DETECT TASK TYPE AND VALIDATE MODELS
def detect_task_type_and_validate_models(y, models):
    """
    Detect task type from y and validate that all models are appropriate.

    Args:
        y (array-like): Target variable.
        models (dict): Dict of models to validate.

    Returns:
        str: Detected task type ('classification' or 'regression').

    Raises:
        ValueError: If model types don't match the task.
    """
    y = np.asarray(y)
    target_type = type_of_target(y)

    # Determine task type based on target
    if target_type in ["binary", "multiclass", "multilabel-indicator", "multiclass-multioutput"]:
        task_type = "classification"
    elif target_type in ["continuous", "continuous-multioutput"]:
        task_type = "regression"
    else:
        raise ValueError(f"❌ Unrecognized target type: {target_type}")

    # Validate models
    for name, model_dict in models.items():
        model = model_dict["model"]
        if task_type == "classification" and not is_classifier(model):
            raise ValueError(f"❌ Model '{name}' is not a classifier but task is classification.")
        elif task_type == "regression" and not is_regressor(model):
            raise ValueError(f"❌ Model '{name}' is not a regressor but task is regression.")

    print(f"\n🧠  Validating candidate models...")
    print(f"    └── Detected ML task type: {task_type}")
    print(f"    └── All models valid.")
    return task_type


if __name__ == "__main__":
    main()