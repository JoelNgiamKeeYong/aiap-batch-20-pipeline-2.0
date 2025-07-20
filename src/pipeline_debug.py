# src/pipeline_debug.py

import time
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

from utils import log_training_summary
from train_classification_models import train_classification_models
from train_regression_models import train_regression_models
from evaluate_classification_models import evaluate_classification_models
from evaluate_regression_models import evaluate_regression_models


def run_debug_pipeline(config):
    start_time = time.time()

    # Extract configuration parameters
    DEBUG_TASK_TYPE = config["debug_task_type"]
    TEST_SIZE = config["test_size"]
    RANDOM_STATE = config["random_state"]
    N_JOBS = config["n_jobs"]
    USE_RANDOMIZED_CV = config["use_randomized_cv"]
    USE_SMOTE_ENN = config["use_smote_enn"]
    GENERATE_PLOTS = config["generate_plots"]
    CV_FOLDS = config["cv_folds"]

    print("🧪 Debugging ML pipeline...")

    #################################################################################################################################
    #################################################################################################################################
    # ✅ CLASSIFICATION TASK
    if DEBUG_TASK_TYPE == "classification":
        print("   └── Using sklearn's Breast Cancer dataset for classification task...")

        # Load the dataset
        data = load_breast_cancer(as_frame=True)
        df_debug = data.frame
        target = data.target.name

        # Define models and hyperparameters
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(random_state=42),
                "params_gscv": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"]
                },
                "params_rscv": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "solver": ["liblinear", "lbfgs"]
                }
            }
        }

        # Split the data into training and testing sets
        X = df_debug.drop(columns=[target])
        y = df_debug[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Scale the features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)  

        # Train classification models
        trained_models = train_classification_models(
            models=models,
            X_train=X_train, y_train=y_train,
            use_randomized_cv=USE_RANDOMIZED_CV,
            use_smote_enn=USE_SMOTE_ENN,
            cv_folds=CV_FOLDS,
            scoring_metric="f1",
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

        # Evaluate the trained models
        trained_models = evaluate_classification_models(
            trained_models=trained_models,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            scoring="f1",
            generate_plots=GENERATE_PLOTS,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

    #################################################################################################################################
    #################################################################################################################################
    # ✅ REGRESSIOn TASK
    elif DEBUG_TASK_TYPE == "regression":
        print("   └── Using sklearn's Diabetes dataset for regression task...")

        # Load the dataset
        data = load_diabetes(as_frame=True)
        df_debug = data.frame
        target = data.target.name

        # Define models and hyperparameters
        models = {
            "Linear Regression": {
                "model": LinearRegression(),
                "params_gscv": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },
                "params_rscv": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                }
            }
        }

        # Split the data into training and testing sets
        X = df_debug.drop(columns=[target])
        y = df_debug[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        # Scale the features
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)  

        # Train regression models
        trained_models = train_regression_models(
            models=models,
            X_train=X_train, y_train=y_train,
            use_randomized_cv=USE_RANDOMIZED_CV,
            scoring_metric="neg_root_mean_squared_error",
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

        # Evaluate the trained models
        trained_models = evaluate_regression_models(
            trained_models=trained_models,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            scoring="neg_root_mean_squared_error",
            generate_plots=GENERATE_PLOTS,
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )

    else:
        raise ValueError("❌ Invalid debug_task_type. Use 'classification' or 'regression' in config.yaml.")

    # Log training summary
    log_training_summary(trained_models=trained_models, start_time=start_time)

    #################################################################################################################################
    #################################################################################################################################