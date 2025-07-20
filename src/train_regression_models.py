# src/train_regression_models.py

import os
import time
import joblib
from imblearn.pipeline import Pipeline 

from utils.create_search_cv import create_search_cv

def train_regression_models(
    task_type,
    models,
    X_train, y_train,
    use_randomized_cv=True,
    cv_folds=5,
    scoring_metric='neg_root_mean_squared_error',
    n_jobs=-1,
    random_state=42
):
    """
    Trains regression models with hyperparameter tuning.

    Parameters:
        task_type (str): Type of task ('classification' or 'regression').
        models (dict): Model name + dict with {'model', 'params_gscv', 'params_rscv'}.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        use_randomized_cv (bool): Use RandomizedSearchCV if True; else GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric to optimize (e.g., 'neg_root_mean_squared_error').
        n_jobs (int): Number of parallel jobs.
        random_state (int): Random seed for reproducibility.

    Returns:
        list: Each item is (model_name, best_model, training_time, model_size_kb).
    """
    try:
        print("\n🤖 Training regression models...")

        # Create directories for models and output
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        # Iterate over each model configuration
        for model_name, model_info in models.items():
            print(f"\n   ⛏️  Training {model_name} model...")
            start_time = time.time()

            model = model_info["model"]

            # Build pipeline
            steps = [("model", model)]
            pipeline = Pipeline(steps)

            # Hyperparameter tuning
            param_config = model_info["params_rscv"] if use_randomized_cv else model_info["params_gscv"]
            search = create_search_cv(
                task_type=task_type,
                model_pipeline=pipeline,
                param_config=param_config,
                use_randomized=use_randomized_cv,
                cv=cv_folds,
                scoring=scoring_metric,
                n_iter=50,
                n_jobs=n_jobs,
                random_state=random_state
            )

            # Fit the model
            search.fit(X_train, y_train)

            # Calculate training time and model size
            end_time = time.time()
            training_time = end_time - start_time
            print(f"      └── Model trained in {training_time:.2f} seconds.")

            # Get the best model and its parameters
            best_model = search.best_estimator_
            best_params = {
                (k[len("model__"):] if k.startswith("model__") else k): (float(round(v, 2)) if isinstance(v, float) else v)
                for k, v in search.best_params_.items()
            }
            print(f"      └── Best parameters: {best_params}")

            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)
            print(f"      └── Saved model to {model_path} ({model_size_kb} KB)")

            trained_models.append([model_name, best_model, training_time, model_size_kb])

        return trained_models

    except Exception as e:
        print(f"❌ Error: {e}")
        raise RuntimeError("Regression model training failed.") from e
    