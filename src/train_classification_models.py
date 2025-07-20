# src/train_classification_models.py

import os
import time
import joblib
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

from utils.create_search_cv import create_search_cv

def train_classification_models(
    task_type,
    models,
    X_train, y_train,
    use_randomized_cv=True,
    use_smote_enn=True,
    cv_folds=5,
    scoring_metric='f1',
    n_jobs=-1,
    random_state=42
):
    """
    Trains classification models with SMOTE-ENN and hyperparameter tuning.

    Parameters:
        task_type (str): Type of task ('classification' or 'regression').
        models (dict): Model name + dict with {'model', 'params_gscv', 'params_rscv'}.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        use_randomized_cv (bool): Use RandomizedSearchCV if True; else GridSearchCV.
        use_smote_enn (bool): Whether to use SMOTEENN or SMOTE only.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric to optimize (e.g., 'f1').
        n_jobs (int): Number of parallel jobs.
        random_state (int): Random seed for reproducibility.

    Returns:
        list: Each item is (model_name, best_model, training_time, model_size_kb).
    """
    try:
        print("\n🤖 Training classification models...")

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
            sampler = choose_sampler(use_smote_enn=use_smote_enn, random_state=random_state)
            steps = [("sampler", sampler), ("model", model)]
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

            # Save the model
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)
            print(f"      └── Saved model to {model_path} ({model_size_kb} KB)")

            # Append training details
            trained_models.append([model_name, best_model, training_time, model_size_kb])

        return trained_models

    except Exception as e:
        print(f"❌ Error: {e}")
        raise RuntimeError("Classification model training failed.") from e


#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
# 📀 CHOOSE SAMPLER
def choose_sampler(use_smote_enn, k_neighbors=5, random_state=42):
    """
    Choose and configure a sampling strategy to address class imbalance in classification tasks.

    This function returns a sampler object based on the specified technique:
    - If `use_smote_enn` is True, it returns a SMOTE-ENN sampler, which combines oversampling 
      (SMOTE) with instance cleaning (Edited Nearest Neighbours).
    - If False, it returns a basic SMOTE sampler that performs only oversampling.

    Args:
        use_smote_enn (bool): If True, use SMOTE-ENN; if False, use SMOTE only.
        k_neighbors (int): Number of nearest neighbors for SMOTE sampling.
        random_state (int): Seed for reproducibility.

    Returns:
        imblearn.BaseSampler: A configured sampling object (SMOTE or SMOTEENN) that can be 
        applied to training data using `fit_resample(X, y)`.
    """
    print("      └── Applying resampling technique to address class imbalance...")

    # Choose the appropriate sampler based on the configuration
    if use_smote_enn:
        # SMOTE-ENN combines oversampling with cleaning
        print("      └── Using SMOTE-ENN (hybrid oversampling + cleaning)...")
        return SMOTEENN(
            smote=SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=random_state),
            enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=k_neighbors, kind_sel='mode')
        )
    else:
        # Use SMOTE for oversampling only
        print("      └── Using SMOTE (oversampling only)...")
        return SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=random_state)
