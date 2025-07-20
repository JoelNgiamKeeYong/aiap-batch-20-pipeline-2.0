# src/train_classification_models.py

import os
import time
import joblib
from scipy.stats import uniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

def train_classification_models(
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

        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        for model_name, model_info in models.items():
            print(f"\n   ⛏️  Training {model_name} model...")
            start_time = time.time()

            model = model_info["model"]

            # Sampler
            sampler = choose_sampler(use_smote_enn=use_smote_enn, random_state=random_state)
            steps = [("sampler", sampler), ("model", model)]
            pipeline = Pipeline(steps)

            # Hyperparameter tuning
            param_config = model_info["params_rscv"] if use_randomized_cv else model_info["params_gscv"]
            search = build_search_cv(
                model_pipeline=pipeline,
                param_config=param_config,
                use_randomized=use_randomized_cv,
                cv=cv_folds,
                scoring=scoring_metric,
                n_iter=50,
                n_jobs=n_jobs,
                random_state=random_state
            )

            search.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time
            print(f"      └── Model trained in {training_time:.2f} seconds.")

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
        raise RuntimeError("Classification model training failed.") from e

#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
# 🔶 CHOOSE SAMPLER
def choose_sampler(use_smote_enn, k_neighbors=5, random_state=42):
    print("      └── Applying sampling technique to address class imbalance...")

    if use_smote_enn:
        print("      └── Using SMOTE-ENN (hybrid oversampling + cleaning)...")
        return SMOTEENN(
            smote=SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=random_state),
            enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=5, kind_sel='mode')
        )
    else:
        print("      └── Using SMOTE (oversampling only)...")
        return SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=random_state)

#################################################################################################################################
#################################################################################################################################
# 🔶 BUILD SEARCH CV
def build_search_cv(
        model_pipeline, param_config,
        use_randomized=True, prefix="model__",
        cv=5, scoring='f1',
        n_iter=50, n_jobs=-1, random_state=42
):
    print(f"      └── Performing hyperparameter tuning...")

    parsed_params = {}
    for param, config in param_config.items():
        param_key = param if param.startswith(prefix) else f"{prefix}{param}"

        if isinstance(config, list):
            parsed_params[param_key] = config
        elif isinstance(config, dict):
            dist_type = config.get("type")
            if dist_type == "uniform":
                parsed_params[param_key] = uniform(loc=config["low"], scale=config["high"] - config["low"])
            elif dist_type == "randint":
                parsed_params[param_key] = randint(config["low"], config["high"])
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
        else:
            raise ValueError(f"Invalid config for param '{param}': {config}")

    if use_randomized:
        print("      └── Using RandomizedSearchCV...")
        return RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=parsed_params,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            error_score='raise'
        )
    else:
        print("      └── Using GridSearchCV...")
        return GridSearchCV(
            estimator=model_pipeline,
            param_grid=parsed_params,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )
