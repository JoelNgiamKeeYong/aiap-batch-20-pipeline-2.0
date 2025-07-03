# src/train_models.py

import joblib
import os
import time
from scipy.stats import uniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.pipeline import Pipeline 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN

def train_models(
    models,
    X_train, y_train,
    use_randomized_cv,
    n_jobs, use_smote_enn, cv_folds, scoring_metric, random_state
):
    """
    Trains multiple machine learning models with optional SMOTE-ENN and hyperparameter tuning.

    Parameters:
        models (dict): Dictionary of models with hyperparameters.
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        use_randomized_cv (bool): Whether to use RandomizedSearchCV.
        n_jobs (int): Number of jobs for parallel processing.
        use_smote_enn (bool): Whether to apply SMOTE-ENN (True) or SMOTE only (False).
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric for tuning.
        random_state (int): Seed for reproducibility.

    Returns:
        list: Each item is (model_name, best_model, training_time, model_size_kb).

    Raises:
        RuntimeError: 
            If an error occurs during model training, a RuntimeError is raised with details about the failure.
    """
    try:
        print("\n🤖 Training the candidate models...")

        # Ensure the models and output directory exists
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        # Loop through each model and perform training and evaluation
        for model_name, model_info in models.items():
            print(f"\n   ⛏️  Training {model_name} model...")
            start_time = time.time()

            # Choose between SMOTE or SMOTE-ENN
            print("      └── Applying sampling technique to address class imbalance...")
            if use_smote_enn:
                print("      └── Using SMOTE-ENN (hybrid oversampling + cleaning)...")
                sampler = SMOTEENN(
                    smote=SMOTE(sampling_strategy='auto',k_neighbors=5,random_state=random_state),
                    enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=5,kind_sel='mode')
                )
            else:
                print("      └── Using SMOTE (oversampling only)...")
                sampler = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=random_state)

            # Create a pipeline with the sampler and the model
            pipeline = Pipeline([
                ("sampler", sampler),  # Apply sampler only to the training data in each fold
                ("model", model_info["model"])              
            ])

            # Perform hyperparameter tuning using GridSearchCV
            print(f"      └── Performing hyperparameter tuning...")
            if use_randomized_cv == True:
                print(f"      └── Utilising randomized search cross-validation...")
                # Use params for RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=pipeline,  
                    param_distributions=parse_hyperparameters(model_info["params_rscv"]),
                    n_iter=50,
                    scoring=scoring_metric,
                    cv=cv_folds,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    error_score='raise'
                )
            else:
                print(f"      └── Utilising grid search cross-validation...")
                # Use params for GridSearchCV
                search = GridSearchCV(
                    estimator=pipeline,  
                    param_grid=parse_hyperparameters(model_info["params_gscv"]),
                    scoring=scoring_metric,
                    cv=cv_folds,
                    n_jobs=n_jobs,
                )

            # Fit training data
            search.fit(X_train, y_train)

            # Measure training time
            end_time = time.time()
            training_time = end_time - start_time
            print(f"      └── Model trained successfully in {training_time:.2f} seconds.")

            # Extract the best model and parameters
            best_model = search.best_estimator_
            best_params = {
                (k[len("model__"):] if k.startswith("model__") else k): (float(round(v, 2)) if isinstance(v, float) else v)
                for k, v in search.best_params_.items()
            }
            print(f"      └── Best parameters: {best_params}")

            # Save the trained model permanently
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            print(f"      └── Saving trained model to {model_path}...")
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)  # Size in KB
            print(f"      └── Model size: {model_size_kb} KB")
            
            # Store the trained model details in a list for later use
            trained_models.append([model_name, best_model, training_time, model_size_kb])

        return trained_models

    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")
        raise RuntimeError("Model training process failed.") from e
    
#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
def parse_hyperparameters(params, prefix="model__"):
    """
    Parse hyperparameters from YAML configuration into a format suitable for RandomizedSearchCV.
    Automatically prepends prefix (e.g., 'model__') to parameter names if not already present.

    Args:
        params (dict): Hyperparameters dictionary without prefixes.
        prefix (str): Prefix to add for pipeline compatibility.

    Returns:
        dict: Parsed hyperparameters with prefix added as needed.
    """
    parsed_params = {}
    for param_name, param_config in params.items():
        # Add prefix if not present
        if not param_name.startswith(prefix):
            param_key = prefix + param_name
        else:
            param_key = param_name
        
        if isinstance(param_config, list):  # Categorical parameters
            parsed_params[param_key] = param_config
        
        elif isinstance(param_config, dict):  # Continuous or discrete parameters
            param_type = param_config.get("type")
            if param_type == "uniform":
                low = param_config["low"]
                high = param_config["high"]
                parsed_params[param_key] = uniform(loc=low, scale=high - low)
            elif param_type == "randint":
                low = param_config["low"]
                high = param_config["high"]
                parsed_params[param_key] = randint(low, high)
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
        
        else:
            raise ValueError(f"Invalid parameter configuration for '{param_name}': {param_config}")
    
    return parsed_params