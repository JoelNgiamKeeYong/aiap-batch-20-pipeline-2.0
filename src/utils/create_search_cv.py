# src/utils/create_search_cv.py

from scipy.stats import uniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def create_search_cv(
    task_type,
    model_pipeline,
    param_config, prefix="model__",
    use_randomized=True, 
    cv=5, scoring=None,
    n_iter=50, n_jobs=-1, random_state=42
):
    """
    Creates a GridSearchCV or RandomizedSearchCV object with hyperparameter search space.

    Args:
        task_type (str): Task type - one of 'binary', 'multiclass', or 'regression'.
        model_pipeline: A sklearn pipeline or estimator.
        param_config (dict): Dictionary of parameters and their ranges or distributions.
        prefix (str): Prefix for parameter names in the pipeline.
        use_randomized (bool): If True, uses RandomizedSearchCV; else GridSearchCV.
        cv (int or cross-validation generator): Number of folds or CV strategy.
        scoring (str or None): Scoring metric. If None, defaults will be applied based on task_type.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        n_jobs (int): Number of jobs to run in parallel.
        random_state (int): Random seed.

    Returns:
        GridSearchCV or RandomizedSearchCV object.
    """
    print(f"      └── Performing hyperparameter tuning...")

    # Assign default scoring based on task type
    if scoring is None:
        if task_type == "classification":
            scoring = "f1"
        elif task_type == "regression":
            scoring = "neg_root_mean_squared_error"
        else:
            raise ValueError(f"Unknown task_type '{task_type}'. Choose from 'classification' or 'regression'.")
    else:
        print(f"      └── Using custom scoring: '{scoring}'")

    # Parse the parameter configuration
    parsed_params = {}
    for param, config in param_config.items():
        # Ensure parameter names start with the prefix
        param_key = param if param.startswith(prefix) else f"{prefix}{param}"

        # Handle different types of parameter configurations
        if isinstance(config, list):
            # If it's a list, use it directly
            parsed_params[param_key] = config
        elif isinstance(config, dict):
            # If it's a dict, check for distribution types
            dist_type = config.get("type")
            if dist_type == "uniform":
                # Uniform distribution
                parsed_params[param_key] = uniform(loc=config["low"], scale=config["high"] - config["low"])
            elif dist_type == "randint":
                # Random integer distribution
                parsed_params[param_key] = randint(config["low"], config["high"])
            else:
                raise ValueError(f"      └── ⚠️  Unsupported distribution type: {dist_type}")
        else:
            raise ValueError(f"      └── ⚠️  Invalid config for param '{param}': {config}")

    # Create the search CV object
    if use_randomized:
        print("      └── Using RandomizedSearchCV for exploratory tuning...")
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
        print("      └── Using GridSearchCV for finer tuning...")
        return GridSearchCV(
            estimator=model_pipeline,
            param_grid=parsed_params,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )