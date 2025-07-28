# src/utils/apply_scaling_and_encoding.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def apply_scaling_and_encoding(
    X_train,
    X_test,
    default_scaler=StandardScaler(),
    default_encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    custom_scaler_dict=None,
    custom_encoder_dict=None,
    impute_missing=False,
):
    """
    Apply imputation, scaling, and encoding to training and testing DataFrames.
    Supports default scalers/encoders with per-column overrides. Optionally imputes missing values.
    Fits on training data, applies to both sets.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        default_scaler (sklearn transformer): Default scaler for numeric features.
        default_encoder (sklearn transformer): Default encoder for categorical features.
        custom_scaler_dict (dict, optional): {col_name: scaler} to override default scaler.
        custom_encoder_dict (dict, optional): {col_name: encoder} to override default encoder.
        impute_missing (bool): If True, imputes missing values before scaling/encoding.

    Returns:
        tuple: Transformed X_train, X_test, and fitted ColumnTransformer.
    """
    print("      └── Applying scaling and encoding...")

    if X_train is None or X_test is None:
        raise ValueError("      └── ❌ X_train and X_test must not be None.")

    X_train, X_test = X_train.copy(), X_test.copy()

    # Validate custom scaler keys
    custom_scaler_dict = custom_scaler_dict or {}
    for col in custom_scaler_dict.keys():
        if col not in X_train.columns:
            raise ValueError(f"Custom scaler column '{col}' not in X_train columns")
    
    # Validate custom encoder keys
    custom_encoder_dict = custom_encoder_dict or {}
    for col in custom_encoder_dict.keys():
        if col not in X_train.columns:
            raise ValueError(f"Custom encoder column '{col}' not in X_train columns")
    
    # Automatically select numeric and categorical columns
    num_cols = make_column_selector(dtype_include=['number'])(X_train)
    cat_cols = make_column_selector(dtype_include=['object', 'category', 'bool'])(X_train)

    transformers = []
    used_num_cols = set()
    used_cat_cols = set()

    # Custom scalers for specific numeric columns
    for col, scaler in custom_scaler_dict.items():
        steps = []
        if impute_missing:
            steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', scaler))
        pipe = Pipeline(steps)
        transformers.append((f'{col}_num_scaler', pipe, [col]))
        used_num_cols.add(col)

    # Default scaler for remaining numeric columns 
    default_num_cols = [col for col in num_cols if col not in used_num_cols]
    if default_num_cols:
        steps = []
        if impute_missing:
            steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('scaler', default_scaler))
        num_pipe = Pipeline(steps)
        transformers.append(('default_num_scaler', num_pipe, default_num_cols))

    # Custom encoders for specific categorical columns
    for col, encoder in custom_encoder_dict.items():
        steps = []
        if impute_missing:
            steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        steps.append(('encoder', encoder))
        pipe = Pipeline(steps)
        transformers.append((f'{col}_cat_encoder', pipe, [col]))
        used_cat_cols.add(col)

    # Default encoder for remaining categorical columns
    default_cat_cols = [col for col in cat_cols if col not in used_cat_cols]
    if default_cat_cols:
        steps = []
        if impute_missing:
            steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        steps.append(('encoder', default_encoder))
        cat_pipe = Pipeline(steps)
        transformers.append(('default_cat_encoder', cat_pipe, default_cat_cols))

    # Combine all transformers into a single column transformer
    col_transformer = ColumnTransformer(transformers, verbose_feature_names_out=True)

    # Fit on training data and transform both train and test sets
    X_train_arr = col_transformer.fit_transform(X_train)
    X_test_arr = col_transformer.transform(X_test)

    # Convert arrays back to DataFrames with proper feature names
    feature_names = col_transformer.get_feature_names_out()
    X_train_transformed = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_transformed = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test.index)

    return X_train_transformed, X_test_transformed, col_transformer
