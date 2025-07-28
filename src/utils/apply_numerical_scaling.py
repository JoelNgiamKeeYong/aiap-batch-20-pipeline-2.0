# sec/utils/apply_numerical_scaling.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector

def apply_numerical_scaling(
    list_of_dfs,
    default_scaler=None,
    custom_scaler_dict=None,
):
    """
    Scale numerical features in training and testing DataFrames.
    Scalers are fit on the training DataFrame and applied to both training and testing sets.
    Allows overriding the default scaler for specific columns using a dictionary.

    Args:
        list_of_dfs (list of pd.DataFrame): List of two DataFrames [train_df, test_df].
        default_scaler (sklearn transformer, optional): Default scaler for numeric columns (default: StandardScaler()).
        custom_scaler_dict (dict, optional): Dict mapping column names to custom scalers to override default.

    Returns:
        list of pd.DataFrame: Scaled training and testing DataFrames with transformed column names.
    """
    print("      └── Scaling numerical features...")

    # Check if the input list of DataFrames is empty
    if not list_of_dfs or len(list_of_dfs) != 2:
        raise ValueError("      └── ❌ The input list of DataFrames must contain at least two DataFrames (train and test).")
    
    # If no default scaler is provided, use StandardScaler
    if default_scaler is None:
        default_scaler = StandardScaler()
    
    # If no custom scaler dictionary is provided, initialize it as an empty dictionary
    if custom_scaler_dict is None:
        custom_scaler_dict = {}
    
    # Extract training and testing DataFrames
    df_train, df_test = list_of_dfs[0].copy(), list_of_dfs[1].copy()

    # Identify numeric columns
    num_cols = make_column_selector(dtype_include=['number'])(df_train)

    # Separate columns with custom scalers and default scaler
    custom_cols = [col for col in num_cols if col in custom_scaler_dict]
    default_cols = [col for col in num_cols if col not in custom_scaler_dict]

    transformers = []

    # Pipelines for custom scaler columns
    for col in custom_cols:
        scaler = custom_scaler_dict[col]
        pipe = Pipeline([('scaler', scaler)])
        transformers.append((f'{col.lower().replace(" ", "_")}_pipe', pipe, [col]))

    # Pipeline for default scaler columns
    if default_cols:
        default_pipe = Pipeline([('scaler', default_scaler)])
        transformers.append(('default_num_pipe', default_pipe, default_cols))

    # Create ColumnTransformer
    col_transformer = ColumnTransformer(transformers)

    # Fit on train and transform both train and test
    df_train_scaled_array = col_transformer.fit_transform(df_train)
    df_test_scaled_array = col_transformer.transform(df_test)

    # Get transformed column names
    feature_names = col_transformer.get_feature_names_out()

    # Return scaled DataFrames with original index and transformed columns
    df_train_scaled = pd.DataFrame(df_train_scaled_array, columns=feature_names, index=df_train.index)
    df_test_scaled = pd.DataFrame(df_test_scaled_array, columns=feature_names, index=df_test.index)

    return [df_train_scaled, df_test_scaled]