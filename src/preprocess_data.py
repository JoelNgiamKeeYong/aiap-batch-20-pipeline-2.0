# src/preprocess_data.py

import os
import joblib
import time
import fnmatch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from kmodes.kmodes import KModes

def preprocess_data(df_cleaned, target, test_size=0.2, run_on_clean_data=False, random_state=42):
    """
    Preprocesses the cleaned dataset to prepare it for machine learning modeling.

    This function performs a series of preprocessing steps as per the 🛠️ indicators identified during Exploratory Data Analysis (EDA). The steps include splitting the data into training and testing sets, encoding categorical variables, scaling numerical features, and saving the processed splits for reuse. If preprocessed files already exist in the specified output path, the function skips the preprocessing steps and loads the existing splits.

    Parameters:
        df_cleaned (pd.DataFrame): 
            The cleaned dataset to be preprocessed. This should be a pandas DataFrame containing all relevant features and the target variable.
        target (str): 
            The name of the target variable in the dataset.
        run_on_clean_data (boolean, optional): 
            Determine if the function should run on the cleaned data or not. Defaults to False.
        test_size (float, optional): 
            Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): 
            Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: 
            A tuple containing the training and testing splits: (X_train, X_test, y_train, y_test), along with the full preprocessed DataFrame and feature names.

    Raises:
        RuntimeError: 
            If an error occurs during preprocessing, a RuntimeError is raised with details about the failure.

    Example Usage:
        >>> cleaned_data = pd.read_csv("data/cleaned_data.csv")
        >>> X_train, X_test, y_train, y_test, df_preprocessed, feature_names = preprocess_data(cleaned_data, target="target_column")
        >>> print(X_train.head())
    """
    try:
        # Define output paths
        df_preprocessed_path = "./data/preprocessed_data.csv"

        # Carry out preprocessing id preprocessed data files do not exist
        print("\n🔧 Preprocessing the dataset...")
        start_time = time.time()

        # Split the data into training and testing sets
        print("\n   ⚙️  Splitting the data into training and testing sets...")
        df_train, df_test, X_train, X_test, y_train, y_test = split_data(df_cleaned=df_cleaned, target=target, test_size=test_size, random_state=random_state)

        # Carry out additional cleaning
        if not run_on_clean_data:
            print("\n   ⚙️  Carrying out additional dataset cleaning...")
            [df_train, df_test] = impute_outlier_values_age(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = bin_age(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = remove_unknown_occupation(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = remove_unknown_marital_status(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = impute_unknowns_housing_loan(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = impute_personal_loan(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = cap_campaign_calls(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = bin_previous_contact_days(list_of_dfs=[df_train, df_test])

        # Create new features
        if not run_on_clean_data:
            print("\n   ⚙️  Creating new features via feature engineering...")
            [df_train, df_test] = create_education_group(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_financial_stability(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_is_married(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_demographic_clusters(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_has_loan(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_contacted_multiple_times(list_of_dfs=[df_train, df_test])
            [df_train, df_test] = create_is_contact_method_cellular(list_of_dfs=[df_train, df_test])
        
        # Drop irrelevant features
        if not run_on_clean_data:
            print("\n   ⚙️  Dropping irrelevant features...")
            [df_train, df_test] = drop_irrelevant_features(list_of_dfs=[df_train, df_test])

        # Checking data integrity after partial preprocessing
        print("\n   ⚙️  Checking data integrity after preprocessing...")
        print(f"      └── Training set shape: {df_train.shape} ...")
        print(f"      └── Test set shape: {df_test.shape} ...")
        total_rows = df_train.shape[0] + df_test.shape[0]
        test_ratio = (df_test.shape[0] / total_rows) * 100
        print(f"      └── Test set constitutes {test_ratio:.2f}% of the total dataset. Original split: {(test_size*100):.2f}%")

        # Separate features (X) and target (y) after partial preprocessing
        print("\n   ⚙️  Separating features (X) and target (y) after partial preprocessing...")
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target] 
        X_test = df_test.drop(columns=[target])
        y_test = df_test[target] 

        # Transform features by apply feature scaling and encoding for linear models
        print("   ⚙️  Transforming features...")
        [X_train, X_test] = transform_features(list_of_dfs=[X_train, X_test])

        # Removing features based on feature selection findings in EDA
        if not run_on_clean_data:
            print("\n   ⚙️  Removing features based on feature selection findings in EDA...")
            [X_train, X_test] = remove_features_based_on_feature_selection(list_of_dfs=[X_train, X_test])

        # Combine features and target into a single DataFrame
        X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        df_preprocessed = pd.concat([X_combined, y_combined], axis=1)

        # Save the preprocessed data to CSV files
        print("\n💾 Saving preprocessed data to /data folder...")

        # Save consoldiated preprocessed file
        df_preprocessed.to_csv(df_preprocessed_path, index=False)

        # Record time taken
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"\n✅ Data preprocessing completed in {elapsed_time:.2f} seconds!")

        # Return data
        return X_train, X_test, y_train, y_test, df_preprocessed

    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        raise RuntimeError("Data preprocessing process failed.") from e
    

#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
def split_data(df_cleaned, target, test_size=0.2, random_state=42):
    print("      └── Separating the features and the target...")
    y = df_cleaned[target]  # Target variable
    X = df_cleaned.drop(columns=[target])  # Feature matrix

    print("      └── Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,                  # Ensures the same class distribution in train and test sets
        test_size=test_size,         # Proportion of the dataset for the test split
        random_state=random_state    # For reproducibility
    )

    # Combine features and target into single DataFrames for training and testing
    print("      └── Combining features and target into single DataFrames...")
    df_train = pd.concat([X_train, y_train], axis=1)  # Combine X_train and y_train
    df_test = pd.concat([X_test, y_test], axis=1)     # Combine X_test and y_test

    # Print shapes for verification
    print(f"      └── Training set shape: {df_train.shape}, Classes: {y_train.value_counts().to_dict()}")
    print(f"      └── Test set shape: {df_test.shape}, Classes: {y_test.value_counts().to_dict()}")

    return df_train, df_test, X_train, X_test, y_train, y_test

#################################################################################################################################
#################################################################################################################################
def impute_outlier_values_age(list_of_dfs):
    print("      └── Imputing outlier values in `age` feature with grouped-by median...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Filter out rows where age = 150 to calculate the median
    df_train = list_of_dfs[0]
    valid_ages = df_train[df_train['age'] != 150]['age']

    # Calculate the median age from valid records
    median_age = valid_ages.median()

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the imputation logic
    for df in list_of_dfs:
        # Replace `age=150` with the calculated median
        df['age'] = df['age'].replace(150, median_age)

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    # Print the median value for reference
    print(f"      └── Outlier values imputed with median: {median_age:.2f}")

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def bin_age(list_of_dfs):
    print("      └── Creating age category bins for `age` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Define age bins and labels based on the data distribution
    bins = [15, 20, 60, 100]
    labels = [
        "young", "middle-age", "elderly",
    ]

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the binning logic
    for df in list_of_dfs:
        # Create a new column for age categories
        df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)

        # Convert the new column to categorical type for memory efficiency
        df['age_category'] = df['age_category'].astype('category')

        # Reorder columns to place 'age_category' before 'age'
        cols = df.columns.tolist()  
        age_index = cols.index('age') 
        cols.insert(age_index, 'age_category') 
        df = df[cols]

        # Drop the duplicate 'age_category' column
        df = df.loc[:, ~df.columns.duplicated()]

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_unknown_occupation(list_of_dfs):
        print("      └── Removing unknown values in the `occupation` feature...")

        if not list_of_dfs:
            raise ValueError("The input list of DataFrames cannot be empty.")

        # Initialize a list to store the modified DataFrames
        modified_dfs = []
        total_rows_removed = 0
        total_rows_before = 0

        # Iterate over each DataFrame and apply the capping logic
        for df in list_of_dfs:
            total_rows_before += len(df)

            # Remove rows where 'unknown' appears in 'occupation' columns
            df_removed = df[df['occupation'] != 'unknown'].copy()  

            # Count the rows removed
            rows_removed = len(df) - len(df_removed)
            total_rows_removed += rows_removed

            # Convert the column to category type
            df_removed['occupation'] = df_removed['occupation'].astype('category')

            # Append the modified DataFrame
            modified_dfs.append(df_removed)

        # Calculate percentage of dataset remaining
        total_rows_after = total_rows_before - total_rows_removed
        percentage_remaining = (total_rows_after / total_rows_before) * 100

        print(f"      └── Total rows removed: {total_rows_removed} ({(total_rows_removed / total_rows_before) * 100:.4f}% of the dataset)")
        print(f"      └── {percentage_remaining:.4f}% of the dataset remains after removal")

        return modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_unknown_marital_status(list_of_dfs):
        print("      └── Removing unknown values in the `marital_status` feature...")

        if not list_of_dfs:
            raise ValueError("The input list of DataFrames cannot be empty.")

        # Initialize a list to store the modified DataFrames
        modified_dfs = []
        total_rows_removed = 0
        total_rows_before = 0

        # Iterate over each DataFrame and apply the capping logic
        for df in list_of_dfs:
            total_rows_before += len(df)

            # Remove rows where 'unknown' appears in 'marital_status' columns
            df_removed = df[df['marital_status'] != 'unknown'].copy()    

            # Count the rows removed
            rows_removed = len(df) - len(df_removed)
            total_rows_removed += rows_removed

            # Convert the column to category type
            df_removed['occupation'] = df_removed['occupation'].astype('category')

            # Append the modified DataFrame
            modified_dfs.append(df_removed)

        # Calculate percentage of dataset remaining
        total_rows_after = total_rows_before - total_rows_removed
        percentage_remaining = (total_rows_after / total_rows_before) * 100

        print(f"      └── Total rows removed: {total_rows_removed} ({(total_rows_removed / total_rows_before) * 100:.4f}% of the dataset)")
        print(f"      └── {percentage_remaining:.4f}% of the dataset remains after removal")

        return modified_dfs

#################################################################################################################################
#################################################################################################################################
def impute_unknowns_housing_loan(list_of_dfs):
    print("      └── Imputing unknown values `housing_loan` with 'no'...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    for df in list_of_dfs:
        # Standardize all entries to lowercase for consistency
        df['housing_loan'] = df['housing_loan'].str.lower()

        # Replace 'unknown' with 'no' as they represent similar perceived behavior
        df['housing_loan'] = df['housing_loan'].replace('unknown', 'no')

        # Convert the column to categorical type
        df['housing_loan'] = df['housing_loan'].astype('category')

        # Append the modified DataFrame
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def impute_personal_loan(list_of_dfs):
    print("      └── Imputing unknown `personal_loan` value with 'no'...")
    print("      └── Imputing missing `personal_loan` values using K-Modes...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    modified_dfs = []

    for df in list_of_dfs:
        df = df.copy()

        # Lowercase values and replace 'unknown' with 'no'
        df['personal_loan'] = df['personal_loan'].str.lower().replace({'unknown': 'no'})

        # Create a mask for missing values (which could be 'missing' or pd.NA)
        missing_mask = (df['personal_loan'] == 'missing') | (df['personal_loan'].isna())

        # If there are no missing values, just append the current dataframe and continue
        if not missing_mask.any():
            modified_dfs.append(df)
            continue

        # Get indices of non-missing and missing values
        non_missing_indices = df.index[~missing_mask]
        missing_indices = df.index[missing_mask]

        # Prepare data for KModes
        # Select features that might be relevant for predicting personal_loan
        features_to_use = df.drop(columns=['personal_loan']).columns.tolist()

        if not features_to_use:
            # If no predictive features are available, just use a simple mode imputation
            mode_value = df.loc[~missing_mask, 'personal_loan'].mode()[0]
            df.loc[missing_mask, 'personal_loan'] = mode_value
        else:
            # Create a training dataset with the non-missing values
            X_train = df.loc[non_missing_indices, features_to_use]
            y_train = df.loc[non_missing_indices, 'personal_loan']

            # Handle categorical features - convert to strings for KModes
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                    X_train[col] = X_train[col].astype(str)
                else:
                    # Normalize numerical features to avoid scale issues
                    X_train[col] = (X_train[col] - X_train[col].mean()) / X_train[col].std()

            # Create a test dataset with the missing values
            X_test = df.loc[missing_indices, features_to_use]

            # Handle categorical features in test data
            for col in X_test.columns:
                if X_test[col].dtype == 'object' or X_test[col].dtype.name == 'category':
                    X_test[col] = X_test[col].astype(str)
                else:
                    # Use the same normalization from training data
                    mean_val = X_train[col].mean()
                    std_val = X_train[col].std()
                    X_test[col] = (X_test[col] - mean_val) / std_val

            # Handle any NaN values in features by filling with mode or mean
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                    fill_value = X_train[col].mode()[0]
                    X_train[col] = X_train[col].fillna(fill_value)
                    X_test[col] = X_test[col].fillna(fill_value)
                else:
                    X_train[col] = X_train[col].fillna(X_train[col].mean())
                    X_test[col] = X_test[col].fillna(X_train[col].mean())

            # Apply KModes clustering to find patterns
            n_clusters = min(5, len(non_missing_indices))  # Adjust number of clusters based on data size
            kmodes = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
            kmodes.fit(X_train)

            # Predict clusters for both training and test data
            train_clusters = kmodes.predict(X_train)
            test_clusters = kmodes.predict(X_test)

            # For each missing value, find its cluster and use the most common 'personal_loan' value from that cluster
            for i, test_idx in enumerate(missing_indices):
                test_cluster = test_clusters[i]

                # Find all training points in the same cluster
                same_cluster_indices = np.where(train_clusters == test_cluster)[0]

                if len(same_cluster_indices) > 0:
                    # Get the mode of personal_loan values in this cluster
                    cluster_personal_loan_values = y_train.iloc[same_cluster_indices]
                    most_common_value = cluster_personal_loan_values.mode()[0]
                    df.loc[test_idx, 'personal_loan'] = most_common_value
                else:
                    # If no training points in this cluster, use overall mode
                    df.loc[test_idx, 'personal_loan'] = y_train.mode()[0]

        # Ensure the personal_loan column is categorical
        df['personal_loan'] = df['personal_loan'].astype('category')

        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def cap_campaign_calls(list_of_dfs, cap_value=11):
        print("      └── Capping `campaign_calls` feature...")

        if not list_of_dfs:
            raise ValueError("The input list of DataFrames cannot be empty.")
        
        # Decide cap value
        if cap_value < 1:
            quantile_val = df['campaign_calls'].quantile(cap_value)
            cap_used = int(round(quantile_val))
            print(f"      └── Capping at {int(cap_value * 100)}th percentile = {cap_used}")
        else:
            cap_used = cap_value
            print(f"      └── Capping at fixed value = {cap_used}")

        # Initialize a list to store the modified DataFrames
        modified_dfs = []

        # Iterate over each DataFrame and apply the capping logic
        for i, df in enumerate(list_of_dfs, start=1):

            # Ensure `campaign_calls` is numerical before clipping
            df['campaign_calls'] = pd.to_numeric(df['campaign_calls'], errors='coerce')

            # Apply capping
            df['campaign_calls'] = df['campaign_calls'].clip(upper=cap_used)

            # Append the modified DataFrame to the results list
            modified_dfs.append(df)

        return modified_dfs

#################################################################################################################################
#################################################################################################################################
def bin_previous_contact_days(list_of_dfs):
    print("      └── Binning `previous_contact_days` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Define binning categories based on `previous_contact_days` values
    def bin_contact_days(value):
        if value == 999:
            return 'no-contact'
        elif value in [0, 1, 2, 3]:
            return 'recent-contact'
        elif value in [4, 5, 6]:
            return 'moderate-contact'
        elif value >= 7:
            return 'long-contact'

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the binning logic
    for df in list_of_dfs:
        # Apply binning
        df['previous_contact_days'] = df['previous_contact_days'].apply(bin_contact_days)

        # Convert the column to categorical type
        df['previous_contact_days'] = df['previous_contact_days'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_education_group(list_of_dfs):
    print("      └── Creating `education_group` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Define the mapping logic for education levels
    def map_education_level(level):
        if level in ['basic-4y', 'basic-6y', 'basic-9y', 'illiterate', 'unknown']:
            return 'low'
        elif level in ['high-school', 'professional-course']:
            return 'medium'
        elif level == 'university-degree':
            return 'high'
        else:
            return 'low'

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    for df in list_of_dfs:
        # Apply the mapping to create the new feature
        df['education_group'] = df['education_level'].apply(map_education_level)

        # Reorder columns to place 'education_group' after 'education_level'
        cols = df.columns.tolist()
        edu_idx = cols.index('education_level') + 1
        cols.insert(edu_idx, 'education_group')
        df = df[cols]

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert to categorical
        df['education_group'] = df['education_group'].astype('category')

        # Append to results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_financial_stability(list_of_dfs):
    print("      └── Creating `financial_stability` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Define the mapping logic for perceived financial stability
    def map_financial_stability(occupation):
        if occupation in ['services', 'retired', 'housemaid', 'entrepreneur']:
            return 'low'  
        elif occupation in ['admin', 'blue-collar', 'technician', 'self-employed']:
            return 'medium'  
        elif occupation in ['management']:
            return 'high'  
        elif occupation in ['unemployed', 'student']:
            return 'none'  
        else:
            return 'none'

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    for df in list_of_dfs:
        # Apply the mapping to create the new feature
        df['financial_stability'] = df['occupation'].apply(map_financial_stability)

        # Reorder columns to place 'financial_stability' after 'occupation'
        cols = df.columns.tolist()
        occ_idx = cols.index('occupation') + 1
        cols.insert(occ_idx, 'financial_stability')
        df = df[cols]

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert to categorical
        df['financial_stability'] = df['financial_stability'].astype('category')

        # Append to results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_is_married(list_of_dfs):
    print("      └── Creating `is_married` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for df in list_of_dfs:
        # Create 'is_married' feature: 'yes' if married, 'no' otherwise (single or divorced)
        df['is_married'] = df['marital_status'].apply(lambda x: 'yes' if x == 'married' else 'no')

        # Reorder columns to place 'is_married' before 'education_level'
        cols = df.columns.tolist()
        insert_at = cols.index('education_level')
        cols.insert(insert_at, 'is_married')
        df = df[cols]

        # Remove duplicated columns, if any
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert to categorical
        df['is_married'] = df['is_married'].astype('category')

        # Append to list
        modified_dfs.append(df)

    return modified_dfs
    
#################################################################################################################################
#################################################################################################################################
def create_demographic_clusters(list_of_dfs):
    print("      └── Creating demographic clusters via K-Prototypes clustering...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Predefined categorical and numerical features
    demographic_features_cat = ['occupation', 'financial_stability', 'is_married', 'education_group', 'education_level']
    demographic_features_num = ['age']
    demographic_features = demographic_features_cat + demographic_features_num

    # Define the folder path for the preprocessing model
    folder_path = 'modules'
    os.makedirs(folder_path, exist_ok=True)
    model_path = os.path.join(folder_path, 'kproto_model.joblib')

    # Check if the model exists before trying to load
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")

    # Load the saved K-Prototypes model using joblib
    kproto = joblib.load(model_path)

    # Get the indices of the categorical features
    categorical_indices = [i for i, col in enumerate(demographic_features) if col in demographic_features_cat]
    
    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the clustering logic
    for df in list_of_dfs:
        # Create a copy of the DataFrame for clustering
        df_demographic = df[demographic_features].copy()

        # Predict cluster labels
        predicted_clusters = kproto.predict(df_demographic, categorical=categorical_indices)

        # Attach predictions to the DataFrame
        df['demographic_cluster'] = predicted_clusters

        # Reorder columns to place 'demographic_cluster' at the first column
        cols = df.columns.tolist()
        cols.insert(0, 'demographic_cluster')  # Insert at the first position (index 0)
        df = df[cols]

        # Remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert the column to categorical type
        df['demographic_cluster'] = df['demographic_cluster'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_has_loan(list_of_dfs):
    print("      └── Creating `has_loan` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for df in list_of_dfs:
        
        # Create the 'has_loan' feature based on conditions
        df['has_loan'] = ((df['housing_loan'] == 'yes') | (df['personal_loan'] == 'yes')).map({True: 'yes', False: 'no'})

        # Handle the case where 'housing_loan' is missing and 'personal_loan' is 'no'
        df['has_loan'] = df.apply(
            lambda x: 'yes' if (x['housing_loan'] == 'yes' or x['personal_loan'] == 'yes') else 'no', axis=1
        )

        # Reorder columns to place 'has_loan' before 'contact_method'
        cols = df.columns.tolist()  
        age_index = cols.index('contact_method') 
        cols.insert(age_index, 'has_loan') 
        df = df[cols]

        # Drop the duplicate 'age_category' column
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert the column to categorical type
        df['has_loan'] = df['has_loan'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_contacted_multiple_times(list_of_dfs):
    print("      └── Creating `contacted_multiple_times` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for df in list_of_dfs:
        # Create the 'contacted_multiple_times' feature with 'yes' or 'no'
        df['contacted_multiple_times'] = df['campaign_calls'].apply(lambda x: 'yes' if x > 1 else 'no')

        # Reorder columns to place 'contacted_multiple_times' after 'campaign_calls'
        cols = df.columns.tolist()
        camp_index = cols.index('campaign_calls') - 1
        cols.insert(camp_index, 'contacted_multiple_times')
        df = df[cols]

        # Remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert the column to categorical type
        df['contacted_multiple_times'] = df['contacted_multiple_times'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_is_contact_method_cellular(list_of_dfs):
    print("      └── Creating `is_contact_method_cellular` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for df in list_of_dfs:
        # Create the 'is_contact_method_cellular' feature with 'yes' for 'cellular', 'no' for 'telephone'
        df['is_contact_method_cellular'] = df['contact_method'].apply(lambda x: 'yes' if x == 'cellular' else 'no')

        # Reorder columns to place 'is_contact_method_cellular' after 'contact_method'
        cols = df.columns.tolist()
        idx = cols.index('contact_method') + 1
        cols.insert(idx, 'is_contact_method_cellular')
        df = df[cols]

        # Remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # Convert the column to categorical type
        df['is_contact_method_cellular'] = df['is_contact_method_cellular'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def drop_irrelevant_features(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Define features to remove
    features_to_drop = [
        # "demographic_cluster",
        # "age_category",  
        # "age",
        # "occupation",
        # "financial_stability",
        # "education_group",
        # "marital_status",
        # "is_married",
        # "education_level",
        # "credit_default",
        # "housing_loan",
        # "personal_loan",
        # "has_loan",
        # "contacted_multiple_times",
        "contact_method",
        # "is_contact_method_cellular",
        # "campaign_calls",
        # "has_prior_contact",
        # "previous_contact_days"
    ]

    # Initialize a list to store the modified DataFrames
    list_of_modified_dfs = []

    # Loop through each DataFrame and apply the logic
    for i, df in enumerate(list_of_dfs, start=1):        
        # Drop the specified features
        modified_df = df.drop(columns=features_to_drop, errors='ignore')
        
        # Append the modified DataFrame to the results list
        list_of_modified_dfs.append(modified_df)

    # Print confirmation
    print(f"      └── Dropped features: {features_to_drop}")
    
    # Return the list of modified DataFrames
    return list_of_modified_dfs

#################################################################################################################################
#################################################################################################################################
def transform_features(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    print("      └── Defining numerical and categorical features...")
    df_sample = list_of_dfs[0].copy()  # Create a copy to avoid modifying the original
    
    # Define feature groups
    numerical_features = df_sample.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define ordinal features and their respective orders
    ordinal_features = ['education_group', 'financial_stability']
    ordinal_order = {
        'education_group': ['low', 'medium', 'high'],
        'financial_stability': ['none', 'low', 'medium', 'high'],
    }
    
    # Remove ordinal features from other lists to avoid duplication
    categorical_features = [col for col in categorical_features if col not in ordinal_features]
    numerical_features = [col for col in numerical_features if col not in ordinal_features]
    
    # Ensure ordinal features are present in the dataframe and convert them to string type
    for feature in ordinal_features:
        if feature in df_sample.columns:
            # Convert to string first to ensure consistent type
            for df in list_of_dfs:
                df[feature] = df[feature].astype(str)
    
    print("      └── Defining the feature transformer pipeline...")
    # Pipeline for numerical features
    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    
    # Pipeline for categorical features (OneHotEncoder)
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    # Pipeline for ordinal features
    ordinal_transformers = []
    for ordinal_feature in ordinal_features:
        if ordinal_feature in df_sample.columns:
            # Define the ordinal encoder for each feature with its predefined categories
            ordinal_pipeline = Pipeline([
                ('ordinal', OrdinalEncoder(categories=[ordinal_order[ordinal_feature]], 
                                          dtype=float))  # Use float type to avoid int conversion issues
            ])
            ordinal_transformers.append((ordinal_feature, ordinal_pipeline, [ordinal_feature]))
    
    # Combine all into a column transformer
    transformers = [
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
    
    # Add ordinal transformers if they exist
    transformers.extend(ordinal_transformers)
    
    # Apply the column transformer
    transformer = ColumnTransformer(transformers)
    
    print("      └── Executing the pipeline...")
    list_of_modified_dfs = []
    
    # Apply transformation to each dataframe
    for i, df in enumerate(list_of_dfs):
        if i == 0:
            transformed_data = transformer.fit_transform(df)
        else:
            transformed_data = transformer.transform(df)
        
        # Convert the transformed data to DataFrame
        feature_names = transformer.get_feature_names_out()
        preprocessed_df = pd.DataFrame(transformed_data, columns=feature_names)
        list_of_modified_dfs.append(preprocessed_df)
    
    return list_of_modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_features_based_on_feature_selection(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Exact feature names to remove
    exact_features_to_remove = [
        "age_category",
        "education_level",
        "campaign_calls",
        # Add more exact feature names here
    ]

    # Wildcard patterns to match feature groups
    feature_patterns = [
        "cat__age_category_*",
        "cat__education_level_*",
        "num__campaign_calls",
        # Add more wildcard patterns here
    ]
    
    list_of_modified_dfs = []
    all_features_removed_from_dfs = set()

    # Process original DataFrames
    for df in list_of_dfs:
        # Track exact features to remove from original DataFrames
        features_to_remove = [col for col in exact_features_to_remove if col in df.columns]
        
        # Track wildcard features to remove based on patterns
        for pattern in feature_patterns:
            features_to_remove.extend([col for col in df.columns if fnmatch.fnmatch(col, pattern)])

        # Ensure we remove only unique features (avoid duplicates)
        features_to_remove = list(set(features_to_remove))
        
        # Track what was actually removed from this DataFrame
        removed = [col for col in features_to_remove if col in df.columns]
        all_features_removed_from_dfs.update(removed)

        # Drop the columns
        modified_df = df.drop(columns=features_to_remove, errors='ignore')
        list_of_modified_dfs.append(modified_df)

    # Print final report of all removed features
    print(f"      └── Removed features from DataFrames: {sorted(all_features_removed_from_dfs)}")

    return list_of_modified_dfs