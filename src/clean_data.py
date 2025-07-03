# src/clean_data.py

import os
import time
import re
import pandas as pd

def clean_data(df):
    """
    This function performs a series of cleaning and preprocessing steps to ensure the dataset is ready for downstream tasks such as feature engineering and model training. The steps include removing irrelevant features, handling missing values, standardizing categorical and numerical columns, and saving the cleaned dataset for reuse.

    Cleaning functions are applied sequentially as per the 🧼 indicators identified during Exploratory Data Analysis (EDA). If a cleaned dataset already exists in the specified output path, the function skips the cleaning process and loads the existing file.

    Parameters:
        df (pd.DataFrame): 
            The raw dataset to be cleaned. This should be a pandas DataFrame containing all relevant features and target variables.

    Returns:
        pd.DataFrame: 
            A cleaned and preprocessed DataFrame ready for further analysis or modeling.

    Raises:
        RuntimeError: 
            If an error occurs during the cleaning process, a RuntimeError is raised with details about the failure.

    Example Usage:
        >>> raw_data = pd.read_csv("data/raw_data.csv")
        >>> cleaned_data = clean_data(raw_data)
        >>> print(cleaned_data.head())
    """
    try:
        # Define output path
        output_path = "./data/cleaned_data.csv"

        # Check if the cleaned data file already exists
        if os.path.exists(output_path):
            print(f"\n✅ Found existing cleaned data. Skipping cleaning process...")
            return pd.read_csv(output_path)

        # Carry out cleaning id cleaned data file does not exist
        print("\n🧼 Cleaning the dataset...")
        start_time = time.time()

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Standardize feature names
        df_cleaned = convert_column_names_to_snake_case(df_cleaned)

        # Remove irrelevant features
        df_cleaned = drop_client_id(df_cleaned)

        # Apply cleaning steps to specified columns
        print("   └── Cleaning specified columns...")
        df_cleaned = clean_subscription_status(df_cleaned)
        df_cleaned = clean_age(df_cleaned)
        df_cleaned = clean_occupation(df_cleaned)
        df_cleaned = clean_marital_status(df_cleaned)
        df_cleaned = clean_education_level(df_cleaned)
        df_cleaned = clean_credit_default(df_cleaned)
        df_cleaned = clean_housing_loan(df_cleaned)
        df_cleaned = clean_personal_loan(df_cleaned)
        df_cleaned = clean_contact_method(df_cleaned)  
        df_cleaned = clean_campaign_calls(df_cleaned)
        df_cleaned = clean_previous_contact_days(df_cleaned)
        
        # Save the cleaned data to a CSV file
        print(f"\n💾 Saving cleaned data to {output_path}...")
        df_cleaned.to_csv(output_path, index=False)

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n✅ Data cleaning completed in {elapsed_time:.2f} seconds!")

        # Return cleaned dataset
        return df_cleaned

    except Exception as e:
        print(f"\n❌ An error occurred during data cleaning: {e}")
        raise RuntimeError("Data cleaning process failed.") from e


#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
def convert_column_names_to_snake_case(df_cleaned):
    # Convert column names to snake_case
    print("   └── Converting column names to snake_case...")
    new_columns = []
    for col in df_cleaned.columns:
        # Convert CamelCase to snake_case and replace spaces/dots with underscore
        col = re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower()
        col = col.replace(' ', '_').replace('.', '_')
        new_columns.append(col)
    df_cleaned.columns = new_columns
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def drop_client_id(df_cleaned):
    # Drop column
    print("   └── Dropping 'client_id' column...")
    df_cleaned = df_cleaned.drop(['client_id'], axis=1)
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_subscription_status(df_cleaned):
    print("\n   🫧  Cleaning target variable column (i.e.'subscription_status')...")
    
    # Binary encode the 'subscription_status' column ("yes" = 1, "no" = 0)
    print("      └── Binary encoding 'subscription_status' column ('yes' = 1, 'no' = 0)...")
    df_cleaned['subscription_status'] = df_cleaned['subscription_status'].map({"yes": 1, "no": 0})
    
    # Convert the column to categorical type for memory efficiency
    print("      └── Converting 'subscription_status' column to categorical type...")
    df_cleaned['subscription_status'] = df_cleaned['subscription_status'].astype('category')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_age(df_cleaned):
    print("\n   🫧  Cleaning 'age' column...")

    # Remove the "years" suffix and extract numeric values
    print("      └── Removing 'years' suffix and converting `age` column to numeric values...")
    df_cleaned['age'] = df_cleaned['age'].str.replace(' years', '', regex=False)

    # Convert the column to integer type
    print("      └── Converting 'age' column to int64 type...")
    df_cleaned['age'] = pd.to_numeric(df_cleaned['age'], errors='coerce').astype('Int64')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_occupation(df_cleaned):
    print("\n   🫧  Cleaning 'occupation' column...")

    # Remove the period in 'admin.'
    print("      └── Removing period in 'admin.'...")
    df_cleaned['occupation'] = df_cleaned['occupation'].str.replace('admin.', 'admin', regex=False)

    # Convert the column to categorical type
    print("      └── Converting 'occupation' column to category data type...")
    df_cleaned['occupation'] = df_cleaned['occupation'].astype('category')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_marital_status(df_cleaned):
    print("\n   🫧  Cleaning 'marital_status' column...")

    # Convert the column to categorical type
    print("      └── Converting 'marital_status' column to category data type...")
    df_cleaned['marital_status'] = df_cleaned['marital_status'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_education_level(df_cleaned):
    print("\n   🫧  Cleaning 'education_level' column...")

    # Step 1: Replace periods with hyphens
    print("      └── Replacing periods with hyphens in 'education_level' column...")
    df_cleaned['education_level'] = df_cleaned['education_level'].str.replace('.', '-', regex=False)

    # Convert the column to integer type
    print(f"      └── Converting 'education_level' column to category type...")
    df_cleaned['education_level'] = df_cleaned['education_level'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_credit_default(df_cleaned):
    print("\n   🫧  Cleaning 'credit_default' column...")

    # Convert the column to categorical type
    print("      └── Converting 'credit_default' column to categorical type...")
    df_cleaned['credit_default'] = df_cleaned['credit_default'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_housing_loan(df_cleaned):
    print("\n   🫧  Cleaning 'housing_loan' column...")

    # Impute missing values (None) with "Missing"
    print("      └── Imputing missing values with 'Missing'...")
    df_cleaned['housing_loan'] = df_cleaned['housing_loan'].fillna('missing')

    # Convert the column to categorical type
    print("      └── Converting 'housing_loan' column to categorical type...")
    df_cleaned['housing_loan'] = df_cleaned['housing_loan'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_personal_loan(df_cleaned):
    print("\n   🫧  Cleaning 'personal_loan' column...")

    # Impute missing values (None) with "Missing"
    print("      └── Imputing missing values with 'Missing'...")
    df_cleaned['personal_loan'] = df_cleaned['personal_loan'].fillna('missing')

    # Convert the column to categorical type
    print("      └── Converting 'personal_loan' column to categorical type...")
    df_cleaned['personal_loan'] = df_cleaned['personal_loan'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_contact_method(df_cleaned):
    print("\n   🫧  Cleaning 'contact_method' column...")

    # Standardize inconsistent string values
    print("      └── Standardizing inconsistent string values...")
    contact_mapping = {
        'Cell': 'cellular',       
        'cellular': 'cellular',  
        'Telephone': 'telephone', 
        'telephone': 'telephone' 
    }
    df_cleaned['contact_method'] = df_cleaned['contact_method'].map(contact_mapping)

    # Convert the column to categorical type
    print("      └── Converting 'contact_method' column to category data type...")
    df_cleaned['contact_method'] = df_cleaned['contact_method'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_campaign_calls(df_cleaned):
    print("\n   🫧  Cleaning 'campaign_calls' column...")

    # Handle negative values by taking the absolute value (mod)
    print("      └── Taking the absolute value of negative entries in 'campaign_calls'...")
    negative_count = (df_cleaned['campaign_calls'] < 0).sum() 
    df_cleaned['campaign_calls'] = df_cleaned['campaign_calls'].abs()
    print(f"      └── Corrected {negative_count} negative values using absolute value (mod).")

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_previous_contact_days(df_cleaned):
    print("\n   🫧  Cleaning 'previous_contact_days' column...")

    # Add a binary feature `has_prior_contact`
    print("      └── Adding binary feature 'has_prior_contact'...")
    has_prior_contact = df_cleaned['previous_contact_days'].apply(lambda x: 'yes' if x != 999 else 'no')

    # Insert the new column before the `previous_contact_days` column
    previous_contact_days_index = df_cleaned.columns.get_loc('previous_contact_days')
    df_cleaned.insert(previous_contact_days_index, 'has_prior_contact', has_prior_contact)

    # Convert the column to categorical type
    print("      └── Converting 'contact_method' column to category data type...")
    df_cleaned['has_prior_contact'] = df_cleaned['has_prior_contact'].astype('category')

    return df_cleaned