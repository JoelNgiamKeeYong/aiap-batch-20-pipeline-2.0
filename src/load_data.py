# load_data.py

import time
import sqlite3
import pandas as pd
from pathlib import Path

def load_data(db_path, db_table_name):
    """
    Loads the dataset from the specified SQLite database path and table name.

    This function performs the following steps:
    - Validates the existence of the SQLite database file.
    - Establishes a connection to the database.
    - Checks if the specified table exists in the database.
    - Queries the table and loads the data into a pandas DataFrame.
    - Closes the database connection after loading the data.

    Parameters:
        db_path (str): 
            Path to the SQLite database file.
        db_table_name (str): 
            Name of the table to load data from.

    Returns:
        pandas.DataFrame: 
            The dataset loaded into a DataFrame.

    Raises:
        FileNotFoundError: 
            If the database file is not found at the specified path.
        ValueError: 
            If the specified table does not exist in the database.
        sqlite3.Error: 
            If there is an error connecting to or querying the database.

    Example Usage:
        >>> db_path = "data/noshow.db"
        >>> db_table_name = "noshow"
        >>> data = load_data(db_path, db_table_name)
        >>> print(data.head())
    """
    print("📥 Starting data loading process...")
    start_time = time.time()

    # Validate that the database file exists
    print(f"   └── Validating database file...")
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database file not found at {db_path}. Please ensure the dataset is in the correct location.")
    print(f"   └── Database file found at: {db_path}")

    try:
        # Connect to the SQLite database
        print(f"   └── Connecting to SQLite database at path: {db_path}...")
        conn = sqlite3.connect(db_path)

        # Check if the specified table exists in the database
        print(f"   └── Checking if table '{db_table_name}' exists in the database...")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        if db_table_name not in tables:
            raise ValueError(f"❌ Table '{db_table_name}' not found in the database. Available tables: {tables}")

        # Query all data from the specified table
        print(f"   └── Loading data from table '{db_table_name}' into a pandas DataFrame...")
        query = f"SELECT * FROM {db_table_name}"
        df = pd.read_sql_query(query, conn)

    except sqlite3.Error as e:
        raise sqlite3.Error(f"❌ Database error: {e}") from e
    except Exception as e:
        raise Exception(f"❌ An unexpected error occurred: {e}") from e
    finally:
        # Ensure the database connection is closed
        if 'conn' in locals():
            conn.close()

    # Print summary and return the data
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ {len(df):,} records successfully loaded in {elapsed_time:.2f} seconds!")

    # Return the data
    return df