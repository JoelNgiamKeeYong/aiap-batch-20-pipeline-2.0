# src/utils/save_evaluation_metrics.py

import pandas as pd
from tabulate import tabulate

def save_evaluation_metrics(results, output_dir='output'):
    """
    Save consolidated classification / regression evaluation metrics to a text file in a formatted table.

    The function writes a summary of metrics for multiple models, showing test set metrics first, followed by training set metrics in parentheses. 
    
    The output is saved as a neatly formatted table using the 'tabulate' library.

    Args:
        results (list of dict): List of dictionaries containing metrics for each model.
        output_dir (str): Directory path where the output file will be saved.

    Returns:
        None
    """
    metrics_file_path = f"{output_dir}/evaluation_metrics_summary.txt"
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write("📋 Consolidated Evaluation Metrics:\n")
        f.write("(Note: Test metrics are shown first, followed by training metrics in brackets.)\n\n")
        metrics_table = tabulate(
            pd.DataFrame(results),
            headers="keys",
            tablefmt="grid",
            floatfmt=".3f"
        )
        f.write(metrics_table + "\n\n")