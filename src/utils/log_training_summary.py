# src/utils/log_training_summary.py

import os
import time
import pandas as pd
from datetime import datetime
from tabulate import tabulate

def log_training_summary(trained_models, start_time):
    """
    Logs and summarizes model training outcomes, including metrics, execution times, and model sizes.

    Parameters:
        trained_models : list of tuples
            A list containing training details for each model. Each tuple should include:
                (model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time)
            - model_name (str): Name of the model.
            - best_model (sklearn/BaseEstimator): The trained model object.
            - training_time (float): Time taken to train the model, in seconds.
            - model_size_kb (float): Size of the model in kilobytes.
            - formatted_metrics (dict): Dictionary of evaluation metrics for the model.
            - evaluation_time (float): Time taken to evaluate the model, in seconds.

        start_time : float
            The timestamp at which the pipeline started execution (typically from `time.time()`).

    Functionality:
        - Displays a tabulated summary of all trained models including training time, evaluation time, and model size.
        - Computes and prints the total pipeline execution time.
        - Archives detailed training logs, including:
            - Timestamp of training
            - Model parameters
            - Evaluation metrics in tabular format
            - Training and evaluation durations
            - Model size
        - Appends new logs to the top of an archived `training_logs.txt` file inside an `archives/` directory.

    Notes:
        - If the `archives/` folder does not exist, it will be created automatically.
        - Existing logs are preserved, and new entries are prepended to maintain chronological order.
    """
    print("\n🌐 Pipeline Summary")

    # Total Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"   └── Completed in {elapsed_time:.2f} seconds")

    # Archive Logs
    archives_dir = "archives"
    log_file_path = os.path.join(archives_dir, "training_logs.txt")
    os.makedirs(archives_dir, exist_ok=True)

    log_entries = []

    for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"{'='*30} 🧠 {model_name.upper()} | 🕛 {current_time} {'='*30}"
        metrics_table = tabulate(
            pd.DataFrame([formatted_metrics]).to_dict(orient='records'),
            headers="keys",
            tablefmt="grid",
            floatfmt=".2f"
        )

        model_info = [
            f"📦 Model Size     : {model_size_kb:.2f} KB",
            f"⏱️ Training Time  : {training_time:.2f} seconds",
            f"🔎 Evaluation Time: {evaluation_time:.2f} seconds"
        ]

        # Extract and clean parameters
        best_params = best_model.get_params()
        relevant_params = {k.replace("model__", ""): v for k, v in best_params.items() if 'model__' in k}

        # Sort alphabetically and format
        param_lines = ["⚙️ Best Parameters:"]
        for key in sorted(relevant_params):
            param_lines.append(f"   └── {key}: {relevant_params[key]}")

        # Combine all sections
        entry = "\n".join([
            header,
            metrics_table,
            *model_info,
            *param_lines,
            "\n"
        ])

        log_entries.append(entry)

    # Prepend new entries to log history
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries) + existing_content)

    # Summary Table
    table_data = [
        [model_name, f"{model_size_kb:.2f}", f"{training_time:.2f}", f"{evaluation_time:.2f}"]
        for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models
    ]
    headers = ["Model Name", "Model Size (KB)", "Training Time (s)", "Evaluation Time (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
