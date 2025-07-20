import os
import time
import pandas as pd
from datetime import datetime
from tabulate import tabulate


def log_training_summary(trained_models, start_time):
    """
    Logs training details, prints summary, and records pipeline execution time.
    """
    # Summary Table
    print("\n📊 Pipeline Summary Table:")
    table_data = [
        [model_name, f"{model_size_kb:.2f}", f"{training_time:.2f}", f"{evaluation_time:.2f}"]
        for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models
    ]
    headers = ["Model Name", "Model Size (KB)", "Training Time (s)", "Evaluation Time (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Total Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")

    # Archive Logs
    archives_dir = "archives"
    log_file_path = os.path.join(archives_dir, "training_logs.txt")
    os.makedirs(archives_dir, exist_ok=True)

    new_content = ""
    for model_name, best_model, training_time, model_size_kb, formatted_metrics, evaluation_time in trained_models:
        new_content += "=" * 135 + "\n"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_content += f"🕛 {current_time} | Model: {model_name}\n"
        metrics_table = tabulate(
            pd.DataFrame([formatted_metrics]).to_dict(orient='records'),
            headers="keys",
            tablefmt="grid",
            floatfmt=".2f"
        )
        new_content += metrics_table + "\n"
        new_content += f"└──── Model Size: {model_size_kb:.2f} KB\n"
        new_content += f"└──── Training Time: {training_time:.2f} seconds\n"
        new_content += f"└──── Evaluation Time: {evaluation_time:.2f} seconds\n"
        new_content += f"└──── Best Parameters: {best_model.get_params()}\n\n"

    if os.path.exists(log_file_path):
        with open(log_file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    else:
        existing_content = ""

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(new_content + existing_content)

    print("💾 Saved training logs to archives folder!")
