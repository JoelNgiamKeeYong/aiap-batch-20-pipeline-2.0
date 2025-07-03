# src/utils/analyse_cat_proportions.py

import matplotlib.pyplot as plt
import pandas as pd

def analyse_cat_proportions(df, columns, title=None, show_graphs=True):
    """
    Analyzes and visualizes the proportions of categorical values in specified columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to analyze.
        title (str): Title for the visualization (default is "Proportions of Values in Financial Health Indicators (FHI)").

    Returns:
        None: Displays a summary table and a grouped horizontal bar chart.
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"Columns {missing_cols} not found in the DataFrame.")

    # Calculate normalized value counts and counts for each column
    proportion_dict = {}
    count_dict = {}
    for col in columns:
        count = df[col].value_counts()
        proportion = df[col].value_counts(normalize=True) * 100
        proportion_dict[col] = proportion
        count_dict[col] = count

    # Combine counts and proportions into a single DataFrame
    summary_df = pd.concat(
        {col: pd.concat([count_dict[col], proportion_dict[col]], axis=1, keys=['Count', 'Proportion (%)']) 
        for col in columns},
        axis=1
    )

    # Format only the 'Proportion (%)' columns
    for col in columns:
        summary_df[(col, 'Proportion (%)')] = summary_df[(col, 'Proportion (%)')].apply(lambda x: f"{x:.2f}%")

    # Display the summary table
    print(f"Proportions of Values in {title if title else columns[0]}:")
    print(summary_df)

    # Show graphs if enabled
    if show_graphs:

        # Create a copy of the summary DataFrame with numeric proportions for visualization
        summary_df_numeric = summary_df.copy()

        # Convert formatted percentages back to numeric values for plotting
        for col in columns:
            summary_df_numeric[(col, 'Proportion (%)')] = summary_df_numeric[(col, 'Proportion (%)')].str.rstrip('%').astype(float)

        # Create a grouped horizontal bar chart for visualization
        fig, ax = plt.subplots(figsize=(12, max(5, len(summary_df_numeric) * 0.8))) 
        bar_width = 0.25
        index = range(len(summary_df_numeric.index))

        # Define colors for each column
        colors = ['violet', 'orange', 'blue', 'green', 'red'][:len(columns)]

        # Plot bars for each column
        for i, col in enumerate(columns):
            bars = ax.barh(
                [idx + i * bar_width for idx in index],
                summary_df_numeric[(col, 'Proportion (%)')],  # Access only the 'Proportion (%)' column
                height=bar_width,
                label=col.replace('_', ' ').title(),
                color=colors[i]
            )
            # Add percentage annotations on the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}%",
                    ha='left',
                    va='center',
                    fontsize=9
                )

        # Add labels and formatting
        ax.set_yticks([i + bar_width * (len(columns) - 1) / 2 for i in index])
        ax.set_yticklabels(summary_df_numeric.index)
        ax.set_xlabel("Percentage (%)", fontsize=12)
        ax.set_title(title if title else columns[0], fontsize=14, fontweight='bold')
        ax.legend(title="Columns", fontsize=10, title_fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.show()