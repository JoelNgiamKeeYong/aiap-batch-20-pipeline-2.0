# src/utils/analyse_group_distributions.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyse_group_distributions(df, filtered_df, groupby_column, figsize=(12, 6), bar_color='teal', filtered_bar_color='orange'):
    """
    Analyzes the distribution of groups in a dataset by calculating raw counts and the proportion of filtered rows 
    (e.g., rows meeting a specific condition) within each group. Visualizes both metrics using bar charts.

    Parameters:
        df (pd.DataFrame): The full dataset containing all rows.
        filtered_df (pd.DataFrame): The filtered dataset containing only the rows of interest.
        groupby_column (str): The column to group by.
        figsize (tuple): Size of the figure for the plots (default is (12, 4)).
        bar_color (str): Color of the bars in the raw counts bar chart (default is 'teal').
        filtered_bar_color (str): Color of the bars in the filtered proportion bar chart (default is 'orange').

    Returns:
        None: Displays the grouped counts and filtered proportion bar charts.
    """
    # Step 1: Grouped Counts Analysis
    grouped_counts = (
        df
        .groupby(groupby_column)
        .size()
        .reset_index(name='count')
    )

    # Step 2: Filtered Proportion Analysis
    filtered_counts = (
        filtered_df
        .groupby(groupby_column)
        .size()
        .reset_index(name='filtered_count')
    )

    total_counts = (
        df
        .groupby(groupby_column)
        .size()
        .reset_index(name='total_count')
    )

    grouped = pd.merge(total_counts, filtered_counts, on=groupby_column, how='left')
    grouped['filtered_count'] = grouped['filtered_count'].fillna(0)
    grouped['percentage'] = (grouped['filtered_count'] / grouped['total_count']) * 100

    # Create a single figure with two subplots arranged vertically
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)  # Two rows, one column

    # Increase spacing between subplots using tight_layout and add padding
    plt.subplots_adjust(hspace=0.4) 

    # Plot 1: Grouped Counts
    ax1 = axes[0]
    sns.barplot(data=grouped_counts, x=groupby_column, y='count', color=bar_color, ax=ax1)

    # Add counts on top of each bar
    for p in ax1.patches:
        ax1.annotate(
            f'{int(p.get_height())}',  # Format the count as an integer
            (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the annotation
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            xytext=(0, 10),  # Increase offset for the label (x, y)
            textcoords='offset points',  # Coordinate system for the offset
            fontsize=10,
            color='black'
        )

    # Add titles and labels for the first plot
    ax1.set_title(f'Grouped Counts by {groupby_column.replace("_", " ").title()}', fontsize=16, fontweight='bold', pad=20) 
    ax1.set_ylabel('Count', fontsize=14, labelpad=15)  
    ax1.tick_params(axis='x', rotation=45, labelsize=12)  
    ax1.tick_params(axis='y', labelsize=12)  

    # Plot 2: Filtered Proportions
    ax2 = axes[1]
    sns.barplot(data=grouped, x=groupby_column, y='percentage', color=filtered_bar_color, ax=ax2)

    # Add percentages on top of each bar
    for p in ax2.patches:
        ax2.annotate(
            f'{p.get_height():.1f}%',  # Format the percentage to one decimal place
            (p.get_x() + p.get_width() / 2., p.get_height()),  # Position of the annotation
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            xytext=(0, 10),  # Increase offset for the label (x, y)
            textcoords='offset points',  # Coordinate system for the offset
            fontsize=10,
            color='black'
        )

    # Add titles and labels for the second plot
    ax2.set_title(f'Filtered Proportion by {groupby_column.replace("_", " ").title()}', fontsize=16, fontweight='bold', pad=20)  
    ax2.set_xlabel(groupby_column.replace('_', ' ').title(), fontsize=14, labelpad=15) 
    ax2.set_ylabel('Percentage (%)', fontsize=14, labelpad=15) 
    ax2.tick_params(axis='x', rotation=45, labelsize=12)  
    ax2.tick_params(axis='y', labelsize=12)  

    # Adjust layout to prevent overlap and ensure proper spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])  

    # Show the combined plot
    plt.show()