# src/perform_multivariate_boxplot.py

import matplotlib.pyplot as plt
import seaborn as sns

def perform_multivariate_countplot(df, features, target, figsize=(20, 6)):
    """
    Visualizes the distribution of multiple binary categorical features using count plots.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the binary features.
    - features (list of str): List of column names in `df` to plot.
    - figsize (tuple): Size of the entire figure (default=(20, 6)).
    - palette (str or list): Color palette to use for the plots (default='Set2').

    Returns:
    - None: Displays the count plots.
    """
    sns.set(style="whitegrid")
    num_features = len(features)
    fig, axes = plt.subplots(1, num_features, figsize=figsize)

    for i, feature in enumerate(features):
        ax = axes[i] if num_features > 1 else axes
        sns.countplot(data=df, x=feature, hue=target, ax=ax)
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel(feature.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()
