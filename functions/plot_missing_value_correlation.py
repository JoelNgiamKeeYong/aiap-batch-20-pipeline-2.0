import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_value_correlation(df, columns, title="Missing Value Correlation Heatmap",
                                   cmap="flare", figsize=(12, 8), annot=True):
    """
    Plot a heatmap showing correlation between missing values of selected features.

    Parameters:
        df (pd.DataFrame): The dataset.
        columns (list): List of column names to check for missing value correlation.
        title (str): Title of the heatmap.
        cmap (str): Colormap used in the heatmap.
        figsize (tuple): Size of the plot.
        annot (bool): Whether to annotate the heatmap cells.

    Returns:
        None. Displays the heatmap.
    """
    # Apply ggplot style but fix background to avoid black spots
    plt.style.use('ggplot')
    sns.set_style("white")  # override dark grid patches
    
    if not all(col in df.columns for col in columns):
        missing = [col for col in columns if col not in df.columns]
        print(f"❌ Columns not found in DataFrame: {missing}")
        return

    null_corr = df[columns].isnull().corr()
    mask = np.triu(np.ones_like(null_corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        null_corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
        cbar=True
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()