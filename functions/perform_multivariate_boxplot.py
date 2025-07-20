# src/perform_multivariate_boxplot.py

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def perform_multivariate_boxplot(df, categorical_var1, categorical_var2, numerical_var, figsize=(12, 6)):
    """
    Generates a box plot to visualize the distribution of a numerical variable across combinations of two categorical variables. Also includes descriptive statistics and ANOVA results to understand the relationship.

    Parameters:
    - df: The dataframe containing the data
    - categorical_var1: The first categorical variable (x-axis)
    - categorical_var2: The second categorical variable (hue)
    - numerical_var: The numerical variable (y-axis)
    - figsize: Tuple specifying the size of the figure (default is (12, 6))
    
    Returns:
    - None: Displays the plot and prints the metrics
    """
    # Step 1: Set up the plot
    plt.figure(figsize=figsize)

    # Step 2: Create the box plot
    sns.boxplot(
        x=categorical_var1,
        y=numerical_var,
        hue=categorical_var2,
        data=df
    )

    # Step 3: Customize the plot with labels and title
    plt.title(f'Box Plot of {numerical_var} by {categorical_var1} and {categorical_var2}', fontsize=16)
    plt.xlabel(categorical_var1, fontsize=12)
    plt.ylabel(numerical_var, fontsize=12)
    plt.legend(title=categorical_var2, loc='upper right')

    # Step 4: Show the plot
    plt.show()

    # Step 5: Descriptive Statistics
    print("\n--- Descriptive Statistics ---")
    stats_summary = df.groupby([categorical_var1, categorical_var2])[numerical_var].describe()
    print(stats_summary.to_string())

    # Step 6: ANOVA Test
    print("\n--- ANOVA Test Results ---")
    grouped_data = [group[numerical_var].values for _, group in df.groupby([categorical_var1, categorical_var2])]
    anova_result = stats.f_oneway(*grouped_data)

    # Print formatted ANOVA results
    print(f"F-statistic: {anova_result.statistic:.4f}")
    print(f"p-value: {anova_result.pvalue:.4f}")
    
    if anova_result.pvalue < 0.05:
        print("The differences between the groups are statistically significant (p < 0.05).")
    else:
        print("The differences between the groups are not statistically significant (p >= 0.05).")