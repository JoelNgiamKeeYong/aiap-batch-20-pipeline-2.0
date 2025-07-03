# src/perform_multivariate_cat_analysis.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

def perform_multivariate_cat_analysis(df, cat_var1, cat_var2, target_var, figsize_heatmap=(12, 8), figsize_bar=(12, 8), show_crosstab=True, show_stats=True, show_heatmap=False, show_bar=False):
    """
    Performs multivariate analysis of three categorical variables:
    - 2 predictors and 1 target variable.
    - Generates a normalized crosstab, a heatmap, and a stacked bar plot to visualize relationships.
    - Performs a Chi-Square Test of Independence.

    Parameters:
    - df: pandas DataFrame containing the data
    - cat_var1: First categorical variable (e.g., age_category)
    - cat_var2: Second categorical variable (e.g., education_level)
    - target_var: Target categorical variable (e.g., subscription_status)
    - figsize_heatmap: Tuple for heatmap size
    - figsize_bar: Tuple for bar plot size
    - show_stats: Boolean to show Chi-square test results

    Returns:
    - None: Displays visualizations and prints test results
    """
    # Step 1: Crosstabulation
    crosstab = pd.crosstab(index=[df[cat_var1], df[cat_var2]],columns=df[target_var])
    crosstab_normalized = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

    # Step 2: Statistical Test (Chi-Square)
    if show_stats:
        contingency_table = pd.crosstab(index=[df[cat_var1], df[cat_var2]],
                                        columns=df[target_var])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        print("\n📊 Chi-Square Test of Independence:")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"Degrees of Freedom: {dof}")
        print(f"P-value: {p:.4e}")

        if p < 0.05:
            print("✅ Result: Statistically significant association (reject H0)")
        else:
            print("❌ Result: No statistically significant association (fail to reject H0)")

    # Step 3: Heatmap
    target_classes = sorted(crosstab_normalized.columns.tolist())
    if show_heatmap:
        if target_classes:
            target_focus = target_classes[-1] 
            try:
                heatmap_data = crosstab_normalized.reset_index().pivot(index=cat_var1,
                                                            columns=cat_var2,
                                                            values=target_focus)

                plt.figure(figsize=figsize_heatmap)
                sns.heatmap(heatmap_data, annot=True, fmt=".2f")
                plt.title(f"Proportion of '{target_var}={target_focus}' by {cat_var1} and {cat_var2}")
                plt.xlabel(cat_var2)
                plt.ylabel(cat_var1)
                plt.tight_layout()
                plt.show()
            except KeyError:
                print(f"\n⚠️ Target class '{target_focus}' not found – skipping heatmap.")
        else:
            print(f"\n⚠️ No classes found in target variable '{target_var}'.")

    # Step 4: Percent Stacked Bar Plot
    if show_bar:
        grouped = df.groupby([cat_var1, cat_var2, target_var]).size().reset_index(name='count')
        total = grouped.groupby([cat_var1, cat_var2])['count'].transform('sum')
        grouped['percent'] = grouped['count'] / total

        pivot_df = grouped.pivot_table(index=[cat_var1, cat_var2],
                                       columns=target_var,
                                       values='percent',
                                       fill_value=0)

        pivot_df.plot(kind='bar', stacked=True, figsize=figsize_bar)
        plt.title(f'{target_var} by {cat_var1} and {cat_var2}')
        plt.ylabel('Percentage')
        plt.xlabel(f'{cat_var1} and {cat_var2}')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=target_var)
        plt.tight_layout()
        plt.show()

    # Step 5: Display Crosstab
    if show_crosstab:
        print("\n🔍 Crosstab:")
        print(crosstab)     