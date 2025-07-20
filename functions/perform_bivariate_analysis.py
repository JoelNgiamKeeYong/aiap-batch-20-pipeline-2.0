import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from IPython.display import display
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr, f_oneway

from sklearn.linear_model import LinearRegression


def perform_bivariate_analysis(df, col1, col2, show_plots=True):
    """
    Perform bivariate analysis between two columns in a DataFrame.

    This function analyzes the relationship between two columns in a DataFrame by determining their types (categorical or numerical) and applying appropriate statistical tests and visualizations. It provides insights into patterns, correlations, and associations between the two variables.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The first column to analyze.
        col2 (str): The second column to analyze.

    Returns:
        None: Displays visualizations, statistics, and insights.
    """
    print(f"🔎 Performing bivariate analysis between:")

    # Validate inputs
    if col1 not in df.columns:
        raise ValueError(f"❌ Column '{col1}' not found in the DataFrame.")
    if col2 not in df.columns:
        raise ValueError(f"❌ Column '{col2}' not found in the DataFrame.")

    # Determine column types
    col1_type = "categorical" if df[col1].dtype in ['object', 'category'] else "numerical"
    col2_type = "categorical" if df[col2].dtype in ['object', 'category'] else "numerical"
    print(f" └── Column '{col1}' (Type: {col1_type})")
    print(f" └── Column '{col2}' (Type: {col2_type})\n")

    # Create a DataFrame with only the two columns of interest
    pair_df = df[[col1, col2]].dropna()

    # Graph design constants
    TITLE_COLOR = '#333333'
    PLOT_TITLE_COLOR = '#444444'

    ###########################################################################################################################################
    ###########################################################################################################################################
    # Case 1: Both columns are categorical
    if col1_type == "categorical" and col2_type == "categorical":
        
        # Crosstabs
        crosstab_raw = pd.crosstab(df[col1], df[col2])
        crosstab_proportions = pd.crosstab(df[col1], df[col2], normalize='index')

        # Chi-Square Test
        chi2, p, dof, expected = chi2_contingency(crosstab_raw)
        print(f"🧪 Chi2 Statistic: {chi2:.2f} (P-Value: {p:.4f})")

        # Cramér's V
        n = crosstab_raw.sum().sum()
        r, c = crosstab_raw.shape
        cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))
        print(f"🧪 Cramér's V: {cramers_v:.2f}")

        if p < 0.05:
            if cramers_v > 0.5:
                print("   └── ⚠️ Significant association with very strong effect.")
            elif cramers_v > 0.3:
                print("   └── ⚠️ Significant association with strong effect.")
            elif cramers_v > 0.1:
                print("   └── ⚠️ Significant association with moderate effect.")
            else:
                print("   └── Significant association with weak effect.")
        else:
            print("   └── No significant association.")

        if show_plots:
            unique_categories_col2 = df[col2].unique()
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))

            # Plot 1: Countplot
            sns.countplot(data=df, x=col1, hue=col2, palette='tab10', ax=axes[0, 0])
            axes[0, 0].set_title("Countplot with Hue", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
            axes[0, 0].set_xlabel("")
            axes[0, 0].set_ylabel("")
            axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)
            axes[0, 0].legend(title=col2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

            # Plot 2: Treemap
            color_palette = sns.color_palette("Greens", len(unique_categories_col2))
            color_map = dict(zip(unique_categories_col2, color_palette))
            proportions = df.groupby([col1, col2], observed=True).size().reset_index(name='counts')
            proportions['proportion'] = proportions['counts'] / proportions['counts'].sum()
            axes[0, 1].axis('off')
            squarify.plot(
                sizes=proportions['proportion'],
                label=proportions.apply(lambda x: f"{x[col1]}-{x[col2]}", axis=1),
                color=[color_map[cat] for cat in proportions[col2]],
                alpha=0.8,
                text_kwargs={'fontsize': 10},
                ax=axes[0, 1]
            )
            axes[0, 1].set_title("Treemap", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)

            # Plot 3: Heatmap with annotations
            counts_str = crosstab_raw.apply(lambda col: col.map(lambda x: f"{x:,}"))
            percent_str = (crosstab_proportions * 100).round(1).apply(lambda col: col.map(lambda x: f"{x}%"))
            annotations = counts_str + "\n(" + percent_str + ")"

            sns.heatmap(
                crosstab_proportions,
                annot=annotations,
                fmt="",
                cmap="coolwarm",
                cbar=False,
                annot_kws={"size": 10},
                ax=axes[1, 0]
            )
            axes[1, 0].set_title("Heatmap", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
            axes[1, 0].set_xlabel("")
            axes[1, 0].set_ylabel("")
            axes[1, 0].tick_params(axis='x', rotation=45, labelsize=10)

            # Plot 4: Stacked Bar Chart
            crosstab_proportions.plot(
                kind='bar',
                stacked=True,
                colormap='tab20',
                ax=axes[1, 1],
                width=0.8
            )
            axes[1, 1].set_title("Stacked Bar Chart", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
            axes[1, 1].set_xlabel("")
            axes[1, 1].set_ylabel("Proportion", fontsize=10)
            axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10)
            axes[1, 1].legend(title=col2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f"Graphical Analysis of '{col1}' vs. '{col2}'", fontsize=16, fontweight='bold', color=TITLE_COLOR)
            plt.show()

    ###########################################################################################################################################
    ###########################################################################################################################################
    # Case 2: One column is categorical, the other is numerical
    elif (col1_type == "categorical" and col2_type == "numerical") or \
     (col1_type == "numerical" and col2_type == "categorical"):
        cat_col, num_col = (col1, col2) if col1_type == "categorical" else (col2, col1)

        # Handle missing values
        pair_df = df[[cat_col, num_col]].dropna()

        # Handle rare categories by grouping them into "Other"
        top_categories = pair_df[cat_col].value_counts().index[:5]  # Keep top 5 categories

        # Check if there are rare categories (categories not in top_categories)
        rare_categories_exist = (~pair_df[cat_col].isin(top_categories)).any()

        # Add 'Other' as a valid category only if rare categories exist
        if rare_categories_exist and pd.api.types.is_categorical_dtype(pair_df[cat_col]):
            pair_df[cat_col] = pair_df[cat_col].cat.add_categories('Other')

        # Replace rare categories with 'Other' if they exist
        if rare_categories_exist:
            pair_df[cat_col] = pair_df[cat_col].where(pair_df[cat_col].isin(top_categories), 'Other')

        # 📑 Table 1: Group numerical data by categories
        grouped_data = pair_df.groupby(cat_col, observed=True)[num_col].agg(['mean', 'median', 'std']).reset_index()
        print("📑 Summary Statistics by Category:")
        display(grouped_data.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        # 🧪 Statistical Test 1: Point-biserial correlation if binary categorical variable
        if pair_df[cat_col].nunique() == 2:
            r, p_value = pointbiserialr(pair_df[cat_col].astype('category').cat.codes, pair_df[num_col])
            print(f"🧪 Point-biserial correlation: {r:.2f}, p-value: {p_value:.4f}")

            # Evaluate statistical significance and effect size
            if p_value < 0.05:  # Check if the result is statistically significant
                # Assess the strength of the correlation using the correlation coefficient (r)
                if abs(r) > 0.7:
                    print("   └── ⚠️ Significant correlation with very strong effect.")
                elif abs(r) > 0.5:
                    print("   └── ⚠️ Significant correlation with strong effect.")
                elif abs(r) > 0.3:
                    print("   └── ⚠️ Significant correlation with moderate effect.")
                else:
                    print("   └── Significant correlation with weak effect.")
            else:
                print("   └── No significant correlation.")

        # 🧪 Statistical Test 2: ANOVA-like comparison for non-binary categorical variables
        else:
            groups = [group[num_col].values for name, group in pair_df.groupby(cat_col, observed=True)]
            f_stat, p_value = f_oneway(*groups)
            print(f"🧪 ANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")

            # Evaluate statistical significance and effect size
            if p_value < 0.05:  # Check if the result is statistically significant
                if f_stat > 10:
                    print("   └── ⚠️ Significant differences with very strong evidence.")
                elif f_stat > 5:
                    print("   └── ⚠️ Significant differences with strong evidence.")
                elif f_stat > 2:
                    print("   └── ⚠️ Significant differences with moderate evidence.")
                else:
                    print("   └── Significant differences with weak evidence.")
            else:
                print("   └── No significant differences in means across categories.")

        # Create a figure with four subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 📊 Plot 1: Box Plot
        sns.boxplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            hue=cat_col,  
            palette="viridis",
            dodge=False,  
            ax=axes[0, 0],
            legend=False 
        )
        axes[0, 0].set_title(f"Box Plot: '{num_col}' Across '{cat_col}'", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("")

        # 📊 Plot 2: Violin Plot
        sns.violinplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            hue=cat_col,
            palette="viridis",
            dodge=False,  
            ax=axes[0, 1],
            legend=False 
        )
        axes[0, 1].set_title(f"Violin Plot: '{num_col}' Across '{cat_col}'", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")

        # 📊 Plot 3: Bar Chart of Mean Values
        sns.barplot(
            data=grouped_data,
            x=cat_col,
            y='mean',
            capsize=0.1,
            hue=cat_col,
            palette="viridis",
            dodge=False,  
            ax=axes[1, 0],
            legend=False 
        )
        axes[1, 0].set_title(f"Bar Chart: Mean '{num_col}' by '{cat_col}'", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("Mean Value", fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 📊 Plot 4: Scatter Plot with Jitter
        sns.stripplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            hue=cat_col,
            palette="viridis",
            dodge=False,  
            jitter=True,
            alpha=0.6,
            ax=axes[1, 1],
            legend=False 
        )
        axes[1, 1].set_title(f"Scatter Plot with Jitter: '{num_col}' by '{cat_col}'", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("")

        # Adjust layout for compactness
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"Graphical Analysis of '{cat_col}' vs. '{num_col}'", fontsize=16, fontweight='bold', color=TITLE_COLOR)
        plt.show()

    ###########################################################################################################################################
    ###########################################################################################################################################
    # Case 3: Both columns are numerical
    elif col1_type == "numerical" and col2_type == "numerical":

        # 🧪 Statistical Test 1: Pearson Correlation
        pearson_corr, pearson_p = pearsonr(pair_df[col1], pair_df[col2])
        print(f"🧪 Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p:.4f}")
        if pearson_p < 0.05:
            if pearson_corr < -0.7:
                print("   └── ⚠️ Significant correlation with very strong negative effect.")
            elif pearson_corr < -0.5:
                print("   └── ⚠️ Significant correlation with strong negative effect.")
            elif pearson_corr < -0.3:
                print("   └── ⚠️ Significant correlation with moderate negative effect.")
            elif pearson_corr < 0.3:
                print("   └── Significant correlation with weak effect.")
            elif pearson_corr < 0.5:
                print("   └── ⚠️ Significant correlation with moderate positive effect.")
            elif pearson_corr < 0.7:
                print("   └── ⚠️ Significant correlation with strong positive effect.")
            else:
                print("   └── ⚠️ Significant correlation with very strong positive effect.")
        else:
            print("   └── No significant linear correlation (Pearson).")

        # 🧪 Statistical Test 2: Spearman Correlation
        spearman_corr, spearman_p = spearmanr(pair_df[col1], pair_df[col2])
        print(f"\n🧪 Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p:.4f}")
        if spearman_p < 0.05:
            if spearman_corr < -0.7:
                print("   └── ⚠️ Significant monotonic correlation with very strong negative effect.")
            elif spearman_corr < -0.5:
                print("   └── ⚠️ Significant monotonic correlation with strong negative effect.")
            elif spearman_corr < -0.3:
                print("   └── ⚠️ Significant monotonic correlation with moderate negative effect.")
            elif spearman_corr < 0.3:
                print("   └── Significant monotonic correlation with weak effect.")
            elif spearman_corr < 0.5:
                print("   └── ⚠️ Significant monotonic correlation with moderate positive effect.")
            elif spearman_corr < 0.7:
                print("   └── ⚠️ Significant monotonic correlation with strong positive effect.")
            else:
                print("   └── ⚠️ Significant monotonic correlation with very strong positive effect.")
        else:
            print("   └── No significant monotonic correlation (Spearman).")

        if show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))

            # 📊 Plot 1: Scatterplot with Regression Line
            sns.regplot(
                data=pair_df,
                x=col1,
                y=col2,
                scatter_kws={'alpha': 0.6, 'color': '#4A90E2'},
                line_kws={'color': '#E94E77'},
                ax=axes[0, 0]
            )
            axes[0, 0].set_title("Scatter Plot with Regression Line", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)

            # 📊 Plot 2: Hexbin / 2D Histogram Plot
            sns.histplot(
                data=pair_df,
                x=col1,
                y=col2,
                bins=30,
                pthresh=0.1,
                cmap="viridis",
                cbar=True,
                ax=axes[0, 1]
            )
            axes[0, 1].set_title("Hexbin Plot: Density of Points", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)

            # 📊 Plot 3: Correlation Heatmap
            sns.heatmap(
                pair_df[[col1, col2]].corr(),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                cbar=False,
                ax=axes[1, 0]
            )
            axes[1, 0].set_title("Heatmap of Correlation Matrix", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)

            # 📊 Plot 4: Residual Plot
            model = LinearRegression()
            model.fit(pair_df[[col1]], pair_df[col2])
            predicted = model.predict(pair_df[[col1]])
            residuals = pair_df[col2] - predicted
            sns.scatterplot(x=predicted, y=residuals, ax=axes[1, 1], color="#50E3C2", alpha=0.6)
            axes[1, 1].axhline(0, color='#D0021B', linestyle='--', linewidth=1)
            axes[1, 1].set_title("Residual Plot", fontsize=12, fontweight='bold', color=PLOT_TITLE_COLOR)
            axes[1, 1].set_xlabel("Predicted Values")
            axes[1, 1].set_ylabel("Residuals")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f"Graphical Analysis of '{col1}' vs. '{col2}'", fontsize=16, fontweight='bold', color=TITLE_COLOR)
            plt.show()