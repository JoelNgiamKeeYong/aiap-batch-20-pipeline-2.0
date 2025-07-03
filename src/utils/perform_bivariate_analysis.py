# src/utils/perform_bivariate_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr, pointbiserialr, f_oneway
from IPython.display import display
from sklearn.linear_model import LinearRegression

def perform_bivariate_analysis(df, col1, col2):
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
    # Validate inputs
    if col1 not in df.columns:
        raise ValueError(f"❌ Column '{col1}' not found in the DataFrame.")
    if col2 not in df.columns:
        raise ValueError(f"❌ Column '{col2}' not found in the DataFrame.")

    print(f"Performing bivariate analysis between:")

    # Determine column types
    col1_type = "categorical" if df[col1].dtype in ['object', 'category'] else "numerical"
    col2_type = "categorical" if df[col2].dtype in ['object', 'category'] else "numerical"
    print(f"   └── Column '{col1}' Type: {col1_type}")
    print(f"   └── Column '{col2}' Type: {col2_type}\n")

    # Create a DataFrame with only the two columns of interest
    pair_df = df[[col1, col2]].dropna()

    ###########################################################################################################################################
    ###########################################################################################################################################
    # Case 1: Both columns are categorical
    if col1_type == "categorical" and col2_type == "categorical":
        
        # Create crosstabs for raw counts and proportions
        crosstab_raw = pd.crosstab(df[col1], df[col2])  # Raw counts
        crosstab_proportions = pd.crosstab(df[col1], df[col2], normalize='index')  # Proportions
        
        # 🧪 Statistical Test 1: Chi-Square Test
        # - Tests the independence of two categorical variables.
        # - Assesses whether the observed frequencies in each category differ from expected frequencies.
        # - Ranges from 0 to infinity, with higher values indicating a greater difference between observed and expected frequencies.
        # - A p-value < 0.05 indicates a significant association between the two columns.        
        chi2, p, dof, expected = chi2_contingency(crosstab_raw)
        print(f"🧪 Chi2 Statistic: {chi2:.2f} (P-Value: {p:.4f})")

        # 🧪 Statistical Test 2: Cramér's V
        # - Measures the strength of association between two categorical variables.
        # - Ranges from 0 (no association) to 1 (perfect association).
        n = crosstab_raw.sum().sum()
        r, c = crosstab_raw.shape
        cramers_v = np.sqrt(chi2 / (n * min(r-1, c-1)))
        print(f"🧪 Cramér's V: {cramers_v:.2f}")

        # Evaluate statistical significance and effect size
        if p < 0.05:  # Check if the result is statistically significant
            # Assess the strength of the association using Cramér’s V
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
        
        # Define a consistent color palette for col2 categories
        unique_categories_col2 = df[col2].unique()
        color_palette = sns.color_palette("inferno", len(unique_categories_col2))
        color_map = dict(zip(unique_categories_col2, color_palette))
        
        # Create a figure with four subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # 📊 Plot 1: Countplot
        # - Visualizes the frequency of each category in col1, colored by col2.
        # - Useful for understanding the distribution of col1 across different categories of col2.
        sns.countplot(
            data=df, 
            x=col1, 
            hue=col2, 
            palette=color_map,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Countplot with Hue", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("")
        axes[0, 0].set_ylabel("")
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)
        axes[0, 0].legend(title=col2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 📊 Plot 2: Treemap
        # - Visualizes the proportions of categories in col1 and col2 using a treemap.
        # - Useful for understanding the hierarchical structure of categorical data.
        # Calculate proportions for the treemap
        proportions = df.groupby([col1, col2], observed=True).size().reset_index(name='counts')
        proportions['proportion'] = proportions['counts'] / proportions['counts'].sum()
        # Plot treemap
        axes[0, 1].axis('off') 
        squarify.plot(
            sizes=proportions['proportion'], 
            label=proportions.apply(lambda x: f"{x[col1]}-{x[col2]}", axis=1), 
            color=[color_map[cat] for cat in proportions[col2]],
            alpha=0.8, 
            text_kwargs={'fontsize': 10},
            ax=axes[0, 1]
        )
        axes[0, 1].set_title(f"Treemap", fontsize=12, fontweight='bold')
        
        # 📊 Plot 3: Heatmap
        # - Visualizes the proportions of categories in col1 and col2 using a heatmap.
        # - Useful for understanding the relationship between two categorical variables.
        # - The color intensity indicates the proportion of each category in col1 for each category in col2.
        # Create a custom annotation matrix with raw counts and percentages
        annotations = (
            crosstab_raw.applymap(lambda x: f"{x:,}")  # Format raw counts with thousands separators
            + "\n("
            + (crosstab_proportions * 100).round(1).applymap(lambda x: f"{x}%")  # Format percentages
            + ")"
        )
        # Create the heatmap with proportions and overlay raw counts + percentages
        sns.heatmap(
            crosstab_proportions, 
            annot=annotations,     # Overlay raw counts and percentages
            fmt="",                # No additional formatting since annotations are already strings
            cmap="flare",          # Color map for proportions
            cbar=False,            # Disable color bar for cleaner visualization
            annot_kws={"size": 10}, # Adjust font size of annotations
            ax=axes[1, 0]
        )
        axes[1, 0].set_title(f"Heatmap with Raw Counts and Percentages", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("")
        axes[1, 0].set_ylabel("")
        axes[1, 0].tick_params(axis='both', labelsize=10)
        
        # 📊 Plot 4: Stacked Bar Chart
        # - Visualizes the proportions of categories in col1 and col2 using a stacked bar chart.
        # - Useful for comparing the distribution of col1 across different categories of col2.
        # - Each bar represents a category in col1, and the segments represent the proportions of col2 categories within it.
        crosstab_proportions.plot(
            kind='bar', 
            stacked=True, 
            color=[color_map[cat] for cat in crosstab_proportions.columns],
            ax=axes[1, 1], 
            width=0.8
        )
        axes[1, 1].set_title(f"Stacked Bar Chart", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Proportion", fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1, 1].legend(title=col2, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout for compactness
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        fig.suptitle(f"Graphical Analysis of '{col1}' vs. '{col2}'", fontsize=14, fontweight='bold')
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
        # - Calculates summary statistics (mean, median, std) for each category in the categorical column.
        grouped_data = pair_df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std']).reset_index()
        print("📑 Summary Statistics by Category:")
        display(grouped_data.style.format({'mean': '{:.2f}', 'median': '{:.2f}', 'std': '{:.2f}'}))

        # 🧪 Statistical Test 1: Point-biserial correlation if binary categorical variable
        # - Measures the strength and direction of association between a binary categorical variable and a continuous variable.
        # - Ranges from -1 to 1, with 0 indicating no correlation.
        # - A p-value < 0.05 indicates a significant correlation.
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
        # - Tests whether the means of the numerical variable are significantly different across the categories of the categorical variable.
        # - A p-value < 0.05 indicates significant differences in means across categories.
        # - F-statistic indicates the ratio of variance between groups to variance within groups.
        # - - A higher F-statistic indicates a greater difference between group means.
        else:
            groups = [group[num_col].values for name, group in pair_df.groupby(cat_col)]
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
        # - Visualizes the distribution of the numerical variable across different categories of the categorical variable.
        # - Useful for identifying outliers and understanding the spread of data.
        # - The box represents the interquartile range (IQR), and the line inside the box represents the median.
        # - Whiskers extend to 1.5 times the IQR, and points outside this range are considered outliers.
        sns.boxplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            palette='inferno', 
            ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Box Plot: '{num_col}' Across '{cat_col}'", fontsize=12, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 📊 Plot 2: Violin Plot
        # - Combines box plot and density plot to visualize the distribution of the numerical variable across categories.
        # - Useful for understanding the shape of the distribution and identifying multimodal distributions.
        # - The width of the violin indicates the density of data points at different values.
        # - The white dot represents the median, and the thick bar in the center represents the interquartile range (IQR).
        sns.violinplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            palette='inferno', 
            ax=axes[0, 1]
        )
        axes[0, 1].set_title(f"Violin Plot: '{num_col}' Across '{cat_col}'", fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 📊 Plot 3: Bar Chart of Mean Values
        # - Visualizes the mean values of the numerical variable for each category in the categorical variable.
        # - Useful for comparing average values across categories.
        sns.barplot(
            data=grouped_data,
            x=cat_col,
            y='mean',
            capsize=0.1,
            palette='inferno', 
            ax=axes[1, 0]
        )
        axes[1, 0].set_title(f"Bar Chart: Mean '{num_col}' by '{cat_col}'", fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel("Mean Value", fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 📊 Plot 4: Scatter Plot with Jitter
        # - Visualizes the individual data points of the numerical variable across categories of the categorical variable.
        # - Useful for identifying patterns and distributions within categories.
        # - Jitter is added to reduce overplotting and make individual points more visible.
        sns.stripplot(
            data=pair_df,
            x=cat_col,
            y=num_col,
            palette='inferno', 
            jitter=True,
            alpha=0.6,
            ax=axes[1, 1]
        )
        axes[1, 1].set_title(f"Scatter Plot with Jitter: '{num_col}' by '{cat_col}'", fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Adjust layout for compactness
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"Graphical Analysis of '{cat_col}' vs. '{num_col}'", fontsize=14, fontweight='bold')
        plt.show()

    ###########################################################################################################################################
    ###########################################################################################################################################
    # Case 3: Both columns are numerical
    elif col1_type == "numerical" and col2_type == "numerical":
        # Correlation Metrics

        # 🧪 Statistical Test 1: Pearson Correlation
        # - Measures the linear correlation between two continuous variables.
        # - Ranges from -1 to 1, with 0 indicating no correlation.
        # - A p-value < 0.05 indicates a significant correlation.
        # - A positive correlation indicates that as one variable increases, the other tends to increase as well, vice versa.
        pearson_corr, pearson_p = pearsonr(pair_df[col1], pair_df[col2])
        print(f"🧪 Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_p:.4f}")
        
        # Evaluate statistical significance and effect size for Pearson Correlation
        if pearson_p < 0.05:  # Check if the correlation is statistically significant
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
        # - Measures the monotonic relationship between two continuous variables.
        # - Monotonic means that the relationship is not necessarily linear, but it can be either increasing or decreasing.
        # - Useful for non-linear relationships or when the data is not normally distributed.
        # - Ranges from -1 to 1, with 0 indicating no correlation.
        # - A p-value < 0.05 indicates a significant correlation.
        # - A positive correlation indicates that as one variable increases, the other tends to increase as well, vice versa.
        spearman_corr, spearman_p = spearmanr(pair_df[col1], pair_df[col2])
        print(f"\n🧪 Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p:.4f}")
        
        # Evaluate statistical significance and effect size for Spearman Correlation
        if spearman_p < 0.05:  # Check if the correlation is statistically significant
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

        # Create a figure with subplots (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 📊 Plot 1: Scatter Plot with Regression Line
        # - Visualizes the relationship between the two numerical variables.
        # - A regression line is fitted to the data points to show the trend.
        # - Useful for identifying linear relationships and outliers.
        sns.regplot(
            data=pair_df,
            x=col1,
            y=col2,
            scatter_kws={'alpha': 0.6, 'color': 'blue'},
            line_kws={'color': 'red'},
            ax=axes[0, 0]
        )
        axes[0, 0].set_title(f"Scatter Plot with Regression Line", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(col1, fontsize=10)
        axes[0, 0].set_ylabel(col2, fontsize=10)

        # 📊 Plot 2: Hexbin Plot
        # - Visualizes the density of points in a hexagonal grid.
        # - Useful for identifying clusters and patterns in large datasets.
        # - The color intensity indicates the number of points in each hexagon.
        sns.histplot(
            data=pair_df,
            x=col1,
            y=col2,
            bins=30,
            pthresh=0.1,
            cmap="Blues",
            cbar=True,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title(f"Hexbin Plot: Density of Points", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel(col1, fontsize=10)
        axes[0, 1].set_ylabel(col2, fontsize=10)

        # 📊 Plot 3: Heatmap of Correlation Matrix
        # - Visualizes the correlation matrix of the two numerical variables.
        # - Useful for understanding the strength and direction of the relationship.
        # - The color intensity indicates the strength of the correlation.
        # - The correlation coefficient ranges from -1 to 1, with 0 indicating no correlation.
        # - A positive correlation indicates that as one variable increases, the other tends to increase as well, vice versa.
        sns.heatmap(
            pair_df[[col1, col2]].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=False,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title(f"Heatmap of Correlation Matrix", fontsize=12, fontweight='bold')

        # 📊 Plot 4: Residual Plot
        # Fit a simple linear regression model
        # - Visualizes residuals (differences between observed and predicted values).
        # - Checks for randomness in residuals; non-random patterns may indicate model inadequacy.
        # - Assesses homoscedasticity (constant variance) and normality of residuals.
        # - Helps validate linear regression assumptions during EDA.
        model = LinearRegression()
        model.fit(pair_df[[col1]], pair_df[col2])
        predicted = model.predict(pair_df[[col1]])
        residuals = pair_df[col2] - predicted
        sns.scatterplot(x=predicted, y=residuals, ax=axes[1, 1], color="green", alpha=0.6)
        axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_title(f"Residual Plot", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Predicted Values", fontsize=10)
        axes[1, 1].set_ylabel("Residuals", fontsize=10)

        # Adjust layout for compactness
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f"Graphical Analysis of '{col1}' vs. '{col2}'", fontsize=14, fontweight='bold')
        plt.show()