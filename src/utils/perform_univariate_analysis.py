# src/utils/perform_univariate_analysis.py

from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore

def perform_univariate_analysis(df, column_name, show_graphs=True, show_distribution=True, bins=30):
    """
    Perform comprehensive univariate analysis on a specific column in the DataFrame.

    This function provides both non-graphical and graphical insights into the distribution and characteristics of a single column. It supports categorical and numerical columns, offering descriptive statistics, missing value analysis, and visualizations (if enabled).

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to analyze.
        column_name (str): The name of the column to analyze. Must exist in the DataFrame.
        show_graphs (bool, optional): Whether to display graphical visualizations. Defaults to True.
        show_distribution (bool, optional): Whether to include distribution-related statistics and plots for numerical columns. Defaults to True.
        bins (int, optional): Number of bins for the histogram (applies to numerical columns). Defaults to 30.

    Returns:
        None: Results are printed and displayed interactively.
    """
    # Non-Graphical Analysis
    print(f"🔢 Data Type: {df[column_name].dtype}")
    print(f"💎 Number of Unique Values: {df[column_name].nunique()}")
    print(f"📋 List of Unique Categories: {df[column_name].unique().tolist()}")
    if show_distribution:
        value_counts = df[column_name].value_counts(normalize=True).to_frame(name='Proportion')
        value_counts['Count'] = df[column_name].value_counts()
        print("📊 Value Distribution:")
        display(value_counts.style.format({
            'Count': '{:,}',  # Add thousand separators
            'Proportion': '{:.2%}'  # Format as percentage with 2 decimal places
        }))
    if df[column_name].dtype not in ['object', 'category']:
        summary_stats = df[column_name].describe().to_frame().T
        skewness = df[column_name].skew()
        kurtosis = df[column_name].kurtosis()

        # Define thresholds for skewness and kurtosis
        skewness_threshold = 1.0  # Values > |1.0| indicate moderate skewness
        kurtosis_threshold = 3.0  # Values > 3.0 indicate heavy tails (potential outliers)

        print("📊 Summary Statistics:")
        display(summary_stats.style.format('{:.2f}'))    
        print("📈 Data Distribution: ")
        skewness_warning = "⚠️" if abs(skewness) > skewness_threshold else ""
        kurtosis_warning = "⚠️" if kurtosis > kurtosis_threshold else ""
        print(f"   └── Skewness: {skewness:.2f} {skewness_warning}")
        print(f"   └── Kurtosis: {kurtosis:.2f} {kurtosis_warning}")

        # Determine outlier detection method based on skewness and kurtosis
        if abs(skewness) > 1 or abs(kurtosis - 3) > 2:  # Non-normal data
            print("\n🔍 Outlier Detection Method: Using IQR (Interquartile Range)")
            print("   └── Reason: High skewness or kurtosis suggests non-normal distribution.")
            
            # Outlier Detection Using IQR
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR
            outliers_iqr = df[(df[column_name] < lower_bound_iqr) | (df[column_name] > upper_bound_iqr)]
            num_outliers_iqr = len(outliers_iqr)
            total_rows = len(df)
            outlier_percentage_iqr = (num_outliers_iqr / total_rows) * 100

            if num_outliers_iqr == 0:
                print("   └── ✅ No outliers detected using the IQR method.")
            else:
                print(f"   └── ⚠️ {num_outliers_iqr} outliers detected ({outlier_percentage_iqr:.2f}% of total rows).")
                print(f"   └── Lower Bound: {lower_bound_iqr:.2f}")
                print(f"   └── Upper Bound: {upper_bound_iqr:.2f}")
                print("   └── Outliers Summary:")
                display(outliers_iqr[[column_name]].describe().style.format('{:.2f}'))

        else:  # Normal data
            print("\n🔍 Outlier Detection Method: Using Z-Score")
            print("   └── Reason: Low skewness and kurtosis suggest approximately normal distribution.")
            
            # Outlier Detection Using Z-Scores
            z_scores = zscore(df[column_name].dropna())
            outliers_z = df[abs(z_scores) > 3]
            num_outliers_z = len(outliers_z)
            outlier_percentage_z = (num_outliers_z / len(df)) * 100
            outliers_list = outliers_z[column_name].to_list()
            formatted_list = ', '.join(f"{value:.2f}" for value in outliers_list)

            if num_outliers_z == 0:
                print("   └── ✅ No outliers detected using the Z-Score method.")
            else:
                print(f"   └── ⚠️ {num_outliers_z} outliers detected ({outlier_percentage_z:.2f}% of total rows).")
                print(f"   └── Outliers List: {formatted_list}")

    # Check for missing values
    missing_count = df[column_name].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100  # Calculate percentage of missing rows
    if missing_count == 0:
        print("\n✅ No missing values found.")
    else:
        print(f"\n⚠️ Rows with missing values: {missing_count} ({missing_percentage:.2f}% of total rows)")
        missing_rows = df[df[column_name].isnull()]
        display(missing_rows)

    # Graphical Analysis (Optional)
    if show_graphs:
        if df[column_name].dtype in ['object', 'category']:
            # Categorical Column Visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
            fig.suptitle(f"Graphical Analysis of '{column_name}'", fontsize=14, fontweight='bold')

            # Countplot with categories sorted by frequency
            value_counts = df[column_name].value_counts()
            sorted_categories = value_counts.index
            sns.countplot(data=df, x=column_name, ax=axes[0], order=sorted_categories)
            axes[0].set_title("Countplot", fontsize=14, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_xlabel("") 
            axes[0].set_ylabel("") 

            # Pie Chart with Top 4 Categories + "Others"
            if len(value_counts) > 5:
                top_4 = value_counts[:4]  # Get the top 4 categories
                others = pd.Series(value_counts[4:].sum(), index=["Others"])  # Combine the rest into "Others"
                pie_data = pd.concat([top_4, others])  # Combine top 4 and "Others"
            else:
                pie_data = value_counts  # If there are 5 or fewer categories, use all of them

            wedges, texts, autotexts = axes[1].pie(
                pie_data, 
                labels=pie_data.index,  # Display category names as labels
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100 * pie_data.sum())})',  # Format: percentage and count
                startangle=90, 
                colors=sns.color_palette('tab20c'),
                textprops={'fontsize': 10}  # Adjust font size of annotations
            )

            axes[1].set_title("Pie Chart", fontsize=14, fontweight='bold')
            axes[1].set_ylabel("") 

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        else:
            # Numerical Column Visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Univariate Analysis of '{column_name}'", fontsize=18, fontweight='bold')

            # 1. Histogram with KDE
            sns.histplot(data=df, x=column_name, kde=True, bins=bins, color='teal', ax=axes[0, 0])
            axes[0, 0].set_title("Histogram with KDE", fontsize=14, fontweight='bold')

            # 2. Boxplot
            sns.boxplot(data=df, y=column_name, color='lightcoral', ax=axes[0, 1])
            axes[0, 1].set_title("Boxplot", fontsize=14, fontweight='bold')

            # 3. Violin Plot
            sns.violinplot(data=df, y=column_name, color='cornflowerblue', ax=axes[1, 0])
            axes[1, 0].set_title("Violin Plot", fontsize=14, fontweight='bold')

            # 4. QQ Plot
            sm.qqplot(df[column_name].dropna(), line='s', ax=axes[1, 1], markerfacecolor='gold', markeredgecolor='black')
            axes[1, 1].set_title("QQ Plot", fontsize=14, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()