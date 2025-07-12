import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore
from IPython.display import display

def perform_univariate_analysis(
    df, feature,
    show_plots=True,
    top_n_pie=5, bins=30, skew_thresh=1.0, kurt_thresh=3.0, high_card_threshold=25, rare_threshold=0.01
):
    """
    Perform a comprehensive univariate analysis of a specified feature (numerical or categorical).

    For numerical features:
    - Prints data type, unique values, summary stats, skewness, and kurtosis.
    - Detects outliers using IQR or Z-score method depending on distribution.
    - Visualizes with histogram + KDE, boxplot, violin plot, and QQ plot.

    For categorical features:
    - Prints unique value counts and frequency table with percentages.
    - Flags constant, high-cardinality, and rare categories.
    - Checks for formatting inconsistencies (e.g., whitespace, capitalization).
    - Visualizes with countplot and pie chart (grouping small categories into "Others").

    Parameters:
        df (pd.DataFrame): The input dataset.
        feature (str): The column name to analyze.
        show_plots (bool): Whether to display plots (default: True).
        top_n_pie (int): Number of top categories to show in pie chart before grouping others (default: 5).
        bins (int): Number of bins for histogram (default: 30).
        skew_thresh (float): Threshold for flagging skewness as high (default: 1.0).
        kurt_thresh (float): Threshold for flagging excess kurtosis as high (default: 3.0).
        high_card_threshold (int): Threshold to flag a feature as high-cardinality (default: 25 unique values).
        rare_threshold (float): Minimum proportion to consider a category as non-rare (default: 0.01 = 1%).

    Returns:
        None. Displays printed summaries and visual plots.
    """
    print(f"🔎 Performing univariate analysis:")
    col_type = "categorical" if df[feature].dtype in ['object', 'category'] else "numerical"
    print(f" └── Column '{feature}' (Type: {col_type})\n")

    # 1. Check existence and data type
    if feature not in df.columns:
        print("❌ Feature not found in the DataFrame.")
        return

    # If data is numeric type
    if pd.api.types.is_numeric_dtype(df[feature]):

        # 2. Data type
        dtype = df[feature].dtype
        print(f"📘 Data Type: {dtype}")

        # 3. Unique non-NA values
        unique_values = df[feature].dropna().unique()
        print(f"💎 Unique Non-NA Values: {len(unique_values)}")

        # 2. Summary statistics
        stats = df[feature].describe()
        print("📊 Summary Statistics:")
        display(stats.to_frame().T.style.format("{:.2f}"))

        # 3. Skewness and Kurtosis
        skew = df[feature].skew()
        kurt = df[feature].kurtosis()
        print("📈 Distribution Shape:")
        print(f"   └── Skewness: {skew:.2f} {'⚠️' if abs(skew) > skew_thresh else ''}")
        print(f"   └── Kurtosis (excess): {kurt:.2f} {'⚠️' if kurt > kurt_thresh else ''}")

        # 4. Outlier detection
        if abs(skew) > skew_thresh or abs(kurt) > 2:
            print("\n🔍 Outlier Detection: IQR Method (due to skew/kurtosis)")
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        else:
            print("\n🔍 Outlier Detection: Z-Score Method")
            z_scores = zscore(df[feature].dropna())
            outliers = df[abs(z_scores) > 3]

        num_outliers = len(outliers)
        outlier_pct = (num_outliers / len(df)) * 100
        if num_outliers == 0:
            print("   └── ✅ No outliers detected.")
        else:
            print(f"   └── ⚠️ {num_outliers} outliers found ({outlier_pct:.2f}% of rows)")
            print("   └── Outliers summary:")
            display(outliers[[feature]].describe().T.style.format("{:.2f}"))

        # 5. Missing values
        missing_count = df[feature].isnull().sum()
        total_rows = len(df)
        missing_pct = (missing_count / total_rows) * 100
        if missing_count == 0:
            print("✅ No rows with missing values found.")
        else:
            print(f"⚠️ Rows with missing values: {missing_count} / {total_rows} ({missing_pct:.2f}%)")
            display(df[df[feature].isnull()].head())

        # 6. Graphs
        if show_plots:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Graphical Analysis of '{feature}'", fontsize=18, fontweight='bold', color='#333333')

            sns.histplot(df[feature], bins=bins, kde=True, ax=axes[0, 0], color="#4C72B0")
            axes[0, 0].set_title("Histogram + KDE", fontsize=14, fontweight='bold', color='#444444')

            sns.boxplot(y=df[feature], ax=axes[0, 1], color='#C44E52')
            axes[0, 1].set_title("Boxplot", fontsize=14, fontweight='bold', color='#444444')

            sns.violinplot(y=df[feature], ax=axes[1, 0], color='#55A868')
            axes[1, 0].set_title("Violin Plot", fontsize=14, fontweight='bold', color='#444444')

            sm.qqplot(df[feature].dropna(), line='s', ax=axes[1, 1], markerfacecolor='#FFA500', markeredgecolor='black')
            axes[1, 1].set_title("QQ Plot", fontsize=14, fontweight='bold', color='#444444')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    # If data is categorical type
    elif pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == 'object':

        # 2. Data type
        dtype = df[feature].dtype
        print(f"📘 Data Type: {dtype}")

        # 3. Unique values
        unique_values = df[feature].dropna().unique()
        print(f"💎 Unique Non-NA Values: {len(unique_values)}")
        print(f"📋 List of Unique Non-NA Values: {df[feature].unique().tolist()}")

        # 4. Frequency table
        counts = df[feature].value_counts()
        percentages = df[feature].value_counts(normalize=True) * 100
        freq_table = pd.DataFrame({
            'Count': counts.apply(lambda x: f"{x:,}"),
            'Percentage (%)': percentages.round(2)
        })
        freq_table.index.name = None
        print("📊 Frequency Table:")
        display(freq_table)

        # 5. Constant / High Cardinality
        if len(unique_values) == 1:
            print(f"⚠️ Constant Value: Feature is constant with value: {unique_values[0]}")
        elif len(unique_values) > high_card_threshold:
            print(f"⚠️ High Cardinality (>{high_card_threshold}): {len(unique_values)} unique categories")

        # 6. Rare categories
        rare_cats = percentages[percentages < (rare_threshold * 100)]
        if not rare_cats.empty:
            print(f"⚠️ Rare Categories (<{rare_threshold*100:.0f}%): {len(rare_cats)} found")

        # 7. Whitespace issues
        stripped = df[feature].astype(str).str.strip()
        if not df[feature].astype(str).equals(stripped):
            print("⚠️ Detected leading/trailing whitespace in some entries.")

        # 8. Capitalization inconsistencies
        lowercase = df[feature].astype(str).str.lower()
        if len(lowercase.unique()) < len(df[feature].astype(str).unique()):
            print("⚠️ Potential inconsistent capitalization (e.g., 'USA' vs 'usa')")

        # 9. Missing values
        missing_count = df[feature].isnull().sum()
        total_rows = len(df)
        missing_pct = (missing_count / total_rows) * 100
        if missing_count == 0:
            print("✅ No rows with missing values found.")
        else:
            print(f"⚠️ Rows with missing values: {missing_count} / {total_rows} ({missing_pct:.2f}%)")
            display(df[df[feature].isnull()].head())

        # 10. Graphical Analysis
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
            fig.suptitle(f"Graphical Analysis of '{feature}'", fontsize=16, fontweight='bold', color='#333333')

            # Define consistent color palette mapping
            categories = counts.index.tolist()
            palette_colors = sns.color_palette("tab20c", n_colors=len(categories))
            color_mapping = dict(zip(categories, palette_colors))

            # 10.1. Countplot (uses mapped colors)
            sns.countplot(
                data=df, x=feature, ax=axes[0],
                order=categories,
                palette=color_mapping
            )
            axes[0].set_title("Countplot", fontsize=14, fontweight='bold', color='#444444')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_xlabel("")
            axes[0].set_ylabel("Frequency")
            axes[0].spines[['top', 'right']].set_visible(False)

            # 10.2. Pie Chart with top_n_pie + "Others"
            if len(counts) > top_n_pie:
                top_n = counts[:top_n_pie]
                others = pd.Series(counts[top_n_pie:].sum(), index=["Others"])
                pie_data = pd.concat([top_n, others])
                pie_colors = [color_mapping.get(cat, '#999999') for cat in pie_data.index]
            else:
                pie_data = counts
                pie_colors = [color_mapping[cat] for cat in pie_data.index]

            wedges, texts, autotexts = axes[1].pie(
                pie_data,
                labels=pie_data.index,
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100 * pie_data.sum())})',
                startangle=90,
                colors=pie_colors,
                textprops={'fontsize': 10}
            )
            axes[1].set_title("Pie Chart", fontsize=14, fontweight='bold', color='#444444')
            axes[1].set_ylabel("")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    else:
        print(f"❌ '{feature}' is neither numerical nor categorical. Data type: {df[feature].dtype}")
