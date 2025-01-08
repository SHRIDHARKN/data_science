import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon

# compare the categorical distributions in two dataframes
# prefer jensen shannon for imabalanced class
def jensen_shannon_compare_dfs(df1,df2):
    p = df1['Count'] / df1['Count'].sum()
    q = df2['Count'] / df2['Count'].sum()
    js_divergence = jensenshannon(p, q)
    print(js_divergence)
    if js_divergence<0.1:
        print("Distributions are very similar")
    elif js_divergence<0.1:
        print("Moderate differences between distributions")
    else:
        print("Distributions are significantly different")

# compare the categorical distributions in two dataframes
# note : results may not be reliable in case of class imbalance
def chi_square_compare_dfs(df1,df2,category_col,p_value=0.05):

    merged_df = pd.merge(df1, df2, on=category_col, suffixes=('_1', '_2'))
    # Perform chi-square test
    count_cols = [x for x in merged_df.columns if x!=category_col]
    observed = merged_df[count_cols].values
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p_value}")
    if p_value<0.05:
        print("there is significant difference in the category dist of two dataframes")
    else:
        print("there is NO significant difference in the category dist of two dataframes")
    return chi2, p_value

def chi_square_test(df,count_col,category_col,p_value=0.05):
    
    observed = df[count_col].values
    total = np.sum(observed)
    num_cats = len(df[category_col].unique())
    expected = np.full(num_cats, total / num_cats)
    chi2, p_value = stats.chisquare(observed, expected)
    if p_value<0.05:
        print("there is significant difference in the category dist")
    else:
        print("there is NO significant difference in the category dist")

    return chi2, p_value
