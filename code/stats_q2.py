"""
Statistical Analysis - Question 2: Correlation Analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Any, Optional

NUM_COMPARISONS = 28
BONFERRONI_ALPHA = 0.05 / NUM_COMPARISONS


def compute_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    return data.corr(method='pearson')


def compute_spearman_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation matrix for numeric columns."""
    return data.corr(method='spearman')


def compute_p_value_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Compute p-value matrix for all pairwise correlations."""
    cols = data.columns
    p_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i < j:
                valid = data[[c1, c2]].dropna()
                if len(valid) > 2:
                    _, p = stats.pearsonr(valid[c1], valid[c2])
                    p_matrix.loc[c1, c2] = p
                    p_matrix.loc[c2, c1] = p
    return p_matrix


def compute_annotation_matrix(corr_matrix: pd.DataFrame, p_matrix: pd.DataFrame) -> pd.DataFrame:
    """Build annotation matrix with r values and Bonferroni-aware asterisks."""
    cols = corr_matrix.columns
    annot = pd.DataFrame("", index=cols, columns=cols)
    for c1 in cols:
        for c2 in cols:
            r = corr_matrix.loc[c1, c2]
            p = p_matrix.loc[c1, c2]
            if c1 == c2:
                annot.loc[c1, c2] = "—"
            else:
                if p < BONFERRONI_ALPHA:
                    stars = "†"
                elif p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                else:
                    stars = ""
                annot.loc[c1, c2] = f"{r:.2f}{stars}"
    return annot


def find_strongest_correlation(corr_matrix: pd.DataFrame) -> Tuple[str, str, float]:
    """
    Find the pair of variables with strongest absolute correlation.
    
    Returns:
        Tuple of (var1, var2, correlation_value)
    """
    # Mask diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask).stack()
    
    max_idx = corr_values.abs().idxmax()
    return max_idx[0], max_idx[1], corr_values[max_idx]


def test_correlation_significance(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """
    Test significance of correlation between two variables.
    
    Returns:
        Dictionary with r, p_value, n, r_squared
    """
    # Remove NaN pairs
    valid_mask = ~(x.isna() | y.isna())
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    r, p = stats.pearsonr(x_clean, y_clean)
    
    return {
        'r': r,
        'p_value': p,
        'n': len(x_clean),
        'r_squared': r ** 2,
        'is_significant': p < 0.05
    }


def compute_partial_correlation(
    x: pd.Series, 
    y: pd.Series, 
    control: pd.Series
) -> Dict[str, float]:
    """
    Compute partial correlation controlling for a third variable.
    
    Returns:
        Dictionary with partial_r and p_value
    """
    # Remove rows with NaN
    valid_mask = ~(x.isna() | y.isna() | control.isna())
    x_clean = x[valid_mask].values
    y_clean = y[valid_mask].values
    ctrl_clean = control[valid_mask].values
    
    # Residualize
    x_resid = x_clean - np.polyval(np.polyfit(ctrl_clean, x_clean, 1), ctrl_clean)
    y_resid = y_clean - np.polyval(np.polyfit(ctrl_clean, y_clean, 1), ctrl_clean)
    
    partial_r, partial_p = stats.pearsonr(x_resid, y_resid)
    
    return {
        'partial_r': partial_r,
        'p_value': partial_p
    }


def analyze_all_correlations(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete correlation analysis.
    
    Returns:
        Dictionary with correlation matrices and strongest pair results
    """
    pearson_matrix = compute_correlation_matrix(data)
    spearman_matrix = compute_spearman_matrix(data)
    p_matrix = compute_p_value_matrix(data)
    annot_matrix = compute_annotation_matrix(pearson_matrix, p_matrix)
    
    var1, var2, r = find_strongest_correlation(pearson_matrix)
    sig_test = test_correlation_significance(data[var1], data[var2])
    partial = compute_partial_correlation(data[var1], data[var2], data['Age'])
    
    return {
        'pearson_matrix': pearson_matrix,
        'spearman_matrix': spearman_matrix,
        'p_value_matrix': p_matrix,
        'annotation_matrix': annot_matrix,
        'bonferroni_alpha': BONFERRONI_ALPHA,
        'num_comparisons': NUM_COMPARISONS,
        'strongest_pair': (var1, var2),
        'strongest_r': r,
        'significance': sig_test,
        'partial_correlation': partial
    }


def get_sig_stars(p: float, bonf_alpha: Optional[float] = None) -> str:
    """Return asterisk notation for significance.

    If bonf_alpha is provided, '†' is used for Bonferroni-surviving results.
    """
    if bonf_alpha is not None and p < bonf_alpha:
        return "†"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def format_results(results: Dict[str, Any]) -> str:
    """Format results as readable string."""
    var1, var2 = results['strongest_pair']
    sig = results['significance']
    partial = results['partial_correlation']
    bonf_alpha = results['bonferroni_alpha']
    n_comp = results['num_comparisons']
    p_sig = get_sig_stars(sig['p_value'], bonf_alpha)
    partial_p_sig = get_sig_stars(partial['p_value'], bonf_alpha)
    
    return f"""
Question 2: Correlation Analysis - Strongest Correlation
{'='*55}
Bonferroni correction: {n_comp} comparisons, adjusted α = {bonf_alpha:.6f}

Variables: {var1} ↔ {var2}

Pearson Correlation:
  r = {results['strongest_r']:.3f}{p_sig}
  r² = {sig['r_squared']:.3f} ({sig['r_squared']*100:.1f}% variance explained)
  n = {sig['n']}, p = {sig['p_value']:.6f}{p_sig}
  {'✅ Survives Bonferroni correction' if sig['p_value'] < bonf_alpha else '❌ Does NOT survive Bonferroni correction'}

Partial Correlation (controlling for Age):
  partial r = {partial['partial_r']:.3f}{partial_p_sig}, p = {partial['p_value']:.4f}{partial_p_sig}

Conclusion: STRONGEST correlation between {var1} and {var2}
"""


if __name__ == "__main__":
    from data_loader import load_raw_data, get_numeric_data
    
    df = load_raw_data()
    data = get_numeric_data(df)
    results = analyze_all_correlations(data)
    
    print(format_results(results))
    print("\nCorrelation Matrix:")
    print(results['pearson_matrix'].round(3).to_string())