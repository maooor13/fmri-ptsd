"""
Statistical Analysis - Question 1: Intervention Effectiveness on PCL Score.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def analyze_pcl_intervention(paired_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze intervention effectiveness using paired t-test.
    
    Args:
        paired_df: DataFrame with PCL_before and PCL_after columns
        
    Returns:
        Dictionary containing statistical results
    """
    n = len(paired_df)
    before = paired_df['PCL_before']
    after = paired_df['PCL_after']
    diff = before - after
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(before, after)
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = diff.mean() / diff.std()
    
    # Descriptive statistics
    results = {
        'n': n,
        'mean_before': before.mean(),
        'mean_after': after.mean(),
        'sd_before': before.std(),
        'sd_after': after.std(),
        'mean_diff': diff.mean(),
        'sd_diff': diff.std(),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'is_significant': p_value < 0.05
    }
    
    return results


def check_assumptions(paired_df: pd.DataFrame) -> Dict[str, float]:
    """
    Check normality assumptions for paired t-test.
    
    Returns:
        Dictionary with Shapiro-Wilk p-values for each condition
    """
    _, p_before = stats.shapiro(paired_df['PCL_before'])
    _, p_after = stats.shapiro(paired_df['PCL_after'])
    
    return {
        'shapiro_before': p_before,
        'shapiro_after': p_after
    }


def get_sig_stars(p: float) -> str:
    """Return asterisk notation for significance."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def format_results(results: Dict[str, Any]) -> str:
    """Format results as readable string."""
    p_str = f"{results['p_value']:.6f}{get_sig_stars(results['p_value'])}"
    return f"""
Question 1: Intervention Effectiveness on PCL Score
{'='*55}
Sample: n = {results['n']} PTSD patients

Descriptive Statistics:
  Before: M = {results['mean_before']:.2f}, SD = {results['sd_before']:.2f}
  After:  M = {results['mean_after']:.2f}, SD = {results['sd_after']:.2f}
  Difference: M = {results['mean_diff']:.2f}, SD = {results['sd_diff']:.2f}

Paired t-test: t({results['n']-1}) = {results['t_stat']:.3f}, p = {p_str}
Effect Size: Cohen's d = {results['cohens_d']:.3f}{get_sig_stars(results['p_value']) if results['is_significant'] else ''}

Conclusion: {'SIGNIFICANT' if results['is_significant'] else 'NOT SIGNIFICANT'}
  The intervention {'is' if results['is_significant'] else 'is not'} effective in reducing PTSD symptoms.
"""


if __name__ == "__main__":
    from data_loader import load_raw_data, get_paired_clinical_data
    
    df = load_raw_data()
    paired_df = get_paired_clinical_data(df)
    results = analyze_pcl_intervention(paired_df)
    assumptions = check_assumptions(paired_df)
    
    print(format_results(results))
    print(f"Normality (Shapiro-Wilk): before p={assumptions['shapiro_before']:.4f}, after p={assumptions['shapiro_after']:.4f}")