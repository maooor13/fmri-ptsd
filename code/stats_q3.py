"""
Statistical Analysis - Question 3: 2x2 Factorial ANOVA (Gender × Time).
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from typing import Dict, Any


def run_factorial_anova(data: pd.DataFrame, dv: str, factor1: str, factor2: str) -> pd.DataFrame:
    """
    Run 2x2 factorial ANOVA.
    
    Args:
        data: DataFrame with columns
        dv: Dependent variable name
        factor1: First factor name (between-subjects)
        factor2: Second factor name (within-subjects)
        
    Returns:
        ANOVA results table
    """
    formula = f'{dv} ~ C({factor1}) * C({factor2})'
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, typ=2)
    
    return anova_table


def compute_effect_sizes(anova_table: pd.DataFrame) -> Dict[str, float]:
    """
    Compute partial eta-squared (η²p) for each effect.
    η²p = SS_effect / (SS_effect + SS_residual)
    
    Returns:
        Dictionary of effect sizes
    """
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    
    effect_sizes = {}
    for effect in anova_table.index:
        if effect != 'Residual':
            ss_effect = anova_table.loc[effect, 'sum_sq']
            effect_sizes[effect] = ss_effect / (ss_effect + ss_residual)
    
    return effect_sizes


def simple_effects_time_within_gender(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Simple effects: Time within each Gender.
    
    Returns:
        Dictionary with t, p, d for each gender
    """
    results = {}
    
    for gender in ['female', 'male']:
        subset = data[data['Gender'] == gender]
        before = subset[subset['Time'] == 'Before']['AmygBOLD']
        after = subset[subset['Time'] == 'After']['AmygBOLD']
        
        t, p = stats.ttest_ind(before, after)
        pooled_std = np.sqrt((before.std()**2 + after.std()**2) / 2)
        d = (before.mean() - after.mean()) / pooled_std
        
        results[gender] = {
            't': t,
            'p_value': p,
            'cohens_d': d,
            'mean_before': before.mean(),
            'mean_after': after.mean()
        }
    
    return results


def simple_effects_gender_at_time(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Simple effects: Gender at each Time point.
    
    Returns:
        Dictionary with t, p, d for each time point
    """
    results = {}
    
    for time in ['Before', 'After']:
        subset = data[data['Time'] == time]
        female = subset[subset['Gender'] == 'female']['AmygBOLD']
        male = subset[subset['Gender'] == 'male']['AmygBOLD']
        
        t, p = stats.ttest_ind(female, male)
        pooled_std = np.sqrt((female.std()**2 + male.std()**2) / 2)
        d = (female.mean() - male.mean()) / pooled_std
        
        results[time] = {
            't': t,
            'p_value': p,
            'cohens_d': d,
            'mean_female': female.mean(),
            'mean_male': male.mean()
        }
    
    return results


def get_descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Get descriptive statistics for each cell."""
    return data.groupby(['Gender', 'Time'])['AmygBOLD'].agg(['mean', 'std', 'count'])


def analyze_anova(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete 2x2 ANOVA analysis.
    
    Returns:
        Dictionary with all results
    """
    anova_table = run_factorial_anova(data, 'AmygBOLD', 'Gender', 'Time')
    effect_sizes = compute_effect_sizes(anova_table)
    desc_stats = get_descriptive_stats(data)
    
    # Extract key statistics
    f_gender = anova_table.loc['C(Gender)', 'F']
    p_gender = anova_table.loc['C(Gender)', 'PR(>F)']
    f_time = anova_table.loc['C(Time)', 'F']
    p_time = anova_table.loc['C(Time)', 'PR(>F)']
    f_interaction = anova_table.loc['C(Gender):C(Time)', 'F']
    p_interaction = anova_table.loc['C(Gender):C(Time)', 'PR(>F)']
    
    simple_time = simple_effects_time_within_gender(data)
    simple_gender = simple_effects_gender_at_time(data)
    
    return {
        'anova_table': anova_table,
        'effect_sizes': effect_sizes,
        'descriptive_stats': desc_stats,
        'f_gender': f_gender,
        'p_gender': p_gender,
        'f_time': f_time,
        'p_time': p_time,
        'f_interaction': f_interaction,
        'p_interaction': p_interaction,
        'simple_effects_time': simple_time,
        'simple_effects_gender': simple_gender
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
    p_gender_sig = get_sig_stars(results['p_gender'])
    p_time_sig = get_sig_stars(results['p_time'])
    p_interaction_sig = get_sig_stars(results['p_interaction'])
    
    f_female = results['simple_effects_time']['female']
    f_male = results['simple_effects_time']['male']
    g_before = results['simple_effects_gender']['Before']
    g_after = results['simple_effects_gender']['After']
    
    return f"""
Question 3: 2x2 ANOVA - Gender × Time on Amygdala BOLD
{'='*55}

Descriptive Statistics (Mean ± SD):
{results['descriptive_stats'].round(4).to_string()}

ANOVA Results:
  Main Effect (Gender):     F = {results['f_gender']:.3f}, p = {results['p_gender']:.4f}{p_gender_sig}, η²p = {results['effect_sizes']['C(Gender)']:.3f}
  Main Effect (Time):       F = {results['f_time']:.3f}, p = {results['p_time']:.4f}{p_time_sig}, η²p = {results['effect_sizes']['C(Time)']:.3f}
  Interaction (Gender×Time): F = {results['f_interaction']:.3f}, p = {results['p_interaction']:.4f}{p_interaction_sig}, η²p = {results['effect_sizes']['C(Gender):C(Time)']:.3f}

Simple Effects - Time within each Gender:
  Female: t = {f_female['t']:.3f}, p = {f_female['p_value']:.4f}{get_sig_stars(f_female['p_value'])}, d = {f_female['cohens_d']:.3f}
  Male:   t = {f_male['t']:.3f}, p = {f_male['p_value']:.4f}{get_sig_stars(f_male['p_value'])}, d = {f_male['cohens_d']:.3f}

Simple Effects - Gender at each Time:
  Before: t = {g_before['t']:.3f}, p = {g_before['p_value']:.4f}{get_sig_stars(g_before['p_value'])}, d = {g_before['cohens_d']:.3f}
  After:  t = {g_after['t']:.3f}, p = {g_after['p_value']:.4f}{get_sig_stars(g_after['p_value'])}, d = {g_after['cohens_d']:.3f}

Conclusion: {'Significant interaction' if results['p_interaction'] < 0.05 else 'No significant interaction'} - {'Different pattern between genders' if results['p_interaction'] < 0.05 else 'Similar pattern across genders'}
"""


if __name__ == "__main__":
    from data_loader import load_raw_data, get_anova_data
    
    df = load_raw_data()
    data = get_anova_data(df)
    results = analyze_anova(data)
    
    print(format_results(results))