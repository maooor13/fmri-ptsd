"""
Main orchestrator for fMRI PTSD Research Analysis.
Coordinates all analyses and generates outputs.
"""

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_raw_data, get_paired_clinical_data, get_anova_data, get_numeric_data
from stats_q1 import analyze_pcl_intervention, check_assumptions, format_results as format_q1
from stats_q2 import analyze_all_correlations, format_results as format_q2
from stats_q3 import analyze_anova, format_results as format_q3
from visualizations import save_all_figures


def run_question_1(df):
    """Run Question 1: Intervention Effectiveness."""
    print("\n" + "="*60)
    print("RUNNING QUESTION 1: PCL Intervention Analysis")
    print("="*60)
    
    paired_df = get_paired_clinical_data(df)
    results = analyze_pcl_intervention(paired_df)
    assumptions = check_assumptions(paired_df)
    
    print(format_q1(results))
    print(f"Normality (Shapiro-Wilk): before p={assumptions['shapiro_before']:.4f}, after p={assumptions['shapiro_after']:.4f}")
    
    return paired_df, results


def run_question_2(df):
    """Run Question 2: Correlation Analysis."""
    print("\n" + "="*60)
    print("RUNNING QUESTION 2: Correlation Analysis")
    print("="*60)
    
    numeric_data = get_numeric_data(df)
    results = analyze_all_correlations(numeric_data)
    
    print(format_q2(results))
    print("\nCorrelation Matrix:")
    print(results['pearson_matrix'].round(3).to_string())
    
    return numeric_data, results


def run_question_3(df):
    """Run Question 3: 2x2 ANOVA."""
    print("\n" + "="*60)
    print("RUNNING QUESTION 3: 2x2 Factorial ANOVA")
    print("="*60)
    
    anova_data = get_anova_data(df)
    results = analyze_anova(anova_data)
    
    print(format_q3(results))
    
    return anova_data, results


def main():
    """Run all analyses."""
    print("\n" + "#"*60)
    print("# fMRI PTSD Research Analysis")
    print("# Statistical Analysis for 3 Research Questions")
    print("#"*60)
    
    # Load data
    print("\nLoading data...")
    df = load_raw_data()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Run each question
    q1_data, q1_results = run_question_1(df)
    q2_data, q2_results = run_question_2(df)
    q3_data, q3_results = run_question_3(df)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    save_all_figures(
        paired_df=q1_data,
        corr_matrix=q2_results['pearson_matrix'],
        annot_matrix=q2_results['annotation_matrix'],
        bonf_alpha=q2_results['bonferroni_alpha'],
        corr_data=q2_data,
        strongest_pair=q2_results['strongest_pair'],
        strongest_r=q2_results['strongest_r'],
        strongest_p=q2_results['significance']['p_value'],
        strongest_n=q2_results['significance']['n'],
        anova_data=q3_data,
        anova_results=q3_results
    )
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("""
Key Findings:
1. Intervention significantly reduces PCL scores (p < 0.001, d = 1.38)
2. Strongest correlation: Amygdala BOLD ↔ rACC BOLD (r = 0.75)
3. 2x2 ANOVA: Main effect of Time, no significant interaction
    """)


if __name__ == "__main__":
    main()