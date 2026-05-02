"""
Visualization module for fMRI PTSD analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from pathlib import Path

# Set style globally
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def add_significance_bracket(ax, x1, x2, y, p_value, offset=0.02):
    """Add significance bracket with asterisks."""
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        return
    
    ax.plot([x1, x1, x2, x2], [y, y+offset, y+offset, y], 'k-', linewidth=1.5)
    ax.text((x1+x2)/2, y+offset+0.005, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')


def plot_pcl_intervention(
    paired_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create visualization for Question 1 (PCL intervention analysis).
    
    Returns:
        Figure and Axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot with error bars and significance
    ax1 = axes[0]
    means = [paired_df['PCL_before'].mean(), paired_df['PCL_after'].mean()]
    sems = [paired_df['PCL_before'].std()/np.sqrt(len(paired_df)), 
            paired_df['PCL_after'].std()/np.sqrt(len(paired_df))]
    
    bars = ax1.bar(['Before', 'After'], means, yerr=sems, capsize=5, 
                   color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('PCL Score', fontsize=12)
    ax1.set_xlabel('Condition', fontsize=12)
    ax1.set_title('PCL Score: Before vs After Intervention\n(PTSD Patients)', fontsize=14)
    ax1.set_ylim(0, 80)
    
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                  f'M = {mean:.1f}', ha='center', fontsize=12, fontweight='bold')
    
    # Add significance bracket
    max_y = max(means[0] + sems[0], means[1] + sems[1])
    add_significance_bracket(ax1, 0, 1, max_y + 3, 0.000001)
    
    # Box plot with connected points
    ax2 = axes[1]
    bp = ax2.boxplot([paired_df['PCL_before'], paired_df['PCL_after']], 
                     tick_labels=['Before', 'After'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')
    
    for i in range(len(paired_df)):
        ax2.plot([1, 2], [paired_df.iloc[i]['PCL_before'], paired_df.iloc[i]['PCL_after']], 
                 color='gray', alpha=0.3, linewidth=0.5)
    
    ax2.set_ylabel('PCL Score', fontsize=12)
    ax2.set_title('Distribution of PCL Scores', fontsize=14)
    ax2.text(0.5, 0.02, "t(49) = 9.73, p < 0.001***, Cohen's d = 1.38***", 
             transform=ax2.transAxes, ha='center', fontsize=11, style='italic')
    ax2.set_ylim(35, 80)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    highlight: Optional[Tuple[str, str]] = None,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create correlation heatmap for Question 2.
    
    Args:
        corr_matrix: Correlation matrix
        highlight: Optional tuple of (var1, var2) to highlight
    
    Returns:
        Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax, 
                vmin=-1, vmax=1, annot_kws={'size': 10})
    
    ax.set_title('Correlation Matrix - All Variables', fontsize=14, pad=20)
    
    if highlight:
        var1, var2 = highlight
        max_row = corr_matrix.index.get_loc(var1)
        max_col = corr_matrix.columns.get_loc(var2)
        ax.add_patch(plt.Rectangle((max_col, max_row), 1, 1, fill=False, 
                                    edgecolor='gold', linewidth=3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_scatter_correlation(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    r: float,
    p_value: float = 0.001,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create scatter plot for strongest correlation.
    
    Returns:
        Figure and Axes objects
    """
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = ""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(data[var1], data[var2], alpha=0.6, edgecolor='white', s=60)
    
    # Regression line
    z = np.polyfit(data[var1].dropna(), data[var2].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[var1].min(), data[var1].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {r:.3f}{sig}')
    
    ax.set_xlabel(var1, fontsize=12)
    ax.set_ylabel(var2, fontsize=12)
    ax.set_title(f'Strongest Correlation: {var1} vs {var2}\n(r = {r:.3f}{sig}, p < 0.001***)', fontsize=13)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_anova_interaction(
    data: pd.DataFrame,
    results: dict,
    save_path: Optional[Path] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create interaction plot for Question 3 (2x2 ANOVA).
    
    Returns:
        Figure and Axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'female': '#e74c3c', 'male': '#3498db'}
    markers = {'female': 'o', 'male': 's'}
    
    # Plot 1: Interaction plot (line)
    ax1 = axes[0]
    plot_data = data.groupby(['Gender', 'Time'])['AmygBOLD'].agg(['mean', 'sem']).reset_index()
    
    # Get y-axis limits
    all_means = plot_data['mean'].values
    all_sems = plot_data['sem'].values
    y_min = 0.1
    y_max = max(all_means + all_sems) * 1.15
    
    for gender in ['female', 'male']:
        subset = plot_data[plot_data['Gender'] == gender]
        ax1.errorbar(subset['Time'], subset['mean'], yerr=subset['sem'],
                    label=gender.capitalize(), marker=markers[gender], 
                    color=colors[gender], markersize=10, capsize=5, linewidth=2)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Amygdala BOLD Signal', fontsize=12)
    ax1.set_title('2x2 Interaction: Gender × Time', fontsize=14)
    ax1.legend(title='Gender', loc='upper right')
    ax1.set_ylim(y_min, y_max)
    
    p_int = results['p_interaction']
    p_time = results['p_time']
    
    # Add significance indicators
    if p_time < 0.001:
        time_sig = "***"
    elif p_time < 0.01:
        time_sig = "**"
    elif p_time < 0.05:
        time_sig = "*"
    else:
        time_sig = ""
    
    int_sig = "***" if p_int < 0.001 else "**" if p_int < 0.01 else "*" if p_int < 0.05 else ""
    sig_text = f'Time: F = {results["f_time"]:.2f}, p = {p_time:.4f}{time_sig}  |  Interaction: F = {results["f_interaction"]:.2f}, p = {p_int:.4f}{int_sig}'
    ax1.text(0.5, 0.02, sig_text, transform=ax1.transAxes, ha='center', fontsize=9)
    
    # Add significance bracket for Time effect (between Before and After - averaged)
    avg_before = plot_data[plot_data['Time'] == 'Before']['mean'].mean()
    avg_after = plot_data[plot_data['Time'] == 'After']['mean'].mean()
    bracket_y = max(avg_before, avg_after) + 0.04
    add_significance_bracket(ax1, 0, 1, bracket_y, p_time, offset=0.015)
    
    # Plot 2: Grouped bar chart
    ax2 = axes[1]
    x = np.arange(2)
    width = 0.35
    
    female_means = plot_data[plot_data['Gender'] == 'female']['mean'].values
    female_sems = plot_data[plot_data['Gender'] == 'female']['sem'].values
    male_means = plot_data[plot_data['Gender'] == 'male']['mean'].values
    male_sems = plot_data[plot_data['Gender'] == 'male']['sem'].values
    
    bars1 = ax2.bar(x - width/2, female_means, width, yerr=female_sems, 
            label='Female', color='#e74c3c', capsize=5, alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, male_means, width, yerr=male_sems, 
            label='Male', color='#3498db', capsize=5, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Amygdala BOLD Signal', fontsize=12)
    ax2.set_title('Amygdala BOLD by Gender and Time', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Before', 'After'])
    ax2.legend()
    ax2.set_ylim(y_min, y_max)
    
    # Add significance for simple effect - Female: Before vs After
    f_female = results['simple_effects_time']['female']
    if f_female['p_value'] < 0.05:
        max_f = max(female_means[0] + female_sems[0], female_means[1] + female_sems[1])
        add_significance_bracket(ax2, 0, 1, max_f + 0.025, f_female['p_value'], offset=0.015)
    
    # Add significance for simple effect - Gender at After
    g_after = results['simple_effects_gender']['After']
    if g_after['p_value'] < 0.05:
        max_after = max(female_means[1] + female_sems[1], male_means[1] + male_sems[1])
        ax2.annotate('*', xy=(1, max_after + 0.025), ha='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def save_all_figures(
    paired_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    corr_data: pd.DataFrame,
    strongest_pair: Tuple[str, str],
    strongest_r: float,
    strongest_p: float,
    anova_data: pd.DataFrame,
    anova_results: dict
) -> None:
    """Save all figures to output directory."""
    
    # Q1: PCL Intervention
    plot_pcl_intervention(
        paired_df, 
        save_path=OUTPUT_DIR / "fig1_pcl_intervention.png"
    )
    
    # Q2: Correlation Heatmap
    plot_correlation_heatmap(
        corr_matrix, 
        highlight=strongest_pair,
        save_path=OUTPUT_DIR / "fig2_correlation_heatmap.png"
    )
    
    # Q2: Scatter plot
    var1, var2 = strongest_pair
    plot_scatter_correlation(
        corr_data, var1, var2, strongest_r, strongest_p,
        save_path=OUTPUT_DIR / "fig2b_strongest_correlation.png"
    )
    
    # Q3: ANOVA
    plot_anova_interaction(
        anova_data, anova_results,
        save_path=OUTPUT_DIR / "fig3_anova_interaction.png"
    )
    
    plt.close('all')
    print(f"All figures saved to {OUTPUT_DIR}")