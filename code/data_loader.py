"""
Data loading and preprocessing module for fMRI PTSD dataset.
"""

import pandas as pd
from typing import Tuple
from config import DATA_FILE, NUMERIC_COLUMNS


def load_raw_data() -> pd.DataFrame:
    """Load raw data from Excel file."""
    return pd.read_excel(DATA_FILE)


def get_paired_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract paired data for clinical group (before/after intervention).
    
    Returns:
        DataFrame with Subject ID, PCL_before, PCL_after
    """
    clinical = df[df['Experimental group'] == 'Clinical'].copy()
    
    before = clinical[clinical['Condition'] == 'Before intervention'][
        ['Subject ID', 'PCL score']
    ].dropna().rename(columns={'PCL score': 'PCL_before'})
    
    after = clinical[clinical['Condition'] == 'After intervention'][
        ['Subject ID', 'PCL score']
    ].dropna().rename(columns={'PCL score': 'PCL_after'})
    
    return before.merge(after, on='Subject ID')


def get_anova_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for 2x2 ANOVA (Gender × Time on Amygdala BOLD).
    
    Returns:
        DataFrame with Gender, Time, AmygBOLD
    """
    anova_df = df[['Gender', 'Condition', 'Amyg (BOLD signal)']].dropna().copy()
    anova_df = anova_df.rename(columns={'Amyg (BOLD signal)': 'AmygBOLD'})
    anova_df['Time'] = anova_df['Condition'].map({
        'Before intervention': 'Before',
        'After intervention': 'After'
    })
    anova_df['Gender'] = anova_df['Gender'].str.lower()
    return anova_df[['Gender', 'Time', 'AmygBOLD']]


def get_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric columns for correlation analysis."""
    return df[NUMERIC_COLUMNS].dropna()