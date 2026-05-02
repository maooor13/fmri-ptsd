"""
Configuration module - Constants and paths for fMRI PTSD analysis.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data files
DATA_FILE = DATA_DIR / "fMRI dataset.xlsx"

# Output files
OUTPUT_DIR.mkdir(exist_ok=True)

# Numeric columns for correlation analysis
NUMERIC_COLUMNS = [
    'Age',
    'Amyg (BOLD signal)',
    'rACC (BOLD signal)',
    'SN-CEN correlation',
    'SN-DMN correlation',
    'Amyg volume (mm^3)',
    'PCL score',
    'Stroop RT (sec)'
]

# Significance level
ALPHA = 0.05