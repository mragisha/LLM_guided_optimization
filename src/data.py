"""
Data loading utilities for MOOT config datasets.
Columns ending in '+' are to be maximized, '-' are to be minimized.
"""
import pandas as pd
import numpy as np
import os

MOOT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "moot", "optimize", "config")


def load_task(csv_path):
    """Load a MOOT CSV. Returns df, x_cols, y_cols, directions (+1=maximize, -1=minimize)."""
    df = pd.read_csv(csv_path)
    y_cols = [c for c in df.columns if c.endswith('+') or c.endswith('-')]
    x_cols = [c for c in df.columns if c not in y_cols]
    directions = [1 if c.endswith('+') else -1 for c in y_cols]
    return df, x_cols, y_cols, directions


def normalize_objectives(df, y_cols, directions):
    """Normalize objectives to [0,1], flipping so 0=best for all."""
    norm = pd.DataFrame()
    for col, d in zip(y_cols, directions):
        vals = df[col].values.astype(float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            norm[col] = np.zeros(len(vals))
        else:
            scaled = (vals - lo) / (hi - lo)
            norm[col] = scaled if d == -1 else 1 - scaled
    return norm


def chebyshev(norm_y_row):
    """Chebyshev distance from utopia point (all zeros). Lower is better."""
    return float(np.max(norm_y_row))


def compute_chebyshev(df, y_cols, directions):
    """Return array of Chebyshev distances for all rows."""
    norm = normalize_objectives(df, y_cols, directions)
    return norm.apply(chebyshev, axis=1).values


def get_all_tasks():
    """Return list of (name, path) for all config CSVs."""
    tasks = []
    for f in sorted(os.listdir(MOOT_CONFIG)):
        if f.endswith('.csv'):
            name = f.replace('.csv', '')
            path = os.path.join(MOOT_CONFIG, f)
            tasks.append((name, path))
    return tasks


def classify_dim(n_x):
    """Low/medium/high dimensionality per Di Fiore et al. split."""
    if n_x < 6:
        return 'low'
    elif n_x <= 11:
        return 'medium'
    else:
        return 'high'
