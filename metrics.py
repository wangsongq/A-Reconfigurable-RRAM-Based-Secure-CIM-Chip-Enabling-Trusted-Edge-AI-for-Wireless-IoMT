"""Evaluation metrics used across training, testing, and figure generation."""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp2d


SEG_CSV_PATH = Path(__file__).resolve().parent / "seg.csv"

# numpy metrics
def surveillance_error(targets, predictions):
    """Compute mean and std of SEG scores using the bundled lookup table."""
    data = np.loadtxt(SEG_CSV_PATH)

    xs = np.linspace(0, 600, data.shape[0])
    ys = np.linspace(0, 600, data.shape[1])
    f = interp2d(xs, ys, np.transpose(data))

    scores = np.concatenate([f(t, p) for (t, p) in zip(targets, predictions)])
    return np.mean(scores), np.std(scores)

def root_mean_squared_error(targets, predictions):
    """Compute RMSE in the same units as the provided targets."""
    return np.sqrt(np.mean(np.power(targets-predictions, 2)))
