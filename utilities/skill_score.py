from __future__ import division

import numpy as np
from scipy.stats.stats import pearsonr

# TODO: Taylor, SST

__all__ = ['both_valid',
           'pearsonr_paired']


def both_valid(x, y):
    """Returns a mask where both series are valid."""
    mask_x = np.isnan(x)
    mask_y = np.isnan(y)
    return np.logical_and(~mask_x, ~mask_y)


def pearsonr_paired(x, y):
    mask = both_valid(x, y)
    r, p = pearsonr(x[mask]-x.mean(), y[mask]-y.mean())
    return r, p
