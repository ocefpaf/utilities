from __future__ import division, absolute_import

import numpy as np


# TODO: Taylor, SST

__all__ = ['both_valid',
           'pearsonr_paired']


def both_valid(x, y):
    """
    Returns a mask where both series are valid.

    Examples
    --------
    >>> import numpy as np
    >>> x = [np.NaN, 1, 2, 3, 4, 5]
    >>> y = [0, 1, np.NaN, 3, 4, 5]
    >>> both_valid(x, y)
    array([False,  True, False,  True,  True,  True], dtype=bool)

    """
    mask_x = np.isnan(x)
    mask_y = np.isnan(y)
    return np.logical_and(~mask_x, ~mask_y)


def pearsonr_paired(x, y):
    """
    Scipy pearsonr for matching series.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([np.NaN, 1, 2, 3, 4, 5])
    >>> y = np.array([0, 1, np.NaN, 3, 4, 5])
    >>> pearsonr_paired(x, y)
    (1.0, 0.0)

    """
    from scipy.stats.stats import pearsonr

    mask = both_valid(x, y)
    x, y = x[mask], y[mask]
    r, p = pearsonr(x-x.mean(), y-y.mean())
    return r, p


if __name__ == '__main__':
    import doctest
    doctest.testmod()
