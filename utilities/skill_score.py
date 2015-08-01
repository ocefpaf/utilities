from __future__ import division, absolute_import

import numpy as np
import numpy.ma as ma
from pandas import DataFrame
from pandas.tseries.frequencies import to_offset

try:
    from sklearn.metrics import (mean_absolute_error,
                                 mean_squared_error,
                                 median_absolute_error,
                                 r2_score,
                                 explained_variance_score)
except ImportError:
    pass

# TODO: Taylor, SST

__all__ = ['both_valid',
           'pearsonr_paired',
           'explained_variance_score',
           'median_absolute_error',
           'mean_absolute_error',
           'r2_score',
           'filter_series']


def rmse(obs, model):
    """
    Compute root mean square between the observed data (`obs`) and the modeled
    data (`model`).
    >>> obs = [3, -0.5, 2, 7]
    >>> model = [2.5, 0.0, 2, 8]
    >>> rmse(obs, model)
    0.61237243569579447
    >>> obs = [[0.5, 1],[-1, 1],[7, -6]]
    >>> model = [[0, 2],[-1, 2],[8, -5]]
    >>> rmse(obs, model)
    0.84162541153017323

    """
    return np.sqrt(mean_squared_error(obs, model))


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


def filter_series(series, window_size=193, T=40, freq=None):
    """
    This function performs a lanczos filter on a pandas time-series and
    returns the low and high pass data in a dataframe.

    series : Pandas series
    window_size : Size of the filter windows (default is 96+1+96).
    T : Period of the filter. (The default of 40 hours filter should
        get rid of all tides.)
    freq : Pandas offset object or string for the resampling frequency
          (default is no resampling).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pandas import Series, date_range
    >>> N = 700
    >>> noise = np.random.randn(N)
    >>> low_noise = 4 * np.sin(np.pi * np.arange(N) / 360.) + noise
    >>> out = 2 * np.sin(2 * np.pi * np.arange(N) / 12.42) + low_noise
    >>> t = date_range(start='1980-01-19', periods=N, freq='1D')
    >>> out = Series(out, index=t)
    >>> kw = dict(window_size=193, T=40)
    >>> df = filter_series(out, **kw)
    >>> fig, ax = plt.subplots(figsize=(9, 2.75))
    >>> ax = df['low'].plot(ax=ax, label='low')
    >>> ax = df['high'].plot(ax=ax, label='high')
    >>> ax = out.plot(ax=ax, label='original')
    >>> leg = ax.legend()

    """
    from oceans import lanc

    T *= 60*60.  # To seconds.

    if freq:
        freq = to_offset(freq)
        series = series.resample(freq)
        T /= freq.delta.total_seconds()
    freq = 1./T

    series = series - series.mean()
    mask = np.isnan(series)
    series.interpolate(inplace=True)

    wt = lanc(window_size, freq)
    low = np.convolve(wt, series, mode='same')
    low = ma.masked_array(low, mask)
    high = series - low
    data = dict(low=low, high=high)
    return DataFrame(data, index=series.index)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
