import numpy as np
import numpy.ma as ma
from pandas.tseries.frequencies import to_offset
from pandas import DatetimeIndex, Series, rolling_median


__all__ = ['is_monotonically_increasing',
           'has_time_gaps',
           'threshold',
           'filter_spikes',
           'tukey53H']


def is_monotonically_increasing(times):
    """
    Check if a given list or array of datetime-like objects is
    monotonically increasing.

    Examples
    --------
    >>> from pandas import date_range
    >>> times = date_range('1980-01-19', periods=10)
    >>> is_monotonically_increasing(times)
    True
    >>> import numpy as np
    >>> is_monotonically_increasing(np.r_[times[-2:-1], times])
    False
    """
    return all(x < y for x, y in zip(times, times[1:]))


def has_time_gaps(times, freq):
    """
    Check for gaps in a series time-stamp. The `freq` can be a string or a
    pandas offset object.  Note the freq cannot be an ambiguous offset like
    week, months, etc.

    Example
    -------
    >>> import numpy as np
    >>> from pandas import date_range
    >>> times = date_range('1980-01-19', periods=48, freq='1H')
    >>> has_time_gaps(times, freq='6min')
    True
    >>> has_time_gaps(times, freq='1H')
    False

    """
    freq = to_offset(freq)
    if hasattr(freq, 'delta'):
        times = DatetimeIndex(times)
    else:
        raise ValueError('Cannot interpret freq {!r} delta'.format(freq))
    return (np.diff(times) > freq.delta.to_timedelta64()).any()


def threshold(series, vmin=None, vmax=None):
    """
    Threshold an series by flagging with NaN values below `vmin` and above
    `vmax`.

    Examples
    --------
    >>> series = [0.1, 20, 35.5, 34.9, 43.5, 34.6, 33.7]
    >>> threshold(series, vmin=30, vmax=40)
    masked_array(data = [-- -- 35.5 34.9 -- 34.6 33.7],
                 mask = [ True  True False False  True False False],
           fill_value = 1e+20)
    <BLANKLINE>
    >>> from pandas import Series, date_range
    >>> series = Series(series, index=date_range('1980-01-19', periods=7))
    >>> threshold(series, vmin=30, vmax=40)
    1980-01-19     NaN
    1980-01-20     NaN
    1980-01-21    35.5
    1980-01-22    34.9
    1980-01-23     NaN
    1980-01-24    34.6
    1980-01-25    33.7
    Freq: D, dtype: float64

    """
    if not vmin:
        vmin = min(series)
    if not vmax:
        vmax = max(series)

    masked = ma.masked_outside(series, vmin, vmax)
    if masked.mask.any():
        if isinstance(series, Series):
            series[masked.mask] = np.NaN
            return series
    return masked


def filter_spikes(series, window_size=3, threshold=3):
    """
    Filter an array-like object using a median filter and a `threshold`
    for the median difference.

    Examples
    --------
    >>> from pandas import Series, date_range
    >>> series = [33.43, 33.45, 34.45, 90.0, 35.67, 34.9, 43.5, 34.6, 33.7]
    >>> series = Series(series, index=date_range('1980-01-19',
    ...                 periods=len(series)))
    >>> filter_spikes(series, window_size=3, threshold=3)
    1980-01-19    33.43
    1980-01-20    33.45
    1980-01-21    34.45
    1980-01-22      NaN
    1980-01-23    35.67
    1980-01-24    34.90
    1980-01-25    43.50
    1980-01-26    34.60
    1980-01-27    33.70
    Freq: D, dtype: float64

    """
    medians = rolling_median(series, window=window_size, center=True)
    medians = medians.fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(series - medians)
    ndiff = difference / difference.std()
    outlier_idx = ndiff > threshold
    series[outlier_idx] = np.NaN
    return series


def _high_pass(data, alpha=0.5):
    """
    Runs a high pass RC filter over the given data.

    Parameters
    ----------
    data : array_like

    alpha : float
            Smoothing factor between 0.0 (exclusive) and 1.0 (inclusive).
            A lower value means more filtering.  A value of 1.0 equals no
            filtering. Defaults is 0.5.

    Returns
    -------
    hpf : array_like
          Filtered data.

    Based on http://en.wikipedia.org/wiki/High-pass_filter

    """
    mean = data.mean()
    data = data - mean
    hpf = data.copy()
    for k in range(1, len(data)):
        hpf[k] = alpha * hpf[k-1] + alpha * (data[k] - data[k-1])
    return hpf + mean


def tukey53H(data, k=1.5):
    """
    Flags suspicious spikes values using Tukey 53H.

    References
    ----------
    .. [1] Goring, Derek G., and Vladimir I. Nikora. "Despiking acoustic
    Doppler velocimeter data." Journal of Hydraulic Engineering 128.1 (2002):
    117-126.  http://dx.doi.org/10.1061/(ASCE)0733-9429(2002)128:1(117)

    Examples
    --------
    >>> from pandas import Series, date_range
    >>> series = [33.43, 33.45, 34.45, 90.0, 35.67, 34.9, 43.5, 34.6, 33.7]
    >>> series = Series(series, index=date_range('1980-01-19',
    ...                 periods=len(series)))
    >>> tukey53H(series.values, k=1.5)
    array([False, False, False,  True, False, False, False, False, False], \
dtype=bool)

    """
    data = np.asanyarray(data)

    data = _high_pass(data, 0.99)
    data = data - data.mean()

    N = len(data)
    stddev = data.std()

    u1 = np.zeros_like(data)
    for n in range(N-4):
        if data[n:n+5].any():
            u1[n+2] = np.median(data[n:n+5])

    u2 = np.zeros_like(data)
    for n in range(N-2):
        if u1[n:n+3].any():
            u2[n+1] = np.median(u1[n:n+3])

    u3 = np.zeros_like(data)
    u3[1:-1] = 0.25*(u2[:-2] + 2*u2[1:-1] + u2[2:])

    delta = np.abs(data-u3)

    return delta > k*stddev

if __name__ == '__main__':
    import doctest
    doctest.testmod()
