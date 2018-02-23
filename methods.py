# -*- coding: utf-8 -*-

__all__ = ["transit_periodogram_fast"]

import numpy as np
from functools import partial

from .transit_periodogram_impl import transit_periodogram_impl, transit_periodogram_impl_all


def transit_periodogram_slow(t, y, ivar, period, duration,
                             oversample, use_likelihood, pool):
    """Compute the periodogram using a brute force reference method

    t : array-like
        Sequence of observation times.
    y : array-like
        Sequence of observations associated with times t.
    ivar : array-like
        The inverse variance of ``y``.
    period : array-like
        The trial periods where the periodogram should be computed.
    duration : array-like
        The durations that should be tested.
    oversample :
        The resolution of the phase grid in units of durations.
    use_likeliood : bool
        If true, maximize the log likelihood over phase, duration, and depth.
    pool :
        If provided, this should be an object with a ``map`` method that will
        be used to parallelize the computation.

    Returns
    -------
    power : array-like
        The periodogram evaluated at the periods in ``period``.
    depth : array-like
        The estimated depth of the maximum power model at each period.
    depth_err : array-like
        The 1-sigma uncertainty on ``depth``.
    duration : array-like
        The maximum power duration at each period.
    transit_time : array-like
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like
        The log likelihood of the maximum power model.

    """
    f = partial(_transit_periodogram_slow_one, t, y, ivar, duration,
                oversample, use_likelihood)
    return _apply(f, pool, period)


def transit_periodogram_fast(t, y, ivar, period, duration, oversample,
                             use_likelihood, pool):
    """Compute the periodogram using an optimized Cython implementation

    t : array-like
        Sequence of observation times.
    y : array-like
        Sequence of observations associated with times t.
    ivar : array-like
        The inverse variance of ``y``.
    period : array-like
        The trial periods where the periodogram should be computed.
    duration : array-like
        The durations that should be tested.
    oversample :
        The resolution of the phase grid in units of durations.
    use_likeliood : bool
        If true, maximize the log likelihood over phase, duration, and depth.
    pool :
        If provided, this should be an object with a ``map`` method that will
        be used to parallelize the computation.

    Returns
    -------
    power : array-like
        The periodogram evaluated at the periods in ``period``.
    depth : array-like
        The estimated depth of the maximum power model at each period.
    depth_err : array-like
        The 1-sigma uncertainty on ``depth``.
    duration : array-like
        The maximum power duration at each period.
    transit_time : array-like
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like
        The log likelihood of the maximum power model.

    """
    return transit_periodogram_impl_all(
        t, y, ivar, period, duration, oversample, use_likelihood
    )

    # Pre-compute some factors that are used in every loop
    sum_y2 = np.sum(y * y * ivar)
    sum_y = np.sum(y * ivar)
    sum_ivar = np.sum(ivar)

    # Convert the durations to bin counts
    bin_duration = np.min(duration) / oversample
    duration_int = np.asarray(np.round(duration / bin_duration),
                              dtype=np.int64)

    # These constants are needed by the Cython code, but we don't compute
    # them there so that we can release the GIL.
    eps = np.finfo(np.float64).eps
    ninf = -np.inf

    # Construct the function that evaluates the result for a single period.
    f = partial(
        transit_periodogram_impl,
        t, y, ivar, duration_int, bin_duration,
        sum_y2, sum_y, sum_ivar, eps, ninf, oversample,
        use_likelihood)

    # Apply this function, possibly in parallel.
    return _apply(f, pool, period)


def _transit_periodogram_slow_one(t, y, ivar, duration, oversample,
                                  use_likelihood, period):
    """A private function to compute the brute force periodogram result"""
    best = (-np.inf, None)
    hp = 0.5*period
    for dur in duration:

        # Compute the phase grid (this is set by the duration and oversample).
        d_phase = dur / oversample
        phase = np.arange(0, period+d_phase, d_phase)

        for t0 in phase:
            # Figure out which data points are in and out of transit.
            m_in = np.abs((t-t0+hp) % period - hp) < 0.5*dur
            m_out = ~m_in

            # Compute the estimates of the in and out-of-transit flux.
            ivar_in = np.sum(ivar[m_in])
            ivar_out = np.sum(ivar[m_out])
            y_in = np.sum(y[m_in] * ivar[m_in]) / ivar_in
            y_out = np.sum(y[m_out] * ivar[m_out]) / ivar_out

            # Use this to compute the best fit depth and uncertainty.
            depth = y_out - y_in
            depth_err = np.sqrt(1.0 / ivar_in + 1.0 / ivar_out)
            snr = depth / depth_err

            # Compute the log likelihood of this model.
            chi2 = np.sum((y_in - y[m_in])**2 * ivar[m_in])
            chi2 += np.sum((y_out - y[m_out])**2 * ivar[m_out])
            loglike = -0.5*chi2

            # Choose which objective should be used for the optimization.
            if use_likelihood:
                objective = loglike
            else:
                objective = snr

            # If this model is better than any before, keep it.
            if objective > best[0]:
                best = (
                    objective,
                    (objective, depth, depth_err, dur, t0, snr, loglike)
                )

    return best[1]


def _apply(f, pool, period):
    if pool is None:
        mapper = map
    else:
        mapper = pool.map
    return tuple(map(np.array, zip(*mapper(f, period))))
