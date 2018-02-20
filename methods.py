# -*- coding: utf-8 -*-

__all__ = ["transit_periodogram_fast"]

import numpy as np
from functools import partial

from .transit_periodogram_impl import transit_periodogram_impl


def transit_periodogram_fast(time, flux, flux_ivar, period, duration,
                             oversample, use_likelihood, pool):
    # Pre-compute some factors that are used in every loop
    sum_flux2 = np.sum(flux * flux * flux_ivar)
    sum_flux = np.sum(flux * flux_ivar)
    sum_ivar = np.sum(flux_ivar)

    # Convert the durations to bin counts
    bin_duration = np.min(duration) / oversample
    duration_int = np.asarray(np.round(duration / bin_duration),
                              dtype=np.int64)

    # A few constants
    eps = np.finfo(np.float64).eps
    ninf = -np.inf

    f = partial(
        transit_periodogram_impl,
        time, flux, flux_ivar, duration_int, bin_duration,
        sum_flux2, sum_flux, sum_ivar, eps, ninf, oversample,
        use_likelihood)

    if pool is None:
        mapper = map
    else:
        mapper = pool.map

    return tuple(map(np.array, zip(*mapper(f, period))))
