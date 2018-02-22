import numpy as np
cimport numpy as np

cimport cython

from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

IDTYPE = np.int64
ctypedef np.int64_t IDTYPE_t


cdef double compute_log_like(
    double y_in,
    double y_out,
    double ivar_in,
    double sum_y2,
    double sum_y,
    double sum_ivar,
) nogil:
    cdef double arg = y_in - y_out
    cdef double chi2 = sum_y2 - 2*y_out*sum_y
    chi2 += y_out*y_out * sum_ivar
    chi2 -= arg*arg * ivar_in
    return -0.5*chi2


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fold(
    # Inputs
    int N,              # Length of the time array
    double* t,       # The list of timestamps
    double* y,       # The y measured at ``t``
    double* ivar,       # The inverse variance of the y array
    double sum_y2,   # The precomputed value of sum(y^2 * ivar)
    double sum_y,    # The precomputed value of sum(y * ivar)
    double sum_ivar,    # The precomputed value of sum(ivar)
    double ninf,        # Negative infinity for DTYPE
    double eps,         # Machine precision for DTYPE

    double period,      # The period to test in units of ``t``

    int n_durations,    # Length of the durations array
    int* durations,     # The durations to test in units of ``bin_duration``
    double bin_duration,# The width of the fine-grain bins to use in units of
                        # ``t``
    int oversample,     # The number of ``bin_duration`` bins in the maximum duration

    int use_likelihood, # A flag indicating the periodogram type
                        # 0 - depth signal-to-noise
                        # 1 - log likelihood

    # Outputs
    double* best_objective,  # The value of the periodogram at maximum
    double* best_depth,      # The estimated depth at maximum
    double* best_depth_std,  # The uncertainty on ``best_depth``
    double* best_duration,   # The best fitting duration in units of ``t``
    double* best_phase,      # The phase of the mid-transit time in units of
                             # ``t``
    double* best_depth_snr,  # The signal-to-noise ratio of the depth estimate
    double* best_log_like    # The log likelihood at maximum
) nogil:

    cdef int ind, n, k
    cdef double y_in, y_out, ivar_in, ivar_out, \
                depth, depth_std, depth_snr, log_like, objective

    cdef int n_bins = <int>(period / bin_duration) + oversample
    cdef double* mean_y = <double*>malloc((n_bins+1)*sizeof(double))
    if not mean_y:
        with gil:
            raise MemoryError()
    cdef double* mean_ivar = <double*>malloc((n_bins+1)*sizeof(double))
    if not mean_ivar:
        free(mean_y)
        with gil:
            raise MemoryError()

    # This first pass bins the data into a fine-grain grid in phase from zero
    # to period and computes the weighted sum and inverse variance for each
    # bin.
    for n in range(n_bins+1):
        mean_y[n] = 0.0
        mean_ivar[n] = 0.0
    for n in range(N):
        ind = <int>((t[n] % period) / bin_duration) + 1
        mean_y[ind] += y[n] * ivar[n]
        mean_ivar[ind] += ivar[n]

    # To simplify calculations below, we wrap the binned values around and pad
    # the end of the array with the first ``oversample`` samples.
    for n in range(1, oversample+1):
        ind = n_bins-oversample+n
        mean_y[ind] = mean_y[n]
        mean_ivar[ind] = mean_ivar[n]

    # To compute the estimates of the in-transit flux, we need the sum of
    # mean_y and mean_ivar over a given set of transit points. To get this
    # fast, we can compute the cumulative sum and then use differences between
    # points separated by ``duration`` bins. Here we convert the mean arrays
    # to cumulative sums.
    for n in range(1, n_bins+1):
        mean_y[n] += mean_y[n-1]
        mean_ivar[n] += mean_ivar[n-1]

    # Then we loop over phases (in steps of n_bin) and durations and find the
    # best fit value. By looping over durations here, we get to reuse a lot of
    # the computations that we did above.
    best_objective[0] = ninf
    for k in range(n_durations):
        for n in range(n_bins-durations[k]+1):
            # Estimate the in-transit and out-of-transit flux
            y_in = mean_y[n+durations[k]] - mean_y[n]
            ivar_in = mean_ivar[n+durations[k]] - mean_ivar[n]
            y_out = sum_y - y_in
            ivar_out = sum_ivar - ivar_in

            # Skip this model if there are no points in transit
            if ivar_in < eps or ivar_out < eps:
                continue

            # Normalize to compute the actual value of the flux
            y_in /= ivar_in
            y_out /= ivar_out

            # Either compute the log likelihood or the signal-to-noise ratio
            if use_likelihood:
                objective = compute_log_like(y_in, y_out, ivar_in,
                                             sum_y2, sum_y, sum_ivar)
            else:
                depth = y_out - y_in
                depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out)
                objective = depth / depth_std

            # If this is the best result seen so far, keep it
            if objective > best_objective[0]:
                if use_likelihood:
                    depth = y_out - y_in
                    depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out)
                    depth_snr = depth / depth_std
                    log_like = objective
                else:
                    log_like = compute_log_like(y_in, y_out, ivar_in,
                                          sum_y2, sum_y,
                                          sum_ivar)
                    depth_snr = objective

                # Save the current model
                best_objective[0] = objective
                best_depth[0] = depth
                best_depth_std[0] = depth_std
                best_depth_snr[0] = depth_snr
                best_log_like[0] = log_like
                best_duration[0] = durations[k] * bin_duration
                best_phase[0] = (n*bin_duration +
                                 0.5*durations[k]*bin_duration) % period

    # Clean up the temporary memory
    free(mean_y)
    free(mean_ivar)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def transit_periodogram_impl(
        np.ndarray[DTYPE_t, mode='c'] t_array,
        np.ndarray[DTYPE_t, mode='c'] y_array,
        np.ndarray[DTYPE_t, mode='c'] ivar_array,
        np.ndarray[IDTYPE_t, mode='c'] duration_int_array,

        double bin_duration,
        double sum_y2,
        double sum_y,
        double sum_ivar,

        double eps,
        double ninf,

        int oversample,
        int use_likelihood,
        double period,
):

    cdef double* t = <double*>t_array.data
    cdef double* y = <double*>y_array.data
    cdef double* ivar = <double*>ivar_array.data
    cdef int* durations = <int*>duration_int_array.data
    cdef int N = len(t_array)
    cdef int n_durations = len(duration_int_array)

    cdef double out_objective = 0.0, out_depth = 0.0, out_depth_std = 0.0, \
                out_phase = 0.0, out_duration = 0.0, out_depth_snr = 0.0, \
                out_log_like = 0.0

    # By wrapping this in `nogil`, we make it possible to use a ThreadPool
    # with shared memory instead of a multiprocessing.Pool. This can lead to
    # better performance in some cases.
    with nogil:
        fold(N, t, y, ivar, sum_y2, sum_y,
             sum_ivar, ninf, eps, period, n_durations, durations,
             bin_duration, oversample, use_likelihood,
             &out_objective, &out_depth, &out_depth_std,
             &out_duration, &out_phase,
             &out_depth_snr, &out_log_like)

    return (out_objective, out_depth, out_depth_std,
            out_duration, out_phase,
            out_depth_snr, out_log_like)
