#include <math.h>
#include <stdlib.h>

void compute_objective(
    double y_in,
    double y_out,
    double ivar_in,
    double ivar_out,
    double sum_y2,
    double sum_y,
    double sum_ivar,
    int obj_flag,
    double* objective,
    double* log_likelihood,
    double* depth,
    double* depth_std,
    double* depth_snr
) {
    if (obj_flag) {
        double arg = y_in - y_out;
        double chi2 = sum_y2 - 2*y_out*sum_y;
        chi2 += y_out*y_out * sum_ivar;
        chi2 -= arg*arg * ivar_in;
        *log_likelihood = -0.5*chi2;
        *objective = *log_likelihood;
    } else {
        *depth = y_out - y_in;
        *depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
        *depth_snr = *depth / *depth_std;
        *objective = *depth_snr;
    }
}

int fold (
    // Inputs
    int N,                   // Length of the time array
    double* t,               // The list of timestamps
    double* y,               // The y measured at ``t``
    double* ivar,            // The inverse variance of the y array

    /*double sum_y2,           // The precomputed value of sum(y^2 * ivar)*/
    /*double sum_y,            // The precomputed value of sum(y * ivar)*/
    /*double sum_ivar,         // The precomputed value of sum(ivar)*/
    double ninf,             // Negative infinity for DTYPE
    double eps,              // Machine precision for DTYPE

    int n_periods,
    double* periods,         // The period to test in units of ``t``

    int n_durations,         // Length of the durations array
    double* durations,       // The durations to test in units of ``bin_duration``
    int oversample,          // The number of ``bin_duration`` bins in the maximum duration

    int obj_flag,      // A flag indicating the periodogram type
                             // 0 - depth signal-to-noise
                             // 1 - log likelihood

    // Work arrays - must be at least n_bins_max+1 each
    /*double* mean_y,*/
    /*double* mean_ivar,*/

    // Outputs
    double* best_objective,  // The value of the periodogram at maximum
    double* best_depth,      // The estimated depth at maximum
    double* best_depth_std,  // The uncertainty on ``best_depth``
    double* best_duration,   // The best fitting duration in units of ``t``
    double* best_phase,      // The phase of the mid-transit time in units of
                             // ``t``
    double* best_depth_snr,  // The signal-to-noise ratio of the depth estimate
    double* best_log_like    // The log likelihood at maximum
) {
    int ind, n, k, p, n_max, dur, n_bins;
    double period, y_in, y_out, ivar_in, ivar_out, depth, depth_std, depth_snr, log_like, objective;

    // Start by finding the period and duration ranges
    double max_period = periods[0], min_period = periods[0];
    for (k = 1; k < n_periods; ++k) {
        if (periods[k] < min_period) min_period = periods[k];
        if (periods[k] > max_period) max_period = periods[k];
    }
    if (min_period < eps) return 1;
    double min_duration = durations[0], max_duration = durations[0];
    for (k = 1; k < n_durations; ++k) {
        if (durations[k] < min_duration) min_duration = durations[k];
        if (durations[k] > max_duration) max_duration = durations[k];
    }
    if ((max_duration > min_period) || (min_duration < eps)) return 2;

    // Compute the durations in terms of bin_duration
    double bin_duration = min_duration / ((double)oversample);
    int max_n_bins = (int)(max_period / bin_duration) + oversample;
    int* durations_index = (int*)malloc(n_durations*sizeof(int));
    if (durations_index == NULL) return -1;
    for (k = 0; k < n_durations; ++k) {
        durations_index[k] = (int)(durations[k] / bin_duration);
        if (durations_index[k] <= 0) durations_index[k] = 1;
    }

    // Allocate the work arrays
    double* mean_y = (double*)malloc((max_n_bins+1)*sizeof(double));
    if (mean_y == NULL) {
        free(durations_index);
        return -2;
    }
    double* mean_ivar = (double*)malloc((max_n_bins+1)*sizeof(double));
    if (mean_ivar == NULL) {
        free(mean_y);
        free(durations_index);
        return -3;
    }

    // Pre-accumulate some factors.
    double sum_y2 = 0.0, sum_y = 0.0, sum_ivar = 0.0;
    for (n = 0; n < N; ++n) {
        double tmp = y[n] * ivar[n];
        sum_y2 += y[n] * tmp;
        sum_y += tmp;
        sum_ivar += ivar[n];
    }

    // Loop over periods and do the search
    for (p = 0; p < n_periods; ++p) {
        period = periods[p];
        n_bins = (int)(period / bin_duration) + oversample;

        // This first pass bins the data into a fine-grain grid in phase from zero
        // to period and computes the weighted sum and inverse variance for each
        // bin.
        for (n = 0; n < n_bins+1; ++n) {
            mean_y[n] = 0.0;
            mean_ivar[n] = 0.0;
        }
        for (n = 0; n < N; ++n) {
            ind = (int)(fabs(fmod(t[n], period)) / bin_duration) + 1;
            mean_y[ind] += y[n] * ivar[n];
            mean_ivar[ind] += ivar[n];
        }

        // To simplify calculations below, we wrap the binned values around and pad
        // the end of the array with the first ``oversample`` samples.
        for (n = 1, ind = n_bins - oversample; n <= oversample; ++n, ++ind) {
            mean_y[ind] = mean_y[n];
            mean_ivar[ind] = mean_ivar[n];
        }

        // To compute the estimates of the in-transit flux, we need the sum of
        // mean_y and mean_ivar over a given set of transit points. To get this
        // fast, we can compute the cumulative sum and then use differences between
        // points separated by ``duration`` bins. Here we convert the mean arrays
        // to cumulative sums.
        for (n = 1; n <= n_bins; ++n) {
            mean_y[n] += mean_y[n-1];
            mean_ivar[n] += mean_ivar[n-1];
        }

        // Then we loop over phases (in steps of n_bin) and durations and find the
        // best fit value. By looping over durations here, we get to reuse a lot of
        // the computations that we did above.
        best_objective[p] = ninf;
        for (k = 0; k < n_durations; ++k) {
            dur = durations[k];
            n_max = n_bins-dur;
            for (n = 0; n <= n_max; ++n) {
                // Estimate the in-transit and out-of-transit flux
                y_in = mean_y[n+dur] - mean_y[n];
                ivar_in = mean_ivar[n+dur] - mean_ivar[n];
                y_out = sum_y - y_in;
                ivar_out = sum_ivar - ivar_in;

                // Skip this model if there are no points in transit
                if ((ivar_in < eps) || (ivar_out < eps)) {
                    continue;
                }

                // Normalize to compute the actual value of the flux
                y_in /= ivar_in;
                y_out /= ivar_out;

                // Either compute the log likelihood or the signal-to-noise
                // ratio
                compute_objective(y_in, y_out, ivar_in, ivar_out, sum_y2, sum_y, sum_ivar, obj_flag,
                        &objective, &log_like, &depth, &depth_std, &depth_snr);

                // If this is the best result seen so far, keep it
                if (objective > best_objective[p]) {
                    best_objective[p] = objective;

                    // Compute the other parameters
                    compute_objective(y_in, y_out, ivar_in, ivar_out, sum_y2, sum_y, sum_ivar, (obj_flag == 0),
                            &objective, &log_like, &depth, &depth_std, &depth_snr);

                    best_depth[p]     = depth;
                    best_depth_std[p] = depth_std;
                    best_depth_snr[p] = depth_snr;
                    best_log_like[p]  = log_like;
                    best_duration[p]  = durations_index[k] * bin_duration;
                    best_phase[p]     = fmod(n*bin_duration + 0.5*best_duration[p], period);
                }
            }
        }
    }

    // Clean up
    free(durations_index);
    free(mean_y);
    free(mean_ivar);

    return 0;
}
