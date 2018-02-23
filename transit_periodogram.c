#include <math.h>
#include <stdlib.h>

double compute_log_like(
    double y_in,
    double y_out,
    double ivar_in,
    double sum_y2,
    double sum_y,
    double sum_ivar
) {
    double arg = y_in - y_out;
    double chi2 = sum_y2 - 2*y_out*sum_y;
    chi2 += y_out*y_out * sum_ivar;
    chi2 -= arg*arg * ivar_in;
    return -0.5*chi2;
}

int fold (
    // Inputs
    int N,                   // Length of the time array
    double* t,               // The list of timestamps
    double* y,               // The y measured at ``t``
    double* ivar,            // The inverse variance of the y array
    double sum_y2,           // The precomputed value of sum(y^2 * ivar)
    double sum_y,            // The precomputed value of sum(y * ivar)
    double sum_ivar,         // The precomputed value of sum(ivar)
    double ninf,             // Negative infinity for DTYPE
    double eps,              // Machine precision for DTYPE

    int n_periods,
    double* periods,         // The period to test in units of ``t``

    int n_durations,         // Length of the durations array
    int* durations,          // The durations to test in units of ``bin_duration``
    double bin_duration,     // The width of the fine-grain bins to use in units of
                             // ``t``
    int oversample,          // The number of ``bin_duration`` bins in the maximum duration

    int use_likelihood,      // A flag indicating the periodogram type
                             // 0 - depth signal-to-noise
                             // 1 - log likelihood

    // Work arrays
    double* mean_y,
    double* mean_ivar,

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

                // Either compute the log likelihood or the signal-to-noise ratio
                if (use_likelihood) {
                    objective = compute_log_like(y_in, y_out, ivar_in, sum_y2, sum_y, sum_ivar);
                } else {
                    depth = y_out - y_in;
                    depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
                    objective = depth / depth_std;
                }

                // If this is the best result seen so far, keep it
                if (objective > best_objective[p]) {
                    if (use_likelihood) {
                        depth = y_out - y_in;
                        depth_std = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
                        depth_snr = depth / depth_std;
                        log_like = objective;
                    } else {
                        log_like = compute_log_like(y_in, y_out, ivar_in, sum_y2, sum_y, sum_ivar);
                        depth_snr = objective;
                    }

                    // Save the current model
                    best_objective[p] = objective;
                    best_depth[p]     = depth;
                    best_depth_std[p] = depth_std;
                    best_depth_snr[p] = depth_snr;
                    best_log_like[p]  = log_like;
                    best_duration[p]  = durations[k] * bin_duration;
                    best_phase[p]     = fmod(n*bin_duration + 0.5*best_duration[p], period);
                }
            }
        }
    }

    return 0;
}
