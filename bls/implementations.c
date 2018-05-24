#include <math.h>
#include <float.h>
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

int get_ranges (int N, double* t, int n_periods, int* periods, int n_durations, int* durations,
  double* min_t, double* max_t, int* min_period, int* max_period, int* min_duration, int* max_duration)
{
  *max_period = periods[0];
  *min_period = periods[0];
  int k;
  for (k = 1; k < n_periods; ++k) {
    if (periods[k] < *min_period) *min_period = periods[k];
    if (periods[k] > *max_period) *max_period = periods[k];
  }
  if (*min_period <= 0) return 1;

  *min_duration = durations[0];
  *max_duration = durations[0];
  for (k = 1; k < n_durations; ++k) {
    if (durations[k] < *min_duration) *min_duration = durations[k];
    if (durations[k] > *max_duration) *max_duration = durations[k];
  }
  if ((*max_duration > *min_period) || (*min_duration <= 0)) return 2;

  *min_t = t[0];
  *max_t = t[0];
  for (k = 1; k < N; ++k) {
    if (t[k] < *min_t) *min_t = t[k];
    if (t[k] > *max_t) *max_t = t[k];
  }

  return 0;
}


int classic_bls (
    // Inputs
    int N,                   // Length of the time array
    double* t,               // The list of timestamps
    double* y,               // The y measured at ``t``
    double* ivar,            // The inverse variance of the y array

    double delta_t,          // The time resolution of the phase search in units of ``t``

    int n_periods,
    int* periods,            // The period to test in units of ``delta_t``

    int n_durations,         // Length of the durations array
    int* durations,          // The durations to test in units of ``delta_t``

    // Outputs (all have length n_periods)
    double* best_power,      // The value of the periodogram at maximum
    double* best_depth,      // The estimated depth at maximum
    double* best_depth_err,  // The uncertainty on ``best_depth``
    double* best_duration,   // The best fitting duration in units of ``t``
    double* best_phase       // The phase of the mid-transit time in units of ``t``
) {
  // Start by finding the period and duration ranges
  double min_t = t[0], max_t = t[0];
  int min_period = periods[0], max_period = periods[0], min_duration = durations[0], max_duration = durations[0];
  int flag = get_ranges (N, t, n_periods, periods, n_durations, durations, &min_t, &max_t, &min_period, &max_period, &min_duration, &max_duration);
  if (flag) return flag;

  // Also work out the minimum time and quantize the times into a delta_t offset
  // from this minimum time.
  int* t_index = (int*)malloc(N * sizeof(int));
  if (t_index == NULL) {
    return -1;
  }
  int k;
  for (k = 0; k < N; ++k) {
    t_index[k] = (int)(floor((t[k] - min_t) / delta_t));
  }

  // The maximum number of bins that will ever be needed is the maximum duration
  // added to the maximum period. This lets us easily deal with wrapping issues
  // where half of the transit is at a phase of zero and the other half is at a
  // phase of period.
  int max_n_bins = max_period + max_duration;

  // Compute the amount of extra memory that will be needed by OpenMP
  int nthreads, blocksize = max_n_bins;
#pragma omp parallel
  {
#if defined(_OPENMP)
    nthreads = omp_get_num_threads();
#else
    nthreads = 1;
#endif
  }

  // Allocate the work arrays
  double* mean_y_0 = (double*)malloc(nthreads*blocksize*sizeof(double));
  if (mean_y_0 == NULL) {
    free(t_index);
    return -2;
  }
  double* mean_ivar_0 = (double*)malloc(nthreads*blocksize*sizeof(double));
  if (mean_ivar_0 == NULL) {
    free(t_index);
    free(mean_y_0);
    return -3;
  }

  // Loop over periods and do the search
  int p;
#pragma omp parallel for
  for (p = 0; p < n_periods; ++p) {
#if defined(_OPENMP)
    int ithread = omp_get_thread_num();
#else
    int ithread = 0;
#endif
    int block = blocksize * ithread;
    int period = periods[p];
    int n_bins = period + max_duration;

    double* mean_y = mean_y_0 + block;
    double* mean_ivar = mean_ivar_0 + block;

    // This first pass bins the data into a fine-grain grid in phase from zero
    // to period and computes the weighted sum and inverse variance for each
    // bin.
    int n, ind;
    for (n = 0; n < n_bins; ++n) {
      mean_y[n] = 0.0;
      mean_ivar[n] = 0.0;
    }
    for (n = 0; n < N; ++n) {
      int ind = t_index[n] % period + 1;
      mean_y[ind] += y[n] * ivar[n];
      mean_ivar[ind] += ivar[n];
    }

    // To simplify calculations below, we wrap the binned values around and pad
    // the end of the array with the first ``max_duration - 1`` samples.
    for (n = 1, ind = period + 1; n < max_duration - 1; ++n, ++ind) {
      mean_y[ind] = mean_y[n];
      mean_ivar[ind] = mean_ivar[n];
    }

    // To compute the estimates of the in-transit flux, we need the sum of
    // mean_y and mean_ivar over a given set of transit points. To get this
    // fast, we can compute the cumulative sum and then use differences between
    // points separated by ``duration`` bins. Here we convert the mean arrays
    // to cumulative sums.
    for (n = 1; n < n_bins; ++n) {
      mean_y[n] += mean_y[n-1];
      mean_ivar[n] += mean_ivar[n-1];
    }

    // Taking into account the wrapping from above, the total sums across the
    // time series will be at the ``period`` index of these arrays.
    double sum_y    = mean_y[period];
    double sum_ivar = mean_ivar[period];

    // Then we loop over phases (in steps of n_bin) and durations and find the
    // best fit value. By looping over durations here, we get to reuse a lot of
    // the computations that we did above.
    double power, depth, ref_y, ref_ivar;
    best_power[p] = -INFINITY;
    for (k = 0; k < n_durations; ++k) {
      int dur = durations[k];
      int phase_step = dur / min_duration;
      if (phase_step < 1) phase_step = 1;
      for (n = 0; n <= period; n += phase_step) {
        ref_y    = mean_y[n];
        ref_ivar = mean_ivar[n];
        int ind = n + dur;  // The index of the end of transit
        double y_in = mean_y[ind] - ref_y;
        double ivar_in = mean_ivar[ind] - ref_ivar;
        double y_out = sum_y - y_in;
        double ivar_out = sum_ivar - ivar_in;

        // Skip this model if there are no points in transit
        if ((ivar_in < DBL_EPSILON) || (ivar_out < DBL_EPSILON)) {
          continue;
        }

        // Normalize to compute the actual value of the flux
        y_in /= ivar_in;
        y_out /= ivar_out;

        // Compute the delta log likelihood
        depth = y_out - y_in;
        power = 0.5*ivar_in*depth*depth;

        // If this is the best result seen so far, keep it
        if (y_out >= y_in && power > best_power[p]) {
          best_power[p]     = power;
          best_depth[p]     = depth;
          best_depth_err[p] = sqrt(1.0 / ivar_in + 1.0 / ivar_out);
          best_duration[p]  = dur * delta_t;
          best_phase[p]     = n * delta_t + 0.5 * delta_t * dur + min_t;
        }
      }
    }
  }

  // Clean up
  free(t_index);
  free(mean_y_0);
  free(mean_ivar_0);

  return 0;
}

int gridsearch_bls (
    // Inputs
    int N,                   // Length of the time array
    double* t,               // The list of timestamps
    double* y,               // The y measured at ``t``
    double* ivar,            // The inverse variance of the y array

    double delta_t,          // The time resolution of the phase search in units of ``t``

    int n_periods,
    int* periods,            // The period to test in units of ``delta_t``

    int n_durations,         // Length of the durations array
    int* durations,          // The durations to test in units of ``delta_t``

    // Outputs (all have length n_periods)
    double* best_power,      // The value of the periodogram at maximum
    double* best_depth,      // The estimated depth at maximum
    double* best_depth_err,  // The uncertainty on ``best_depth``
    double* best_duration,   // The best fitting duration in units of ``t``
    double* best_phase       // The phase of the mid-transit time in units of ``t``
) {
  // Start by finding the period and duration ranges
  double min_t = t[0], max_t = t[0];
  int min_period = periods[0], max_period = periods[0], min_duration = durations[0], max_duration = durations[0];
  int flag = get_ranges (N, t, n_periods, periods, n_durations, durations, &min_t, &max_t, &min_period, &max_period, &min_duration, &max_duration);
  if (flag) return flag;

  // Compute the total number of bins
  int n_bins = (int)(ceil((max_t - min_t) / delta_t)) + 1;

  // Histogram the data:
  // Allocate the bins
  double* numerator = malloc(n_bins * sizeof(double));
  if (numerator == NULL) return -1;
  double* denominator = malloc(n_bins * sizeof(double));
  if (denominator == NULL) {
    free(numerator);
    return -2;
  }

  // The initial values should be set to zero
  int n;
  for (n = 0; n < n_bins; ++n) {
    numerator[n]   = 0.0;
    denominator[n] = 0.0;
  }

  // Loop over data points and assign each one to its bin
  for (int n = 0; n < N; ++n) {
    int ind = (int)(floor((t[n] - min_t) / delta_t)) + 1;
    if (ind >= n_bins) ind = n_bins;
    numerator[ind]   += y[n] * ivar[n];
    denominator[ind] += ivar[n];
  }

  // Accumulate the cumsum
  for (int n = 1; n < n_bins; ++n) {
    numerator[n] += numerator[n-1];
    denominator[n] += denominator[n-1];
  }
  double num_all = numerator[n_bins-1];
  double denom_all = denominator[n_bins-1];
  double log_like_ref = 0.5 * num_all * num_all / denom_all;

  // Allocate memory for the grid of models
  int n_grid = n_bins * n_durations;
  double* delta_log_like = malloc(n_grid * sizeof(double));
  if (delta_log_like == NULL) {
    free(numerator);
    free(denominator);
    return -3;
  }
  double* depth = malloc(n_grid * sizeof(double));
  if (depth == NULL) {
    free(delta_log_like);
    free(numerator);
    free(denominator);
    return -4;
  }
  double* depth_ivar = malloc(n_grid * sizeof(double));
  if (depth_ivar == NULL) {
    free(depth);
    free(delta_log_like);
    free(numerator);
    free(denominator);
    return -5;
  }

  int m;
  int ind = 0;
  for (m = 0; m < n_durations; ++m) {
    int dur = durations[m];
    for (n = 0; n < n_bins; ++n, ++ind) {
      int upper = n + dur;
      if (upper >= n_bins) {
        delta_log_like[ind] = 0.0;
        depth[ind]          = 0.0;
        depth_ivar[ind]     = 0.0;
        continue;
      }

      double denom = denominator[upper] - denominator[n];

      // No points in transit
      if (denom <= DBL_EPSILON) {
        delta_log_like[ind] = 0.0;
        depth[ind]          = 0.0;
        depth_ivar[ind]     = 0.0;
        continue;
      }

      double num = numerator[upper] - numerator[n];
      double num_out = num_all - num;
      double denom_out = denom_all - denom;

      double h_in = num / denom;
      double h_out = num_out / denom_out;
      depth[ind]          = h_out - h_in;
      depth_ivar[ind]     = denom * denom_out / denom_all;
      delta_log_like[ind] = 0.5*h_out*num_out + 0.5*h_in*num - log_like_ref;
    }
  }

  free(numerator);
  free(denominator);

  // GRID SEARCH

  int i, k, p, phase, index;
  for (p = 0; p < n_periods; ++p) {
    int period = periods[p];
    best_power[p] = -INFINITY;
    for (k = 0; k < n_durations; ++k) {
      int dur = durations[k];
      int phase_step = dur / min_duration;
      if (phase_step < 1) phase_step = 1;
      for (phase = 0; phase <= period; phase += phase_step) {
        // Loop over transits and accumulate the log likelihood and depth estimates
        double log_like = 0.0;
        double iv = 0.0;
        double y = 0.0;
        double y2 = 0.0;
        for (i = phase, index = k*n_bins; i < n_bins; i += period, index += period) {
          log_like += delta_log_like[index];
          double value = depth_ivar[index];
          iv += value;
          value *= depth[index];
          y += value;
          value *= depth[index];
          y2 += value;
        }

        // Use the expanded form of the likelihood to compute the likelihood
        log_like -= 0.5 * y2;
        y /= iv;
        log_like += 0.5 * y * y * iv;

        // If this is the best result seen so far, keep it
        if (y > 0 && log_like > best_power[p]) {
          best_power[p]     = log_like;
          best_depth[p]     = y;
          best_depth_err[p] = 1.0 / sqrt(iv);
          best_duration[p]  = dur * delta_t;
          best_phase[p]     = phase * delta_t + 0.5 * delta_t * dur + min_t;
        }
      }
    }
  }

  free(depth);
  free(depth_ivar);
  free(delta_log_like);

  return 0;
}
