#include <cmath>
#include <limits>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
using namespace Eigen;

auto compute_bins (
  const Ref<const Matrix<double, Dynamic, 1>>& t,
  const Ref<const Matrix<double, Dynamic, 1>>& y,
  const Ref<const Matrix<double, Dynamic, 1>>& ivar,
  const Ref<const Matrix<double, Dynamic, 1>>& durations,
  int oversample
) {
  int N = t.rows();
  int n_durations = durations.rows();
  double delta_bin = durations.minCoeff() / oversample;

  // Quantize the durations
  Array<int, Dynamic, 1> duration_inds(n_durations);
  for (int m = 0; m < n_durations; ++m)
    duration_inds(m) = int(round(durations(m) / delta_bin));

  // Compute the total number of bins
  double min_t = t.minCoeff();
  int n_bins = int(ceil((t.maxCoeff() - min_t) / delta_bin));

  // Histogram the data
  Array<double, Dynamic, 1> numerator(n_bins), denominator(n_bins);
  numerator.setZero();
  denominator.setZero();
  for (int n = 0; n < N; ++n) {
    int ind = int(floor((t(n) - min_t) / delta_bin));
    if (ind >= n_bins) ind = n_bins;
    numerator(ind)   += y(n) * ivar(n);
    denominator(ind) += ivar(n);
  }

  for (int n = 1; n < n_bins; ++n) {
    numerator(n) += numerator(n-1);
    denominator(n) += denominator(n-1);
  }
  double num_all = numerator(n_bins-1);
  double denom_all = denominator(n_bins-1);
  double log_like_ref = 0.5 * num_all * num_all / denom_all;

  Matrix<double, Dynamic, Dynamic, RowMajor> delta_log_like(n_bins, n_durations),
                                             depth(n_bins, n_durations),
                                             depth_ivar(n_bins, n_durations);
  for (int n = 0; n < n_bins; ++n) {
    for (int m = 0; m < n_durations; ++m) {
      int upper = n + duration_inds(m);
      if (upper >= n_bins) {
        delta_log_like(n, m) = 0.0;
        depth(n, m)          = 0.0;
        depth_ivar(n, m)     = 0.0;
        continue;
      }

      double denom = denominator(upper) - denominator(n);

      // No points in transit
      if (denom <= std::numeric_limits<double>::epsilon()) {
        delta_log_like(n, m) = 0.0;
        depth(n, m)          = 0.0;
        depth_ivar(n, m)     = 0.0;
        continue;
      }

      double num = numerator(upper) - numerator(n);
      double num_out = num_all - num;
      double denom_out = denom_all - denom;

      double h_in = num / denom;
      double h_out = num_out / denom_out;
      depth(n, m) = h_out - h_in;
      depth_ivar(n, m) = denom * denom_out / denom_all;
      delta_log_like(n, m) = 0.5*h_out*num_out + 0.5*h_in*num - log_like_ref;
    }
  }

  return std::make_tuple(delta_log_like, depth, depth_ivar);
}

auto grid_search (
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& log_like,
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& x,
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& ivar,
  const Ref<const Matrix<int, Dynamic, 1>>& periods
) {
  // Compute the dimensions
  int N = log_like.rows();  // number of bins
  int M = log_like.cols();  // number of durations
  int N_out = periods.rows();

  // Check all the dimensions
  if (N_out <= 0 || periods.minCoeff() <= 1 || periods.maxCoeff() >= N)
    throw std::runtime_error("'periods' must be constrained to the exclusive range (1, N)");
  if (x.rows() != N || x.cols() != M)
    throw std::runtime_error("'x' must be the same shape as 'log_like'");
  if (ivar.rows() != N || ivar.cols() != M)
    throw std::runtime_error("'ivar' must be the same shape as 'log_like'");

  // Allocate the outputs
  VectorXd log_like_out(N_out),
           x_out(N_out),
           ivar_out(N_out);
  Matrix<int, Dynamic, 2, RowMajor> inds_out(N_out, 2);

  // Initialize the output arrays
  log_like_out.setConstant(-std::numeric_limits<double>::infinity());
  x_out.setZero();
  ivar_out.setZero();
  inds_out.setZero();

  // Pre-compute some quantities
  Array<double, Dynamic, Dynamic, RowMajor> y = ivar.array() * x.array();
  Array<double, Dynamic, Dynamic, RowMajor> y2 = y * x.array();

  // The outer loop is over test periods
  // Run the loop in parallel if openmp is enabled
#if defined(_OPENMP)
#pragma omp parallel for
  for (int period = 0; period < N_out; ++period) {
    Array<double, Dynamic, 1> log_like_tmp(M), y2_tmp(M), y_tmp(M), ivar_tmp(M);
#else
  Array<double, Dynamic, 1> log_like_tmp(M), y2_tmp(M), y_tmp(M), ivar_tmp(M);
  for (int period = 0; period < N_out; ++period) {
#endif

    int bins_per_period = periods(period);

    // Loop over phase
    for (int k = 0; k < bins_per_period; ++k) {
      // Initialize all the accumulators to zero
      log_like_tmp.setZero();
      y2_tmp.setZero();
      y_tmp.setZero();
      ivar_tmp.setZero();

      // Accumulate over all the transits
      for (int n = k; n < N; n += bins_per_period) {
        log_like_tmp += log_like.row(n).array();
        ivar_tmp += ivar.row(n).array();
        y_tmp += y.row(n);
        y2_tmp += y2.row(n);
      }

      // Use the expanded form of the likelihood to compute the likelihood
      log_like_tmp -= 0.5 * y2_tmp;
      y_tmp.array() /= ivar_tmp.array();
      log_like_tmp += 0.5 * y_tmp * y_tmp * ivar_tmp;

      int ind = 0;
      double max_log_like = log_like_tmp.maxCoeff(&ind);

      // Save this point if it is a better fit than the current best
      if (ivar_tmp(ind) > 0.0 && max_log_like > log_like_out(period)) {
        log_like_out(period) = max_log_like;
        x_out(period)        = y_tmp(ind);
        ivar_out(period)     = ivar_tmp(ind);
        inds_out(period, 0)  = k;
        inds_out(period, 1)  = ind;
      }
    }
  }

  return std::make_tuple(log_like_out, x_out, ivar_out, inds_out);
}


PYBIND11_MODULE(grid_search, m) {
  m.def("compute_bins", &compute_bins,
        py::arg("t").noconvert(),
        py::arg("y").noconvert(),
        py::arg("ivar").noconvert(),
        py::arg("durations").noconvert(),
        py::arg("oversample") = 10);
  m.def("grid_search", &grid_search,
        py::arg("log_like").noconvert(),
        py::arg("x").noconvert(),
        py::arg("ivar").noconvert(),
        py::arg("periods"));
}
