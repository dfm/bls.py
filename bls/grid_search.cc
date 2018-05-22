#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace py = pybind11;
using namespace Eigen;

auto grid_search (
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& log_like,
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& x,
  const Ref<const Matrix<double, Dynamic, Dynamic, RowMajor>>& ivar,
  const Ref<const Matrix<int, Dynamic, 1>>& periods
) {
  // Compute the dimensions
  int N = log_like.rows();
  int M = log_like.cols();
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

  // The outer loop is over test periods
  // Run the loop in parallel if openmp is enabled
#if defined(_OPENMP)
#pragma omp parallel for
  for (int period = 0; period < N_out; ++period) {
    Array<double, Dynamic, 1> log_like_tmp(M), y2_tmp(M), y_tmp(M), ivar_tmp(M), value(M);
#else
  Array<double, Dynamic, 1> log_like_tmp(M), y2_tmp(M), y_tmp(M), ivar_tmp(M), value(M);
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

        value = ivar.row(n).array();
        ivar_tmp += value;

        value *= x.row(n).array();
        y_tmp += value;

        value *= x.row(n).array();
        y2_tmp += value;
      }

      // Use the expanded form of the likelihood to compute the likelihood
      log_like_tmp -= 0.5 * y2_tmp;
      y_tmp.array() /= ivar_tmp.array();
      log_like_tmp += 0.5 * y_tmp * y_tmp * ivar_tmp;

      // Loop over durations and find the maximum likelihood
      for (int m = 0; m < M; ++m) {
        // Save this point if it is a better fit than the current best
        if (ivar_tmp(m) > 0.0 && log_like_tmp(m) > log_like_out(period)) {
          log_like_out(period) = log_like_tmp(m);
          x_out(period)        = y_tmp(m);
          ivar_out(period)     = ivar_tmp(m);
          inds_out(period, 0)  = k;
          inds_out(period, 1)  = m;
        }
      }
    }
  }

  return std::make_tuple(log_like_out, x_out, ivar_out, inds_out);
}


PYBIND11_MODULE(grid_search, m) {
  m.def("grid_search", &grid_search);
}
