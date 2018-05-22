#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

auto grid_search (
  py::array_t<double> log_like_array,
  py::array_t<double> x_array,
  py::array_t<double> ivar_array,
  int min_n,
  int max_n
) {
  // Check the log likelihood input and compute the shapes
  auto log_like = log_like_array.unchecked<2>();
  if (log_like.ndim() != 2)
    throw std::runtime_error("'log_like_array' must be 2-dimensional");
  int N = log_like.shape(0);
  int M = log_like.shape(1);

  // Check the search range parameters
  int N_out = max_n - min_n + 1;
  if (N_out > N || N_out <= 0 || max_n <= 0 || min_n <= 0)
    throw std::runtime_error("invalid values for 'min_n' and/or 'max_n'");

  // Check the other inputs
  auto x = x_array.unchecked<2>();
  if (x.ndim() != 2 || x.shape(0) != N || x.shape(1) != M)
    throw std::runtime_error("'x_array' must be the same shape as 'log_like_array'");
  auto ivar = ivar_array.unchecked<2>();
  if (ivar.ndim() != 2 || ivar.shape(0) != N || ivar.shape(1) != M)
    throw std::runtime_error("'ivar_array' must be the same shape as 'log_like_array'");

  // Allocate the outputs
  py::array_t<double> log_like_out_array(N_out);
  py::array_t<double> x_out_array(N_out);
  py::array_t<double> ivar_out_array(N_out);
  py::array_t<int> inds_out_array({N_out, 2});

  // Access the outputs
  auto log_like_out = log_like_out_array.mutable_unchecked<1>();
  auto x_out = x_out_array.mutable_unchecked<1>();
  auto ivar_out = ivar_out_array.mutable_unchecked<1>();
  auto inds_out = inds_out_array.mutable_unchecked<2>();

  std::vector<double> log_like_tmp(M), y2_tmp(M), y_tmp(M), ivar_tmp(M);

  double value;
  for (int phase = 0; phase < N_out; ++phase) {
    int bins_per_period = min_n + phase;

    // Initialize the "best" model
    log_like_out(phase) = -std::numeric_limits<double>::infinity();
    x_out(phase)        = 0.0;
    ivar_out(phase)     = 0.0;
    inds_out(phase, 0)  = 0;
    inds_out(phase, 1)  = 0;

    // Loop over phase
    for (int k = 0; k < bins_per_period; ++k) {
      // Initialize all the accumulators to zero
      for (int m = 0; m < M; ++m) {
        log_like_tmp[m] = 0.0;
        y2_tmp[m]       = 0.0;
        y_tmp[m]        = 0.0;
        ivar_tmp[m]     = 0.0;
      }

      // Accumulate over all the transits
      int n = k;
      while (n < N) {
        for (int m = 0; m < M; ++m) {
          log_like_tmp[m] += log_like(n, m);

          value = ivar(n, m);
          ivar_tmp[m] += ivar(n, m);

          value *= x(n, m);
          y_tmp[m] += value;

          value *= x(n, m);
          y2_tmp[m] += value;
        }
        n += bins_per_period;
      }

      // Compute the maximum likelihood x and evaluate the maximum likelihood for that model
      for (int m = 0; m < M; ++m) {
        double x_ml = 0.0,
               log_like_value = log_like_tmp[m];

        // Use the expanded form of the likelihood to compute the likelihood
        log_like_value -= 0.5 * y2_tmp[m];

        // Only include the x terms if some points were in transit
        if (ivar_tmp[m] > 0.0) {
          x_ml = y_tmp[m] / ivar_tmp[m];
          log_like_value += 0.5 * x_ml * y_tmp[m];
          //log_like_value -= 0.5 * x_ml * x_ml * ivar_tmp[m];
        }

        // Save this point if it is a better fit than the current best
        if (log_like_value > log_like_out(phase)) {
          log_like_out(phase) = log_like_value;
          x_out(phase)        = x_ml;
          ivar_out(phase)     = ivar_tmp[m];
          inds_out(phase, 0)  = k;
          inds_out(phase, 1)  = m;
        }
      }
    }
  }

  return std::make_tuple(log_like_out_array, x_out_array, ivar_out_array, inds_out_array);
}


PYBIND11_MODULE(grid_search, m) {
  m.def("grid_search", &grid_search);
}
