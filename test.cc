#include <iostream>
#include <functional>
#include <random>
#include <vector>
#include <cmath>

// Timer for the benchmark.
#if defined(_MSC_VER)
    //no sys/time.h in visual c++
    //http://jakascorner.com/blog/2016/04/time-measurement.html
    #include <chrono>
    double get_timestamp ()
    {
      using micro_s = std::chrono::microseconds;
      auto tnow = std::chrono::steady_clock::now();
      auto d_micro = std::chrono::duration_cast<micro_s>(tnow.time_since_epoch()).count();
      return double(d_micro) * 1.0e-6;
    }
#else
   //no std::chrono in g++ 4.8
   #include <sys/time.h>
   double get_timestamp ()
   {
     struct timeval now;
     gettimeofday (&now, NULL);
     return double(now.tv_usec) * 1.0e-6 + double(now.tv_sec);
   }
#endif

extern "C" int classic_bls (int N, double* t, double* y, double* ivar, double delta_t,
  int n_periods, int* periods, int n_durations, int* durations,
  double* best_power, double* best_depth, double* best_depth_err,
  double* best_duration, double* best_phase);

extern "C" int gridsearch_bls (int N, double* t, double* y, double* ivar, double delta_t,
  int n_periods, int* periods, int n_durations, int* durations,
  double* best_power, double* best_depth, double* best_depth_err,
  double* best_duration, double* best_phase);

int main () {
  std::default_random_engine rng(42);
  std::normal_distribution<double> randn;

  double period = 10.0, t0 = 9.99, tau = 0.1, yerr = 0.05;
  const int N = 5000;
  std::vector<double> t(N), y(N), ivar(N);
  for (int n = 0; n < N; ++n) {
    t[n] = 1000.0 + n * 0.5 / 24;
    if (std::abs(std::fmod(t[n] - t0 + 0.5*period, period) - 0.5*period) < 0.5*tau) {
      y[n] = -0.1 + yerr * randn(rng);
    } else {
      y[n] = yerr * randn(rng);
    }
    ivar[n] = 1.0 / (yerr * yerr);
  }

  double delta_t = 0.001;
  //int n_periods = int((40 - 2) / delta_t);
  int n_periods = 10000;
  std::cout << n_periods << "\n";
  int n_durations = 1;
  std::vector<int> periods(n_periods), durations(n_durations);
  durations[0] = tau / delta_t;

  for (int k = 0; k < n_periods; ++k) {
    periods[k] = 2 / delta_t + k;
  }
  std::cout << "min period = " << (periods[0] * delta_t) << "\n";
  std::cout << "max period = " << (periods[n_periods-1] * delta_t) << "\n";

  std::vector<double> best_power(n_periods), best_depth(n_periods), best_depth_err(n_periods), best_duration(n_periods), best_phase(n_periods);

  double strt = get_timestamp();
  int flag = gridsearch_bls(N, t.data(), y.data(), ivar.data(), delta_t, n_periods, periods.data(), n_durations, durations.data(),
                         best_power.data(), best_depth.data(), best_depth_err.data(), best_duration.data(), best_phase.data());
  std::cout << (get_timestamp() - strt) << " " << flag << "\n";

  int argmax = 0;
  double max = 0.0;
  for (int n = 0; n < n_periods; ++n) {
    if (best_power[n] > max) {
      max = best_power[n];
      argmax = n;
    }
  }

  std::cout << (periods[argmax] * delta_t) << "\n";
  std::cout << (best_phase[argmax]) << "\n";

  return flag;
}
