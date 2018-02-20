# -*- coding: utf-8 -*-

__all__ = ["TransitPeriodogram"]

import numpy as np

from ... import units

from .methods import transit_periodogram_fast


def has_units(obj):
    return hasattr(obj, 'unit')


def strip_units(*arrs):
    strip = lambda a: None if a is None else np.asarray(a)  # NOQA
    if len(arrs) == 1:
        return strip(arrs[0])
    else:
        return map(strip, arrs)


class TransitPeriodogram(object):

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy)

    def _validate_inputs(self, t, y, dy):
        # Validate shapes of inputs
        if dy is None:
            t, y = np.broadcast_arrays(t, y, subok=True)
        else:
            t, y, dy = np.broadcast_arrays(t, y, dy, subok=True)
        if t.ndim != 1:
            raise ValueError("Inputs (t, y, dy) must be 1-dimensional")

        # validate units of inputs if any is a Quantity
        if any(has_units(arr) for arr in (t, y, dy)):
            t, y = map(units.Quantity, (t, y))
            if dy is not None:
                dy = units.Quantity(dy)
                try:
                    dy = units.Quantity(dy, unit=y.unit)
                except units.UnitConversionError:
                    raise ValueError("Units of dy not equivalent "
                                     "to units of y")
        return t, y, dy

    def _validate_duration(self, duration):
        duration = np.atleast_1d(np.abs(duration))

        if has_units(self.t):
            duration = units.Quantity(duration)
            try:
                duration = units.Quantity(duration, unit=self.t.unit)
            except units.UnitConversionError:
                raise ValueError("Units of duration not equivalent to "
                                 "units of t")
        else:
            if has_units(duration):
                raise ValueError("duration have units while t doesn't.")

        return duration

    def _validate_period_and_duration(self, period, duration):
        duration = self._validate_duration(duration)
        period = np.atleast_1d(np.abs(period))

        if has_units(self.t):
            period = units.Quantity(period)
            try:
                period = units.Quantity(period, unit=self.t.unit)
            except units.UnitConversionError:
                raise ValueError("Units of period not equivalent to "
                                 "units of t")
        else:
            if has_units(period):
                raise ValueError("period have units while t doesn't.")

        if not np.min(period) > np.max(duration):
            raise ValueError("The maximum transit duration must be shorter "
                             "than the minimum period")

        return period, duration

    def _format_results(self, method, period, results):
        (power, depth, depth_err, transit_time, duration, depth_snr,
         log_likelihood) = results

        if has_units(self.t):
            transit_time = units.Quantity(transit_time, unit=self.t.unit)
            duration = units.Quantity(duration, unit=self.t.unit)

        if has_units(self.y):
            depth = units.Quantity(depth, unit=self.y.unit)
            depth_err = units.Quantity(depth_err, unit=self.y.unit)

            power = units.Quantity(power, unit=units.dimensionless_unscaled)
            depth_snr = units.Quantity(depth_snr,
                                       unit=units.dimensionless_unscaled)
            log_likelihood = units.Quantity(log_likelihood,
                                            unit=units.dimensionless_unscaled)

        return TransitPeriodogramResults(method, period, power, depth,
                                         depth_err, transit_time, duration,
                                         depth_snr, log_likelihood)

    def autoperiod(self, duration, minimum_n_transit=3, frequency_factor=1.0,
                   minimum_period=None, maximum_period=None):
        duration = self._validate_duration(duration)
        baseline = self.t.max() - self.t.min()
        min_duration = np.min(duration)

        # Estimate the required frequency spacing
        # Because of the sparsity of a transit, this must be much finer than
        # the frequency resolution for a sinusoidal fit. For a sinusoidal fit,
        # df would be 1/baseline (see LombScargle), but here this should be
        # scaled proportionally to the duration in units of baseline.
        df = frequency_factor * min_duration / baseline**2

        # If a minimum period is not provided, choose one that is twice the
        # maximum duration because we won't be sensitive to any periods
        # shorter than that.
        if minimum_period is None:
            minimum_period = 2.0 * np.max(duration)

        # If no maximum period is provided, choose one by requiring that
        # all signals with at least minimum_n_transit should be detectable.
        if maximum_period is None:
            maximum_period = baseline / minimum_n_transit

        # Compute the number of frequencies and the frequency grid
        nf = 1 + int(np.round((1.0/minimum_period - 1.0/maximum_period)/df))
        return 1.0 / (1.0 / minimum_period - df * np.arange(nf))

    def autopower(self, duration, objective=None, method=None, oversample=10,
                  pool=None, minimum_n_transit=3, minimum_period=None,
                  maximum_period=None):
        period = self.autoperiod(duration,
                                 minimum_n_transit=minimum_n_transit,
                                 minimum_period=minimum_period,
                                 maximum_period=maximum_period)
        return self.power(period, duration, objective=objective, method=method,
                          oversample=oversample, pool=pool)

    def power(self, period, duration, objective=None, method=None,
              oversample=10, pool=None):
        period, duration = self._validate_period_and_duration(period, duration)

        # Check for absurdities in the ``oversample`` choice
        try:
            oversample = int(oversample)
        except TypeError:
            raise ValueError("oversample must be an integer,"
                             " got {0}".format(oversample))
        if oversample < 1:
            raise ValueError("oversample must be greater than or equal to 1")

        # Select the periodogram objective
        if objective is None:
            objective = "likelihood"
        allowed_objectives = ["snr", "likelihood"]
        if objective not in allowed_objectives:
            raise ValueError(("Unrecognized method '{0}'\n"
                              "allowed methods are: {1}")
                             .format(objective, allowed_objectives))
        use_likelihood = (objective == "likelihood")

        # Select the computational method
        if method is None:
            method = "fast"
        allowed_methods = ["fast", "fast_python", "slow"]
        if method not in allowed_methods:
            raise ValueError(("Unrecognized method '{0}'\n"
                              "allowed methods are: {1}")
                             .format(method, allowed_methods))

        # Format and check the input arrays
        time = np.ascontiguousarray(strip_units(self.t), dtype=np.float64)
        flux = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            flux_ivar = np.ones_like(flux)
        else:
            flux_ivar = 1.0 / np.ascontiguousarray(strip_units(self.dy),
                                                   dtype=np.float64)**2

        # Select the correct implementation for the chosen method
        if method == "fast":
            transit_periodogram = transit_periodogram_fast
        elif method == "fast_python":
            assert 0
        else:
            assert 0

        # Run the implementation
        results = transit_periodogram(
            time, flux - np.median(flux), flux_ivar, strip_units(period),
            strip_units(duration), oversample,
            use_likelihood, pool)

        return self._format_results(method, period, results)


class TransitPeriodogramResults(dict):
    """The results of a TransitPeriodogram search

    Attributes
    ----------


    """
    def __init__(self, *args):
        super(TransitPeriodogramResults, self).__init__(zip(
            ("method", "period", "power", "depth", "depth_err", "transit_time",
             "duration", "depth_snr", "log_likelihood"),
            args
        ))

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
