# -*- coding: utf-8 -*-

__all__ = ["TransitPeriodogram", "TransitPeriodogramResults"]

import numpy as np

from ...tests.helper import assert_quantity_allclose
from ... import units
from ..lombscargle.core import has_units, strip_units

from . import methods


def validate_unit_consistency(reference_object, input_object):
    if has_units(reference_object):
        input_object = units.Quantity(input_object, unit=reference_object.unit)
    else:
        if has_units(input_object):
            input_object = units.Quantity(input_object, unit=units.one)
            input_object = input_object.value
    return input_object


class TransitPeriodogram(object):
    """Compute the Transit Periodogram

    This method is a commonly used tool for discovering transiting exoplanets
    or eclipsing binaries in photometric time series datasets. This
    implementation is based on the "box least squares (BLS)" method described
    in [1]_ with added support for observational uncertainties,
    parallelization, and a likelihood model.

    Parameters
    ----------
    t : array-like or Quantity
        Sequence of observation times.
    y : array-like or Quantity
        Sequence of observations associated with times ``t``.
    dy : float, array-like or Quantity, optional
        Error or sequence of observational errors associated with times ``t``.

    Examples
    --------
    Generate noisy data with a transit:

    >>> rand = np.random.RandomState(42)
    >>> t = rand.uniform(0, 10, 500)
    >>> y = np.ones_like(t)
    >>> y[np.abs((t + 1.0)%2.0-1)<0.08] = 1.0 - 0.1
    >>> y += 0.01 * rand.randn(len(t))

    Compute the transit periodogram on a heuristically determined period grid
    and find the period with maximum power:

    >>> model = TransitPeriodogram(t, y)
    >>> results = model.autopower(0.16)
    >>> results.period[np.argmax(results.power)]  # doctest: +FLOAT_CMP
    2.005441310651872

    Compute the periodogram on a user-specified period grid:

    >>> periods = np.linspace(1.9, 2.1, 5)
    >>> results = model.power(periods, 0.16)
    >>> results.power  # doctest: +FLOAT_CMP
    array([-0.142265  , -0.12027131, -0.02520321, -0.10649646, -0.13725468])

    If the inputs are AstroPy Quantities with units, the units will be
    validated and the outputs will also be Quantities with appropriate units:

    >>> from astropy import units as u
    >>> t = t * u.day
    >>> y = y * u.dimensionless_unscaled
    >>> model = TransitPeriodogram(t, y)
    >>> results = model.autopower(0.16 * u.day)
    >>> results.period.unit
    Unit("d")
    >>> results.power.unit
    Unit(dimensionless)

    References
    ----------
    .. [1] Kovacs, Zucker, & Mazeh (2002), A&A, 391, 369
        (arXiv:astro-ph/0206099)

    """

    def __init__(self, t, y, dy=None):
        self.t, self.y, self.dy = self._validate_inputs(t, y, dy)

    def autoperiod(self, duration,
                   minimum_period=None, maximum_period=None,
                   minimum_n_transit=3, frequency_factor=1.0):
        """Determine a suitable grid of periods

        This method uses a set of heuristics to select a conservative period
        grid that is uniform in frequency. This grid might be too fine for
        some user's needs depending on the precision requirements or the
        sampling of the data. The grid can be made coarser by increasing
        ``frequency_factor``.

        Parameters
        ----------
        duration : float, array-like or Quantity
            The set of durations that will be considered.
        minimum_period, maximum_period : float or Quantity, optional
            The minimum/maximum periods to search. If not provided, these will
            be computed as described in the notes below.
        minimum_n_transits : int, optional
            If ``maximum_period`` is not provided, this is used to compute the
            maximum period to search by asserting that any systems with at
            least ``minimum_n_transits`` will be within the range of searched
            periods. Note that this is not the same as requiring that
            ``minimum_n_transits`` be required for detection. The default
            value is ``3``.
        frequency_factor : float, optional
            A factor to control the frequency spacing as described in the
            notes below. The default value is ``1.0``.

        Returns
        -------
        period : array-like or Quantity
            The set of periods computed using these heuristics with the same
            units as ``t``.

        Notes
        -----
        The default minimum period is chosen to be twice the maximum duration
        because there won't be much sensitivity to periods shorter than that.

        The default maximum period is computed as

        .. code-block:: python

            maximum_period = (max(t) - min(t)) / minimum_n_transits

        ensuring that any systems with at least ``minimum_n_transits`` are
        within the range of searched periods.

        The frequency spacing is given by

        .. code-block:: python

            df = frequency_factor * min(duration) / (max(t) - min(t))**2

        so the grid can be made finer by decreasing ``frequency_factor`` or
        coarser by increasing ``frequency_factor``.

        """
        duration = self._validate_duration(duration)
        baseline = strip_units(self.t.max() - self.t.min())
        min_duration = strip_units(np.min(duration))

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
            minimum_period = 2.0 * strip_units(np.max(duration))
        else:
            minimum_period = validate_unit_consistency(self.t, minimum_period)
            minimum_period = strip_units(minimum_period)

        # If no maximum period is provided, choose one by requiring that
        # all signals with at least minimum_n_transit should be detectable.
        if maximum_period is None:
            maximum_period = baseline / minimum_n_transit
        else:
            maximum_period = validate_unit_consistency(self.t, maximum_period)
            maximum_period = strip_units(maximum_period)

        if maximum_period < minimum_period:
            minimum_period, maximum_period = maximum_period, minimum_period

        # Convert bounds to frequency
        minimum_frequency = 1.0/strip_units(maximum_period)
        maximum_frequency = 1.0/strip_units(minimum_period)

        # Compute the number of frequencies and the frequency grid
        nf = 1 + int(np.round((maximum_frequency - minimum_frequency)/df))
        return 1.0/(maximum_frequency-df*np.arange(nf)) * self._t_unit()

    def autopower(self, duration, objective=None, method=None, oversample=10,
                  pool=None, minimum_n_transit=3, minimum_period=None,
                  maximum_period=None):
        """Compute the periodogram at set of heuristically determined periods

        This method calls :func:`TransitPeriodogram.autoperiod` to determine
        the period grid and then :func:`TransitPeriodogram.power` to compute
        the periodogram. See those methods for documentation of the arguments.

        """
        period = self.autoperiod(duration,
                                 minimum_n_transit=minimum_n_transit,
                                 minimum_period=minimum_period,
                                 maximum_period=maximum_period)
        return self.power(period, duration, objective=objective, method=method,
                          oversample=oversample, pool=pool)

    def power(self, period, duration, objective=None, method=None,
              oversample=10, pool=None):
        """Compute the periodogram for a set of periods

        Parameters
        ----------
        period : array-like or Quantity
            The periods where the power should be computed
        duration : float, array-like or Quantity
            The set of durations to test
        objective : {'likelihood', 'snr'}, optional
            The scalar that should be optimized to find the best fit phase,
            duration, and depth. This can be either ``'likelihood'`` (default)
            to optimize the log-likelihood of the model, or ``'snr'`` to
            optimize the signal-to-noise with which the transit depth is
            measured.
        method : {'fast', 'slow'}, optional
            The computational method used to compute the periodogram. This is
            mainly included for the purposes of testing and most users will
            want to use the optimized ``'fast'`` method (default) that is
            implemented in Cython.  ``'slow'`` is a brute-force method that is
            used to test the results of the ``'fast'`` method.
        oversample : int, optional
            The number of bins per duration that should be used. This sets the
            time resolution of the phase fit with larger values of
            ``oversample`` yielding a finer grid and higher computational cost.
        pool : optional
            If provided, this should be an object with a ``map`` method that
            will be used to parallelize the computation. For example, this can
            be a :class:`multiprocessing.Pool` or a
            :class:`multiprocessing.pool.ThreadPool`.

        Returns
        -------
        results : TransitPeriodogramResults
            The periodogram results as a :class:`TransitPeriodogramResults`
            object.

        Raises
        ------
        ValueError
            If ``oversample`` is not an integer greater than 0 or if
            ``objective`` or ``method`` are not valid.

        """
        period, duration = self._validate_period_and_duration(period, duration)

        # Check for absurdities in the ``oversample`` choice
        try:
            oversample = int(oversample)
        except TypeError:
            raise ValueError("oversample must be an int, got {0}"
                             .format(oversample))
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
        allowed_methods = ["fast", "slow"]
        if method not in allowed_methods:
            raise ValueError(("Unrecognized method '{0}'\n"
                              "allowed methods are: {1}")
                             .format(method, allowed_methods))

        # Format and check the input arrays
        t = np.ascontiguousarray(strip_units(self.t), dtype=np.float64)
        y = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            ivar = np.ones_like(y)
        else:
            ivar = 1.0 / np.ascontiguousarray(strip_units(self.dy),
                                              dtype=np.float64)**2

        # Make sure that the period and duration arrays are C-order
        period_fmt = np.ascontiguousarray(strip_units(period),
                                          dtype=np.float64)
        duration = np.ascontiguousarray(strip_units(duration),
                                        dtype=np.float64)

        # Select the correct implementation for the chosen method
        if method == "fast":
            transit_periodogram = methods.transit_periodogram_fast
        else:
            transit_periodogram = methods.transit_periodogram_slow

        # Run the implementation
        results = transit_periodogram(
            t, y - np.median(y), ivar, period_fmt, duration,
            oversample, use_likelihood, pool)

        return self._format_results(objective, period, results)

    def model(self, t_model, period, duration, transit_time):
        """Compute the transit model at the given period, duration, and phase

        Parameters
        ----------
        t_model : array-like or Quantity
            Times at which to compute the model.
        period : float or Quantity
            The period of the transits.
        duration : float or Quantity
            The duration of the transit.
        transit_time : float or Quantity
            The mid-transit time of a reference transit.

        Returns
        -------
        y_model : array-like or Quantity
            The model evaluated at the times ``t_model`` with units of ``y``.

        """
        period, duration = self._validate_period_and_duration(period, duration)
        transit_time = validate_unit_consistency(self.t, transit_time)
        t_model = strip_units(validate_unit_consistency(self.t, t_model))

        period = float(strip_units(period))
        duration = float(strip_units(duration))
        transit_time = float(strip_units(transit_time))

        t = np.ascontiguousarray(strip_units(self.t), dtype=np.float64)
        y = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            ivar = np.ones_like(y)
        else:
            ivar = 1.0 / np.ascontiguousarray(strip_units(self.dy),
                                              dtype=np.float64)**2

        # Compute the depth
        hp = 0.5*period
        m_in = np.abs((t-transit_time+hp) % period - hp) < 0.5*duration
        m_out = ~m_in
        y_in = np.sum(y[m_in] * ivar[m_in]) / np.sum(ivar[m_in])
        y_out = np.sum(y[m_out] * ivar[m_out]) / np.sum(ivar[m_out])

        # Evaluate the model
        y_model = y_out + np.zeros_like(t_model)
        m_model = np.abs((t_model-transit_time+hp) % period-hp) < 0.5*duration
        y_model[m_model] = y_in

        return y_model * self._y_unit()

    def _validate_inputs(self, t, y, dy):
        """Private method used to check the consistency of the inputs

        Parameters
        ----------
        t : array-like or Quantity
            Sequence of observation times.
        y : array-like or Quantity
            Sequence of observations associated with times t.
        dy : float, array-like or Quantity
            Error or sequence of observational errors associated with times t.

        Returns
        -------
        t, y, dy : array-like or Quantity
            The inputs with consistent shapes and units.

        Raises
        ------
        ValueError
            If the dimensions are incompatible or if the units of dy cannot be
            converted to the units of y.

        """
        # Validate shapes of inputs
        if dy is None:
            t, y = np.broadcast_arrays(t, y, subok=True)
        else:
            t, y, dy = np.broadcast_arrays(t, y, dy, subok=True)
        if t.ndim != 1:
            raise ValueError("Inputs (t, y, dy) must be 1-dimensional")

        # validate units of inputs if any is a Quantity
        if dy is not None:
            dy = validate_unit_consistency(y, dy)

        return t, y, dy

    def _validate_duration(self, duration):
        """Private method used to check a set of test durations

        Parameters
        ----------
        duration : float, array-like or Quantity
            The set of durations that will be considered.

        Returns
        -------
        duration : array-like or Quantity
            The input reformatted with the correct shape and units.

        Raises
        ------
        ValueError
            If the units of duration cannot be converted to the units of t.

        """
        duration = np.atleast_1d(np.abs(duration))
        if duration.ndim != 1 or duration.size == 0:
            raise ValueError("duration must be 1-dimensional")
        return validate_unit_consistency(self.t, duration)

    def _validate_period_and_duration(self, period, duration):
        """Private method used to check a set of periods and durations

        Parameters
        ----------
        period : float, array-like or Quantity
            The set of test periods.
        duration : float, array-like or Quantity
            The set of durations that will be considered.

        Returns
        -------
        period, duration : array-like or Quantity
            The inputs reformatted with the correct shapes and units.

        Raises
        ------
        ValueError
            If the units of period or duration cannot be converted to the
            units of t.

        """
        duration = self._validate_duration(duration)
        period = np.atleast_1d(np.abs(period))
        if period.ndim != 1 or period.size == 0:
            raise ValueError("period must be 1-dimensional")
        period = validate_unit_consistency(self.t, period)

        if not np.min(period) > np.max(duration):
            raise ValueError("The maximum transit duration must be shorter "
                             "than the minimum period")

        return period, duration

    def _format_results(self, objective, period, results):
        """A private method used to wrap and add units to the periodogram

        Parameters
        ----------
        objective : string
            The name of the objective used in the optimization.
        period : array-like or Quantity
            The set of trial periods.
        results : tuple
            The output of one of the periodogram implementations.

        """
        (power, depth, depth_err, transit_time, duration, depth_snr,
         log_likelihood) = results

        if has_units(self.t):
            transit_time = units.Quantity(transit_time, unit=self.t.unit)
            duration = units.Quantity(duration, unit=self.t.unit)

        if has_units(self.y):
            depth = units.Quantity(depth, unit=self.y.unit)
            depth_err = units.Quantity(depth_err, unit=self.y.unit)

            depth_snr = units.Quantity(depth_snr, unit=units.one)

            if self.dy is None:
                if objective == "likelihood":
                    power = units.Quantity(power, unit=self.y.unit**2)
                else:
                    power = units.Quantity(power, unit=units.one)
                log_likelihood = units.Quantity(log_likelihood,
                                                unit=self.y.unit**2)
            else:
                power = units.Quantity(power, unit=units.one)
                log_likelihood = units.Quantity(log_likelihood, unit=units.one)

        return TransitPeriodogramResults(objective, period, power, depth,
                                         depth_err, transit_time, duration,
                                         depth_snr, log_likelihood)

    def _t_unit(self):
        if has_units(self.t):
            return self.t.unit
        else:
            return 1

    def _y_unit(self):
        if has_units(self.y):
            return self.y.unit
        else:
            return 1


class TransitPeriodogramResults(dict):
    """The results of a TransitPeriodogram search

    Attributes
    ----------
    objective : string
        The scalar used to optimize to find the best fit phase, duration, and
        depth. See :func:`TransitPeriodogram.power` for more information.
    period : array-like or Quantity
        The set of test periods.
    power : array-like or Quantity
        The periodogram evaluated at the periods in ``period``. If
        ``objective`` is:
        * ``'likelihood'``: the values of ``power`` are the
          log likelihood maximized over phase, depth, and duration, or
        * ``'snr'``: the values of ``power`` are the signal-to-noise with
          which the depth is measured maximized over phase, depth, and
          duration.
    depth : array-like or Quantity
        The estimated depth of the maximum power model at each period.
    depth_err : array-like or Quantity
        The 1-sigma uncertainty on ``depth``.
    duration : array-like or Quantity
        The maximum power duration at each period.
    transit_time : array-like or Quantity
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like or Quantity
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like or Quantity
        The log likelihood of the maximum power model.

    """
    def __init__(self, *args):
        super(TransitPeriodogramResults, self).__init__(zip(
            ("objective", "period", "power", "depth", "depth_err",
             "duration", "transit_time", "depth_snr", "log_likelihood"),
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

    def assert_allclose(self, other):
        for k, v in self.items():
            if k not in other:
                raise AssertionError("missing key '{0}'".format(k))
            if k == "objective":
                assert v == other[k], (
                    "Mismatched objectives. Expected '{0}', got '{1}'"
                    .format(v, other[k])
                )
                continue
            assert_quantity_allclose(v, other[k],
                                     err_msg="Mismatch in attribute '{0}'"
                                     .format(k))
