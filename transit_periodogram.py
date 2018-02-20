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
    """Compute the Transit Periodogram

    This method is a commonly used tool for discovering transiting exoplanets
    or eclipsing binaries in photometric time series datasets. This
    implementation is based on the "box least squares (BLS)" method described
    in [1]_ with added support for observational uncertainties,
    parallelization, and a likelihood model.

    Parameters
    ----------
    t : array-like or Quantity
        Sequence of observation times
    y : array-like or Quantity
        Sequence of observations associated with times t
    dy : float, array-like or Quantity, optional
        Error or sequence of observational errors associated with times t

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

    def _time_unit(self):
        if has_units(self.t):
            return self.t.unit
        else:
            return 1

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
        duration : array-like or Quantity
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
            if has_units(self.t):
                minimum_period = units.Quantity(minimum_period)
                try:
                    minimum_period = units.Quantity(minimum_period,
                                                    unit=self.t.unit)
                except units.UnitConversionError:
                    raise ValueError("Units of minimum_period not equivalent "
                                     "to units of t")
            else:
                if has_units(minimum_period):
                    raise ValueError("minimum_period has units while t "
                                     "doesn't.")

        # If no maximum period is provided, choose one by requiring that
        # all signals with at least minimum_n_transit should be detectable.
        if maximum_period is None:
            maximum_period = baseline / minimum_n_transit
        else:
            if has_units(self.t):
                maximum_period = units.Quantity(maximum_period)
                try:
                    maximum_period = units.Quantity(maximum_period,
                                                    unit=self.t.unit)
                except units.UnitConversionError:
                    raise ValueError("Units of maximum_period not equivalent "
                                     "to units of t")
            else:
                if has_units(maximum_period):
                    raise ValueError("maximum_period has units while t "
                                     "doesn't.")

        # Convert bounds to frequency
        minimum_frequency = 1.0/strip_units(maximum_period)
        maximum_frequency = 1.0/strip_units(minimum_period)

        # Compute the number of frequencies and the frequency grid
        nf = 1 + int(np.round((maximum_frequency - minimum_frequency)/df))
        return 1.0/(maximum_frequency-df*np.arange(nf)) * self._time_unit()

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
        """Compute the periodogram for a set of periods

        Parameters
        ----------
        period : array-like or Quantity
            The periods where the power should be computed
        duration : array-like or Quantity
            The set of durations to test
        objective : {'likelihood', 'snr'}, optional
            The scalar that should be optimized to find the best fit phase,
            duration, and depth. This can be either ``'likelihood'`` (default)
            to optimize the log-likelihood of the model, or ``'snr'`` to
            optimize the signal-to-noise with which the transit depth is
            measured.
        method : {'fast', 'fast_python', 'slow'}, optional
            The computational method used to compute the periodogram. This is
            mainly included for the purposes of testing and most users will
            want to use the optimized ``'fast'`` method (default) that is
            implemented in Cython. The ``'fast_python'`` method is an
            implementation of the ``'fast'`` method in pure Python and
            ``'slow'`` is a brute-force method that is used to test the
            results of the other methods.
        oversample : int, optional
            The number of bins per duration that should be used. This sets the
            time resolution of the phase fit with larger values of
            ``oversample`` yielding a finer grid and higher computational cost.

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
