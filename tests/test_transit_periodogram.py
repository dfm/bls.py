# -*- coding: utf-8 -*-

import pytest
import numpy as np
from numpy.testing import assert_allclose

from .... import units
from ....tests.helper import assert_quantity_allclose
from .. import TransitPeriodogram
from ...lombscargle.core import has_units


@pytest.fixture
def data():
    rand = np.random.RandomState(123)
    t = rand.uniform(0, 10, 500)
    y = np.ones_like(t)
    dy = rand.uniform(0.005, 0.01, len(t))
    period = 2.0
    transit_time = 0.5
    duration = 0.16
    depth = 0.1
    m = np.abs((t-transit_time+0.5*period) % period-0.5*period) < 0.5*duration
    y[m] = 1.0 - depth
    y += dy * rand.randn(len(t))
    return t, y, dy, dict(period=period, transit_time=transit_time,
                          duration=duration, depth=depth)


@pytest.mark.parametrize("objective", ["likelihood", "snr"])
def test_correct_model(data, objective):
    t, y, dy, params = data
    model = TransitPeriodogram(t, y, dy)
    periods = np.exp(np.linspace(np.log(params["period"]) - 0.1,
                                 np.log(params["period"]) + 0.1, 1000))
    results = model.power(periods, params["duration"], objective=objective)
    ind = np.argmax(results.power)
    for k, v in params.items():
        assert_allclose(results[k][ind], v, atol=0.01)
    chi = (results.depth[ind]-params["depth"]) / results.depth_err[ind]
    assert np.abs(chi) < 1


@pytest.mark.parametrize("objective", ["likelihood", "snr"])
def test_fast_method(data, objective):
    t, y, dy, params = data
    model = TransitPeriodogram(t, y, dy)
    periods = np.exp(np.linspace(np.log(params["period"]) - 1,
                                 np.log(params["period"]) + 1, 10))
    results = model.power(periods, params["duration"], objective=objective)
    results.assert_allclose(model.power(periods, params["duration"],
                                        method="slow", objective=objective))


def test_input_units(data):
    t, y, dy, params = data

    t_unit = units.day
    y_unit = units.mag

    with pytest.raises(units.UnitConversionError):
        TransitPeriodogram(t * t_unit, y * y_unit, dy * units.one)
    with pytest.raises(units.UnitConversionError):
        TransitPeriodogram(t * t_unit, y * units.one, dy * y_unit)
    with pytest.raises(units.UnitConversionError):
        TransitPeriodogram(t * t_unit, y, dy * y_unit)
    model = TransitPeriodogram(t*t_unit, y * units.one, dy)
    assert model.dy.unit == model.y.unit
    model = TransitPeriodogram(t*t_unit, y * y_unit, dy)
    assert model.dy.unit == model.y.unit
    model = TransitPeriodogram(t*t_unit, y*y_unit)
    assert model.dy is None


def test_period_units(data):
    t, y, dy, params = data
    t_unit = units.day
    y_unit = units.mag
    model = TransitPeriodogram(t * t_unit, y * y_unit, dy)

    p = model.autoperiod(params["duration"])
    assert p.unit == t_unit
    p = model.autoperiod(params["duration"] * 24 * units.hour)
    assert p.unit == t_unit
    with pytest.raises(units.UnitConversionError):
        model.autoperiod(params["duration"] * units.mag)

    p = model.autoperiod(params["duration"], minimum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(units.UnitConversionError):
        p = model.autoperiod(params["duration"], minimum_period=0.5*units.mag)

    p = model.autoperiod(params["duration"], maximum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(units.UnitConversionError):
        p = model.autoperiod(params["duration"], maximum_period=0.5*units.mag)

    p = model.autoperiod(params["duration"], minimum_period=0.5,
                         maximum_period=1.5)
    p2 = model.autoperiod(params["duration"], maximum_period=0.5,
                          minimum_period=1.5)
    assert_quantity_allclose(p, p2)


@pytest.mark.parametrize("method", ["fast", "slow"])
def test_results_units(data, method):
    t, y, dy, params = data
    t_unit = units.day
    y_unit = units.mag

    periods = np.linspace(params["period"]-1.0, params["period"]+1.0, 3)

    model = TransitPeriodogram(t * t_unit, y * y_unit, dy)
    results = model.power(periods, params["duration"], method=method)
    assert results.period.unit == t_unit
    assert results.power.unit == units.one
    assert results.depth.unit == y_unit
    assert results.depth_err.unit == y_unit
    assert results.transit_time.unit == t_unit
    assert results.duration.unit == t_unit
    assert results.depth_snr.unit == units.one
    assert results.log_likelihood.unit == units.one

    model = TransitPeriodogram(t * t_unit, y, dy)
    results = model.power(periods, params["duration"], method=method)
    assert results.period.unit == t_unit
    assert not has_units(results.power)
    assert not has_units(results.depth)
    assert not has_units(results.depth_err)
    assert results.transit_time.unit == t_unit
    assert results.duration.unit == t_unit
    assert not has_units(results.depth_snr)
    assert not has_units(results.log_likelihood)

    model = TransitPeriodogram(t, y * y_unit, dy)
    results = model.power(periods, params["duration"])
    assert not has_units(results.period)
    assert results.power.unit == units.one
    assert results.depth.unit == y_unit
    assert results.depth_err.unit == y_unit
    assert not has_units(results.transit_time)
    assert not has_units(results.duration)
    assert results.depth_snr.unit == units.one
    assert results.log_likelihood.unit == units.one

    model = TransitPeriodogram(t * t_unit, y * y_unit)
    results = model.power(periods, params["duration"], method=method)
    assert results.period.unit == t_unit
    assert results.power.unit == y_unit * y_unit
    assert results.depth.unit == y_unit
    assert results.depth_err.unit == y_unit
    assert results.transit_time.unit == t_unit
    assert results.duration.unit == t_unit
    assert results.depth_snr.unit == units.one
    assert results.log_likelihood.unit == y_unit * y_unit

    results = model.power(periods, params["duration"], objective="snr",
                          method=method)
    assert results.power.unit == units.one


def test_autopower(data):
    t, y, dy, params = data
    duration = params["duration"] + np.linspace(-0.1, 0.1, 3)

    model = TransitPeriodogram(t, y, dy)
    period = model.autoperiod(duration)
    results1 = model.power(period, duration)
    results2 = model.autopower(duration)
    results1.assert_allclose(results2)


@pytest.mark.parametrize("with_units", [True, False])
def test_model(data, with_units):
    t, y, dy, params = data

    # Compute the model using linear regression
    A = np.zeros((len(t), 2))
    p = params["period"]
    dt = np.abs((t-params["transit_time"]+0.5*p) % p-0.5*p)
    m_in = dt < 0.5*params["duration"]
    A[~m_in, 0] = 1.0
    A[m_in, 1] = 1.0
    w = np.linalg.solve(np.dot(A.T, A / dy[:, None]**2),
                        np.dot(A.T, y / dy**2))
    model_true = np.dot(A, w)

    if with_units:
        t *= units.day
        y *= units.mag
        dy *= units.mag
        model_true *= units.mag

    # Compute the model using the periodogram
    pgram = TransitPeriodogram(t, y, dy)
    model = pgram.model(t, p, params["duration"], params["transit_time"])

    assert_quantity_allclose(model, model_true)


@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (2, 3)])
def test_shapes(data, shape):
    t, y, dy, params = data
    duration = params["duration"]
    model = TransitPeriodogram(t, y, dy)

    period = np.empty(shape)
    period.flat = np.linspace(params["period"]-1, params["period"]+1,
                              period.size)
    if len(period.shape) > 1:
        with pytest.raises(ValueError):
            results = model.power(period, duration)
    else:
        results = model.power(period, duration)
        for k, v in results.items():
            if k == "objective":
                continue
            assert v.shape == shape
