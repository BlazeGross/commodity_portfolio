import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_engine.analytics import (
    backtest,
    calculate_risk,
    default_shocks,
    empirical_tail,
    evaluate_limits,
    exposures,
    stress_test,
)
from risk_engine.market import default_portfolio


@pytest.fixture
def sample():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.multivariate_normal([0.0001, -0.0002], [[0.0001, 0.00004], [0.00004, 0.0004]], 800),
        columns=["FX", "GAS"],
        index=pd.bdate_range("2022-01-01", periods=800),
    )
    return returns, pd.Series({"FX": 250_000.0, "GAS": -80_000.0})


def test_units_and_short_exposure():
    portfolio = default_portfolio()
    result = exposures(
        portfolio, pd.Series({"EURUSD=X": 1.1, "GBPUSD=X": 1.3, "NG=F": 3.0, "XLU": 80})
    )
    assert result.to_dict() == {
        "EURUSD=X": 275_000.0,
        "GBPUSD=X": -130_000.0,
        "NG=F": 240_000.0,
        "XLU": 160_000.0,
    }


@pytest.mark.parametrize("confidence", [0.75, 0.85])
def test_fractional_tail(confidence):
    var, weights = empirical_tail(np.array([1.0, 2.0, 5.0, 10.0, 10.0]), confidence)
    assert weights.sum() == pytest.approx(1.0)
    # At 75% the tail contains 1.25 observations, both tied at 10.
    assert float(weights @ np.array([1.0, 2.0, 5.0, 10.0, 10.0])) == 10.0
    assert var == 10.0


def test_fractional_boundary_without_ties():
    losses = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    var, weights = empirical_tail(losses, 0.75)
    assert var == 3.0
    assert weights @ losses == pytest.approx((10 + 0.25 * 3) / 1.25)


def test_tied_tail_attribution_is_permutation_invariant():
    losses = np.array([10.0, 10.0, 0.0, -1.0])
    _, weights = empirical_tail(losses, 0.9)
    assert weights.tolist() == pytest.approx([0.5, 0.5, 0, 0])


@pytest.mark.parametrize("method", ["Historical", "Normal", "Monte Carlo"])
@pytest.mark.parametrize("horizon", [1, 10])
def test_additive_es_and_ordering(sample, method, horizon):
    returns, exposure = sample
    result = calculate_risk(returns, exposure, 0.975, horizon, method)
    assert result.component_es.sum() == pytest.approx(result.es)
    assert result.es >= result.var
    if result.component_var is not None:
        assert result.component_var.sum() == pytest.approx(result.var)


def test_normal_formula(sample):
    returns, exposure = sample
    result = calculate_risk(returns, exposure, 0.99, 5, "Normal")
    sigma = np.sqrt(exposure @ returns.cov() @ exposure * 5)
    mean_loss = -returns.mean() @ exposure * 5
    assert result.var == pytest.approx(mean_loss + norm.ppf(0.99) * sigma)
    assert result.es == pytest.approx(mean_loss + norm.pdf(norm.ppf(0.99)) / 0.01 * sigma)


def test_monte_carlo_reproducible_and_close_to_normal(sample):
    returns, exposure = sample
    normal = calculate_risk(returns, exposure, method="Normal")
    first = calculate_risk(returns, exposure, method="Monte Carlo", simulations=100_000)
    second = calculate_risk(returns, exposure, method="Monte Carlo", simulations=100_000)
    np.testing.assert_array_equal(first.losses, second.losses)
    assert first.var == pytest.approx(normal.var, rel=0.03)
    assert first.es == pytest.approx(normal.es, rel=0.03)


def test_historical_horizon_is_rolling_sum(sample):
    returns, exposure = sample
    result = calculate_risk(returns, exposure, horizon=5)
    expected = -(returns.rolling(5).sum().dropna() * exposure).sum(axis=1)
    np.testing.assert_allclose(result.losses, expected)
    assert result.observations == len(returns) - 4


@pytest.mark.parametrize("method", ["Historical", "Normal", "Monte Carlo"])
def test_zero_and_single_asset(sample, method):
    returns, exposure = sample
    result = calculate_risk(returns, exposure * 0, method=method)
    assert result.var == 0
    assert result.es == 0
    single = calculate_risk(returns[["FX"]], exposure[["FX"]], method=method)
    assert np.isfinite(single.es)


@pytest.mark.parametrize("method", ["Historical", "Normal", "Monte Carlo"])
def test_perfect_hedge_has_zero_risk(sample, method):
    returns, _ = sample
    same_returns = pd.DataFrame({"A": returns["FX"], "B": returns["FX"]})
    result = calculate_risk(
        same_returns, pd.Series({"A": 100_000.0, "B": -100_000.0}), method=method
    )
    assert result.var == pytest.approx(0.0, abs=0.001)
    assert result.es == pytest.approx(0.0, abs=0.001)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 1},
        {"confidence": 0.5},
        {"confidence": np.nan},
        {"horizon": 0},
        {"horizon": 1.2},
        {"method": "invalid"},
        {"method": "Monte Carlo", "simulations": 10},
    ],
)
def test_invalid_risk_inputs(sample, kwargs):
    with pytest.raises(ValueError):
        calculate_risk(*sample, **kwargs)


def test_exposure_alignment_and_invalid_inputs(sample):
    returns, exposure = sample
    first = calculate_risk(returns, exposure)
    second = calculate_risk(returns.iloc[:, ::-1], exposure)
    assert first.es == second.es
    with pytest.raises(ValueError):
        calculate_risk(returns, pd.Series({"MISSING": 10}))
    with pytest.raises(ValueError):
        exposures(default_portfolio(), pd.Series({"EURUSD=X": 1.1}))
    bad = default_portfolio()
    bad.loc[0, "multiplier"] = -1
    with pytest.raises(ValueError):
        exposures(bad, pd.Series({ticker: 1 for ticker in bad["ticker"]}))


def test_stress_sign_and_additivity():
    exposure = pd.Series({"FX": 100.0, "GAS": -200.0})
    shocks = pd.DataFrame({"GAS": [0.5], "FX": [-0.1]}, index=["Crisis"])
    summary, components = stress_test(exposure, shocks)
    assert summary.loc["Crisis", "PnL USD"] == -110
    assert summary.loc["Crisis", "Loss USD"] == 110
    assert components.sum(axis=1).iloc[0] == -110
    with pytest.raises(ValueError):
        stress_test(exposure, shocks[["FX"]])
    with pytest.raises(ValueError):
        stress_test(exposure, shocks * 20)
    assert set(default_shocks(default_portfolio()).columns) == set(default_portfolio()["ticker"])


def test_limit_boundaries():
    names = ["low", "warn", "equal", "breach", "negative"]
    values = dict(zip(names, [79, 80, 100, 101, -10]))
    limits = dict.fromkeys(names, 100)
    result = evaluate_limits(values, limits)
    assert result["Status"].tolist() == ["OK", "WARNING", "WARNING", "BREACH", "OK"]
    assert result["Utilization"].iloc[-1] == 0
    with pytest.raises(ValueError):
        evaluate_limits({"x": 1}, {"x": 0})
    with pytest.raises(ValueError):
        evaluate_limits({"x": np.nan}, {"x": 10})


def test_backtest_no_lookahead_and_kupiec(sample):
    returns, exposure = sample
    frame, stats = backtest(returns, exposure, window=60)
    losses = -(returns * exposure).sum(axis=1)
    assert frame.iloc[0]["VaR USD"] == np.quantile(losses.iloc[:60], 0.99, method="inverted_cdf")
    altered = returns.copy()
    altered.iloc[60] *= 1_000
    altered_frame, _ = backtest(altered, exposure, window=60)
    assert frame.iloc[0]["VaR USD"] == altered_frame.iloc[0]["VaR USD"]
    assert stats["observations"] == len(returns) - 60
    assert stats["breaches"] == frame["Breach"].sum()
    assert 0 <= stats["kupiec_p"] <= 1


def test_backtest_no_breaches_is_finite():
    returns = pd.DataFrame({"A": np.zeros(100)})
    frame, stats = backtest(returns, pd.Series({"A": 100.0}), window=60)
    assert not frame["Breach"].any()
    assert np.isfinite(stats["kupiec_lr"])
    assert np.isfinite(stats["kupiec_p"])
    with pytest.raises(ValueError):
        backtest(returns, pd.Series({"A": 1.0}), window=100)
