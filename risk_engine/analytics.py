"""Linear, fixed-current-exposure risk in USD; losses have a positive sign."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from risk_engine.market import validate_prices

Method = Literal["Historical", "Normal", "Monte Carlo"]


@dataclass(frozen=True)
class RiskResult:
    method: str
    confidence: float
    horizon: int
    var: float
    es: float
    component_var: pd.Series | None
    component_es: pd.Series
    losses: np.ndarray | None
    observations: int


def exposures(portfolio: pd.DataFrame, last_prices: pd.Series) -> pd.Series:
    required = {"ticker", "quantity", "multiplier"}
    if not required.issubset(portfolio.columns) or portfolio.empty:
        raise ValueError("Portfel wymaga ticker, quantity i multiplier.")
    if portfolio["ticker"].duplicated().any() or portfolio["ticker"].isna().any():
        raise ValueError("Instrumenty portfela musza byc unikalne i niepuste.")
    positions = portfolio.set_index("ticker")
    missing = positions.index.difference(last_prices.index)
    if not missing.empty:
        raise ValueError(f"Brak cen instrumentow: {', '.join(missing)}.")
    quantities = positions["quantity"].to_numpy(dtype=float)
    multipliers = positions["multiplier"].to_numpy(dtype=float)
    prices = last_prices.reindex(positions.index).to_numpy(dtype=float)
    if (
        not np.isfinite(quantities).all()
        or not np.isfinite(multipliers).all()
        or not np.isfinite(prices).all()
        or (multipliers <= 0).any()
        or (prices <= 0).any()
    ):
        raise ValueError("Ilosci musza byc skonczone, ceny i mnozniki dodatnie i skonczone.")
    return pd.Series(quantities * multipliers * prices, index=positions.index, name="USD")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    clean = validate_prices(prices)
    return clean.pct_change(fill_method=None).iloc[1:]


def _validate_inputs(
    returns: pd.DataFrame, exposure: pd.Series, confidence: float, horizon: int
) -> pd.DataFrame:
    if not np.isfinite(confidence) or not 0.5 < confidence < 1:
        raise ValueError("Poziom ufnosci musi nalezec do (0.5, 1).")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("Horyzont musi byc dodatnia liczba calkowita.")
    if (
        returns.empty
        or exposure.empty
        or returns.columns.duplicated().any()
        or exposure.index.duplicated().any()
        or not exposure.index.isin(returns.columns).all()
    ):
        raise ValueError("Brak zwrotow lub niezgodne/nieunikalne instrumenty.")
    selected = returns.loc[:, exposure.index]
    if len(selected) < max(2, horizon) or not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Za malo obserwacji lub niefinitywne stopy zwrotu.")
    if not np.isfinite(exposure.to_numpy(dtype=float)).all():
        raise ValueError("Ekspozycje musza byc skonczone.")
    return selected


def empirical_tail(losses: np.ndarray, confidence: float) -> tuple[float, np.ndarray]:
    """Exact empirical ES, including fractional mass at the tail boundary."""
    if (
        losses.ndim != 1
        or len(losses) == 0
        or not np.isfinite(losses).all()
        or not 0.5 < confidence < 1
    ):
        raise ValueError("Niepoprawny rozklad strat lub poziom ufnosci.")
    var = float(np.quantile(losses, confidence, method="inverted_cdf"))
    mass = len(losses) * (1 - confidence)
    order = np.argsort(-losses, kind="stable")
    ranks = np.arange(len(losses), dtype=float)
    allocation = np.clip(mass - ranks, 0.0, 1.0)
    # Share boundary mass equally across ties to keep attribution permutation-invariant.
    positive = np.flatnonzero(allocation > 0)
    boundary = losses[order[positive[-1]]]
    ties = losses[order] == boundary
    allocation[ties] = allocation[ties].sum() / ties.sum()
    weights = np.empty(len(losses), dtype=float)
    weights[order] = allocation / mass
    return var, weights


def calculate_risk(
    returns: pd.DataFrame,
    exposure: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    method: Method = "Historical",
    simulations: int = 30_000,
    seed: int = 42,
) -> RiskResult:
    selected = _validate_inputs(returns, exposure, confidence, horizon)
    values = selected.to_numpy(dtype=float)
    e = exposure.to_numpy(dtype=float)
    if method == "Normal":
        mean = values.mean(axis=0) * horizon
        covariance = np.atleast_2d(np.cov(values, rowvar=False)) * horizon
        cov_e = covariance @ e
        sigma = float(np.sqrt(max(float(e @ cov_e), 0.0)))
        mean_loss = -mean * e
        gradient_scale = cov_e * e / sigma if sigma > 0 else np.zeros_like(e)
        cvar = mean_loss + norm.ppf(confidence) * gradient_scale
        ces = mean_loss + norm.pdf(norm.ppf(confidence)) / (1 - confidence) * gradient_scale
        return RiskResult(
            method,
            confidence,
            horizon,
            float(cvar.sum()),
            float(ces.sum()),
            pd.Series(cvar, index=exposure.index),
            pd.Series(ces, index=exposure.index),
            None,
            len(values),
        )
    if method == "Historical":
        scenario_returns = selected.rolling(horizon).sum().dropna().to_numpy()
    elif method == "Monte Carlo":
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, (int, np.integer))
            or simulations < 1_000
        ):
            raise ValueError("Monte Carlo wymaga co najmniej 1000 symulacji.")
        generator = np.random.default_rng(seed)
        scenario_returns = generator.multivariate_normal(
            values.mean(axis=0) * horizon,
            np.atleast_2d(np.cov(values, rowvar=False)) * horizon,
            size=simulations,
            check_valid="raise",
        )
    else:
        raise ValueError(f"Nieznana metoda: {method}.")
    component_losses = -scenario_returns * e
    losses = component_losses.sum(axis=1)
    var, weights = empirical_tail(losses, confidence)
    component_es = pd.Series(weights @ component_losses, index=exposure.index, name="ES USD")
    return RiskResult(
        method,
        confidence,
        horizon,
        var,
        float(component_es.sum()),
        None,
        component_es,
        losses,
        len(scenario_returns),
    )


def stress_test(exposure: pd.Series, shocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        exposure.empty
        or exposure.index.duplicated().any()
        or shocks.empty
        or shocks.columns.duplicated().any()
        or shocks.index.duplicated().any()
        or set(shocks.columns) != set(exposure.index)
    ):
        raise ValueError("Scenariusze musza zawierac dokladnie instrumenty portfela.")
    if not np.isfinite(exposure.to_numpy()).all() or not np.isfinite(shocks.to_numpy()).all():
        raise ValueError("Szoki i ekspozycje musza byc skonczone.")
    if (shocks.to_numpy() < -1).any():
        raise ValueError("Szok procentowy nie moze byc mniejszy niz -100%.")
    components = shocks.reindex(columns=exposure.index).mul(exposure, axis=1)
    pnl = components.sum(axis=1)
    summary = pd.DataFrame({"PnL USD": pnl, "Loss USD": -pnl})
    return summary, components


def default_shocks(portfolio: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Kryzys energetyczny (hipotetyczny)": {"FX": -0.10, "Gas": 0.80, "Power proxy": -0.25},
        "Recesja (hipotetyczna)": {"FX": -0.08, "Gas": -0.45, "Power proxy": -0.30},
        "Slaby dolar (hipotetyczny)": {"FX": 0.12, "Gas": 0.15, "Power proxy": 0.08},
    }
    supported = {"FX", "Gas", "Power proxy"}
    if not set(portfolio["asset_class"]).issubset(supported):
        raise ValueError("Brak domyslnych szokow dla tej klasy aktywow.")
    return pd.DataFrame(
        {
            str(row.ticker): {name: shocks[row.asset_class] for name, shocks in scenarios.items()}
            for row in portfolio.itertuples()
        }
    )


def evaluate_limits(metrics: dict[str, float], limits: dict[str, float]) -> pd.DataFrame:
    if metrics.keys() != limits.keys():
        raise ValueError("Kazda miara musi miec odpowiadajacy limit.")
    rows = []
    for name, value in metrics.items():
        limit = limits[name]
        if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0:
            raise ValueError("Miary musza byc skonczone, a limity dodatnie i skonczone.")
        utilization = max(value, 0.0) / limit
        rows.append(
            {
                "Measure": name,
                "Value": value,
                "Limit": limit,
                "Utilization": utilization,
                "Status": "BREACH" if value > limit else "WARNING" if utilization >= 0.8 else "OK",
            }
        )
    return pd.DataFrame(rows)


def backtest(
    returns: pd.DataFrame, exposure: pd.Series, confidence: float = 0.99, window: int = 250
) -> tuple[pd.DataFrame, dict[str, float]]:
    selected = _validate_inputs(returns, exposure, confidence, 1)
    if isinstance(window, bool) or not isinstance(window, int) or window < 20:
        raise ValueError("Okno backtestu musi byc liczba calkowita >= 20.")
    if len(selected) <= window:
        raise ValueError(f"Backtest wymaga wiecej niz {window} stop zwrotu.")
    losses = -selected.mul(exposure, axis=1).sum(axis=1)
    predictions = (
        losses.rolling(window)
        .apply(
            lambda sample: float(np.quantile(sample, confidence, method="inverted_cdf")), raw=True
        )
        .shift(1)
    )
    frame = pd.DataFrame({"Loss USD": losses, "VaR USD": predictions}).dropna()
    frame["Breach"] = frame["Loss USD"] > frame["VaR USD"]
    n = len(frame)
    count = int(frame["Breach"].sum())
    p = 1 - confidence
    rate = count / n
    null_ll = xlogy(count, p) + xlogy(n - count, 1 - p)
    fitted_ll = xlogy(count, rate) + xlogy(n - count, 1 - rate)
    lr = max(0.0, float(2 * (fitted_ll - null_ll)))
    return frame, {
        "observations": float(n),
        "breaches": float(count),
        "expected": n * p,
        "rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": float(chi2.sf(lr, df=1)),
    }
