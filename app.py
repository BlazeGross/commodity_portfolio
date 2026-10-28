import json
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from risk_engine.analytics import (
    RiskResult,
    backtest,
    calculate_risk,
    daily_returns,
    default_shocks,
    evaluate_limits,
    exposures,
    stress_test,
)
from risk_engine.market import (
    MarketDataError,
    default_portfolio,
    download_prices,
    read_prices_csv,
)

st.set_page_config(
    page_title="Commodity Portfolio | Risk Engine",
    page_icon=":material/monitoring:",
    layout="wide",
)


@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def yahoo_prices(start: str, end: str) -> tuple[pd.DataFrame, int]:
    return download_prices(start, end)


@st.cache_data(max_entries=12, show_spinner=False)
def risk_report(
    returns: pd.DataFrame, exposure: pd.Series, confidence: float, horizon: int
) -> dict[str, RiskResult]:
    return {
        method: calculate_risk(returns, exposure, confidence, horizon, method)
        for method in ("Historical", "Normal", "Monte Carlo")
    }


@st.cache_data(max_entries=12, show_spinner=False)
def rolling_backtest(
    returns: pd.DataFrame, exposure: pd.Series, confidence: float, window: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    return backtest(returns, exposure, confidence, window)


def csv_bytes(frame: pd.DataFrame, index: bool = True) -> bytes:
    return frame.to_csv(index=index).encode("utf-8-sig")


def money(value: float) -> str:
    return f"${value:,.0f}"


st.caption("PROJEKT 02 / MARKET RISK ANALYTICS")
st.title("Commodity portfolio risk engine")
st.write("FX, gaz i sektor energii. Jeden portfel, przejrzyste zrodla ryzyka.")
st.info(
    "XLU to ETF amerykanskiego sektora utilities, nie cena energii elektrycznej. "
    "NG=F to rolowana seria futures Henry Hub, nie europejski TTF. "
    "Projekt edukacyjny, nie system produkcyjny ani rekomendacja inwestycyjna.",
    icon=":material/info:",
)

with st.sidebar:
    st.header("Parametry analizy")
    source = st.selectbox(
        "Zrodlo cen", ["Yahoo Finance", "Import CSV", "Yahoo Finance (zapisany przyklad)"]
    )
    with st.form("market_settings"):
        start = st.date_input("Od", value=date.today() - timedelta(days=3 * 365))
        end = st.date_input("Do (wlacznie)", value=date.today())
        load = st.form_submit_button("Pobierz / wczytaj dane", width="stretch")
    upload = st.file_uploader("Ceny w CSV", type=["csv"]) if source == "Import CSV" else None
    confidence = st.select_slider("Poziom ufnosci", options=[0.95, 0.975, 0.99], value=0.99)
    horizon = st.select_slider("Horyzont (dni obserwacyjne)", options=[1, 5, 10, 20], value=1)
    nav = st.number_input("Kapital referencyjny NAV (USD)", min_value=1.0, value=1_000_000.0)
    st.caption("NAV jest podawany osobno. Notional futures nie jest wartoscia aktywow netto.")

if source == "Yahoo Finance":
    if start >= end:
        st.error("Data poczatkowa musi poprzedzac date koncowa.")
        st.stop()
    try:
        if load:
            yahoo_prices.clear()
        with st.spinner("Pobieranie notowan Yahoo Finance..."):
            prices, dropped = yahoo_prices(start.isoformat(), (end + timedelta(days=1)).isoformat())
    except MarketDataError as exc:
        st.error(str(exc))
        st.info(
            "Ponow pobieranie przyciskiem w panelu lub wybierz Import CSV. Brak danych zastepczych."
        )
        st.stop()
elif source == "Import CSV":
    st.caption(
        "CSV: Date,EURUSD=X,GBPUSD=X,NG=F,XLU. Mozesz uzyc wlasnych symboli, "
        "np. POWER_USD. Wszystkie ceny musza byc w USD za jednostke."
    )
    if upload is None:
        st.info("Wybierz plik CSV w panelu bocznym (minimum 61 wspolnych dni cen).")
        st.stop()
    try:
        prices, dropped = read_prices_csv(upload.getvalue())
    except MarketDataError as exc:
        st.error(str(exc))
        st.stop()
else:
    snapshot = Path(__file__).resolve().parent / "data" / "yahoo_prices.csv"
    if not snapshot.is_file():
        st.error(
            "Brak zapisanego przykladu. Uruchom: "
            ".\\.venv\\Scripts\\python.exe scripts\\download_sample.py"
        )
        st.stop()
    try:
        prices, dropped = read_prices_csv(snapshot.read_bytes())
    except (OSError, MarketDataError) as exc:
        st.error(f"Nie mozna odczytac zapisanego przykladu: {exc}")
        st.stop()
    st.info(
        "Zapisane rzeczywiste notowania Yahoo. Zakres dat w panelu dotyczy tylko pobierania online."
    )

st.caption(
    f"Zrodlo: {source} | {prices.index.min():%Y-%m-%d} - {prices.index.max():%Y-%m-%d} "
    f"| {len(prices):,} wspolnych obserwacji | waluta bazowa: USD"
)
if dropped:
    st.warning(
        f"Usunieto {dropped} dni z brakujacymi cenami. Nie uzupelniamy brakow poprzednia cena. "
        "Zwroty przez luki moga obejmowac wiecej niz jedna sesje."
    )
if (pd.Timestamp(date.today()) - prices.index.max()).days > 5:
    st.warning("Ostatnia cena ma ponad 5 dni. Wyniki bazuja na tej dacie, nie na cenach biezacych.")

with st.expander("Portfel i jednostki", expanded=False):
    st.caption(
        "Ilosc dodatnia = long, ujemna = short. EUR/GBP w jednostkach waluty, "
        "NG=F w kontraktach (10 000 MMBtu), XLU w jednostkach ETF. "
        "Dla wlasnego instrumentu ustaw mnoznik; szoki wprowadzisz osobno."
    )
    base = default_portfolio().set_index("ticker").reindex(prices.columns)
    for column in ("name", "asset_class", "unit"):
        base[column] = base[column].fillna(
            pd.Series(prices.columns, index=prices.columns) if column == "name" else "Custom"
        )
    base["quantity"] = base["quantity"].fillna(0.0)
    base["multiplier"] = base["multiplier"].fillna(1.0)
    base = base.rename_axis("ticker").reset_index()
    portfolio = st.data_editor(
        base,
        disabled=["ticker", "name", "asset_class", "unit"],
        hide_index=True,
        key=f"portfolio_{','.join(prices.columns)}",
        alt="Portfel: edytowalne ilosci i mnozniki instrumentow",
    )

try:
    exposure = exposures(portfolio, prices.iloc[-1])
    returns = daily_returns(prices)
    with st.spinner("Obliczanie miar ryzyka..."):
        report = risk_report(returns, exposure, confidence, horizon)
except (ValueError, TypeError) as exc:
    st.error(f"Nie mozna obliczyc ryzyka: {exc}")
    st.stop()

historical = report["Historical"]
gross = float(exposure.abs().sum())
net = float(exposure.sum())
if gross == 0:
    st.warning("Portfel ma zerowe ekspozycje. Ustaw ilosci, aby otrzymac niezerowe miary ryzyka.")
if historical.observations * (1 - confidence) < 10:
    st.warning(
        "Mniej niz 10 obserwacji w ogonie historycznym. Estymacja ES/VaR jest niestabilna; "
        "wydluz historie lub obniz poziom ufnosci."
    )
with st.container(horizontal=True):
    st.metric("Ekspozycja brutto", money(gross), border=True)
    st.metric("Historyczny VaR", money(historical.var), border=True)
    st.metric("Expected Shortfall", money(historical.es), border=True)
    st.metric("ES / NAV", f"{historical.es / nav:.2%}", border=True)

section = st.segmented_control(
    "Obszar analizy",
    ["Przeglad", "Dekompozycja", "Stress i limity", "Backtest", "Dane i eksport"],
    default="Przeglad",
    selection_mode="single",
)

summary = pd.DataFrame(
    [
        {"Metoda": method, "VaR USD": result.var, "ES USD": result.es}
        for method, result in report.items()
    ]
).set_index("Metoda")

if section == "Przeglad":
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Profil ryzyka")
        st.dataframe(summary.style.format("{:,.2f}"), alt="VaR i ES dla trzech metod")
        st.bar_chart(summary, alt="Porownanie VaR i ES wedlug metody")
    with right:
        st.subheader("Ekspozycje USD")
        st.bar_chart(exposure, horizontal=True, alt="Ekspozycja netto kazdego instrumentu w USD")
        st.caption(f"Ekspozycja netto: {money(net)} | Kapital referencyjny: {money(nav)}")
    st.subheader("Rozklad strat historycznych")
    distribution = pd.DataFrame({"Loss USD": historical.losses})
    histogram = (
        alt.Chart(distribution)
        .mark_bar(color="#20C9A6")
        .encode(
            x=alt.X("Loss USD:Q", bin=alt.Bin(maxbins=60), title="Strata USD (ujemna = zysk)"),
            y=alt.Y("count():Q", title="Liczba scenariuszy"),
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"USD": [historical.var, historical.es], "Miara": ["VaR", "ES"]}))
        .mark_rule(strokeWidth=2)
        .encode(x="USD:Q", color="Miara:N", tooltip=["Miara", alt.Tooltip("USD:Q", format=",.2f")])
    )
    st.altair_chart(histogram + rules, alt="Historyczne straty z progami VaR i ES")
    st.subheader("Ceny indeksowane do 100")
    st.line_chart(prices.div(prices.iloc[0]).mul(100), alt="Porownanie przebiegu cen od poczatku")

elif section == "Dekompozycja":
    st.subheader("Kontrybucja do Expected Shortfall")
    attribution = pd.DataFrame({method: result.component_es for method, result in report.items()})
    st.bar_chart(attribution, stack=False, alt="Kontrybucje instrumentow do ES w USD")
    st.dataframe(attribution.style.format("{:,.2f}"), alt="Dekompozycja ES w USD")
    st.caption(
        "Kontrybucje sumuja sie do ES portfela. Ujemna kontrybucja oznacza efekt zabezpieczenia. "
        "Historyczna i Monte Carlo: wazone straty w ogonie. Normalna: alokacja Eulera."
    )
    normal = report["Normal"]
    st.subheader("Parametryczna dekompozycja VaR")
    st.dataframe(normal.component_var, alt="Kontrybucje Eulera do normalnego VaR")
    standalone = pd.Series(
        {
            ticker: calculate_risk(
                returns[[ticker]], exposure[[ticker]], confidence, horizon, "Normal"
            ).var
            for ticker in exposure.index
        },
        name="Standalone VaR USD",
    )
    st.metric("Efekt dywersyfikacji (normalny VaR)", money(float(standalone.sum() - normal.var)))
    st.caption("Suma samodzielnych VaR minus VaR portfela, przy tych samych parametrach.")
    st.subheader("Korelacja dziennych stop zwrotu")
    st.dataframe(
        returns.corr().style.format("{:.2f}"),
        alt="Macierz korelacji instrumentow",
    )

elif section == "Stress i limity":
    st.subheader("Scenariusze hipotetyczne")
    st.caption(
        "Szoki jednorazowe, niezalezne od horyzontu VaR. Wartosci w procentach "
        "(np. -30 = spadek o 30%). Nie sa rekonstrukcja historycznych kryzysow."
    )
    if "Custom" in set(portfolio["asset_class"]):
        initial_shocks = pd.DataFrame(
            [-0.20 * np.ones(len(exposure)), 0.20 * np.ones(len(exposure))],
            index=["Spadek 20%", "Wzrost 20%"],
            columns=exposure.index,
        )
    else:
        initial_shocks = default_shocks(portfolio)
    shocks = st.data_editor(
        initial_shocks * 100,
        key=f"shocks_{','.join(prices.columns)}",
        alt="Edytowalne szoki procentowe dla instrumentow i scenariuszy",
    )
    try:
        stress, components = stress_test(exposure, shocks / 100)
    except (ValueError, TypeError) as exc:
        st.error(f"Niepoprawne scenariusze: {exc}")
        st.stop()
    st.bar_chart(stress[["PnL USD"]], horizontal=True, alt="PnL scenariuszy stresowych")
    st.dataframe(components.style.format("{:,.2f}"), alt="PnL stresowy wedlug instrumentu")
    st.subheader("Limity ryzyka")
    st.caption("Limity VaR i ES dotycza metody historycznej i wybranego horyzontu.")
    metrics = {
        "Historical VaR USD": historical.var,
        "Historical ES USD": historical.es,
        "Gross exposure USD": gross,
        "Worst stress loss USD": max(0.0, float(stress["Loss USD"].max())),
        "Max concentration %": float(exposure.abs().max() / gross * 100) if gross else 0.0,
    }
    defaults = [75_000.0, 100_000.0, 1_500_000.0, 250_000.0, 60.0]
    limit_input = st.data_editor(
        pd.DataFrame({"Measure": list(metrics), "Limit": defaults}),
        disabled=["Measure"],
        hide_index=True,
        key="limits",
        alt="Limity kwotowe i koncentracji",
    )
    try:
        limits = dict(zip(limit_input["Measure"], limit_input["Limit"], strict=True))
        limit_report = evaluate_limits(metrics, limits)
    except (ValueError, TypeError) as exc:
        st.error(f"Niepoprawne limity: {exc}")
        st.stop()
    st.dataframe(limit_report, hide_index=True, alt="Wykorzystanie limitow i status naruszenia")
    breaches = int((limit_report["Status"] == "BREACH").sum())
    if breaches:
        st.error(f"Naruszone limity: {breaches}.")
    else:
        st.success("Brak naruszen limitow.")
    st.caption("WARNING od 80%; BREACH powyzej 100%. Koncentracja = max |ekspozycja| / brutto.")
    with st.container(horizontal=True):
        st.download_button("Eksport stress CSV", csv_bytes(stress), "stress.csv", "text/csv")
        st.download_button(
            "Eksport limitow CSV", csv_bytes(limit_report, False), "limits.csv", "text/csv"
        )

elif section == "Backtest":
    st.subheader("Kroczacy backtest jednodniowego VaR")
    st.caption(
        "Historyczny VaR jest estymowany wylacznie z poprzednich obserwacji. "
        "Stale dzisiejsze ekspozycje: diagnostyka modelu, nie rzeczywisty historyczny PnL. "
        "Ten panel zawsze uzywa horyzontu 1, niezaleznie od ustawienia globalnego."
    )
    window = st.select_slider("Okno estymacji (obserwacje)", [60, 125, 250], value=250)
    if len(returns) <= window:
        st.warning(f"Potrzeba wiecej niz {window} stop zwrotu. Wydluz zakres lub zmniejsz okno.")
    else:
        result, stats = rolling_backtest(returns, exposure, confidence, window)
        with st.container(horizontal=True):
            st.metric("Przekroczenia", f"{stats['breaches']:.0f} / {stats['observations']:.0f}")
            st.metric("Oczekiwane", f"{stats['expected']:.1f}")
            st.metric("Kupiec p-value", f"{stats['kupiec_p']:.4f}")
        st.line_chart(result[["Loss USD", "VaR USD"]], alt="Straty i prognozy jednodniowego VaR")
        st.dataframe(result.loc[result["Breach"]], alt="Dni przekroczenia prognozowanego VaR")
        st.caption(
            "Test Kupca: H0 = poprawna czestosc przekroczen; p < 0.05 sugeruje odrzucenie H0. "
            "To test asymptotyczny, nie test niezaleznosci. Krotka proba i dyskretne kwantyle "
            "ograniczaja jego interpretacje."
        )
        st.download_button("Eksport backtestu", csv_bytes(result), "backtest.csv", "text/csv")

else:
    st.subheader("Dane i raport")
    st.dataframe(prices.tail(20), alt="Ostatnie 20 wspolnych dni cen")
    st.dataframe(
        portfolio.set_index("ticker").assign(price=prices.iloc[-1], exposure_usd=exposure),
        alt="Portfel, ostatnie ceny i ekspozycje w USD",
    )
    payload = {
        "source": source,
        "as_of": str(prices.index.max().date()),
        "currency": "USD",
        "confidence": confidence,
        "horizon": horizon,
        "nav": nav,
        "price_observations": len(prices),
        "dropped_rows": dropped,
        "exposure_usd": exposure.to_dict(),
        "portfolio": portfolio.to_dict(orient="records"),
        "risk": summary.reset_index().to_dict(orient="records"),
        "seed": 42,
        "simulations": 30_000,
        "limitations": [
            "XLU is a utilities equity proxy",
            "NG=F is a rolled futures series",
            "Linear fixed-exposure risk, not actual account PnL",
        ],
    }
    with st.container(horizontal=True):
        st.download_button("Ceny CSV", csv_bytes(prices), "prices.csv", "text/csv")
        st.download_button("Portfel CSV", csv_bytes(portfolio, False), "portfolio.csv", "text/csv")
        st.download_button("Miary ryzyka CSV", csv_bytes(summary), "risk.csv", "text/csv")
        st.download_button(
            "Raport JSON",
            json.dumps(payload, indent=2, allow_nan=False),
            "risk_report.json",
            "application/json",
        )

with st.expander("Metodyka i ograniczenia"):
    st.markdown(
        """
        - **Ekspozycja USD** = ilosc x mnoznik x ostatnia cena. FX jest kwotowany USD/EUR
          i USD/GBP; NG=F USD/MMBtu; XLU USD/jednostke ETF.
        - **Strata** = minus suma ekspozycji x stopy zwrotu. Model liniowy bez opcji,
          depozytow zabezpieczajacych, kosztow transakcji i finansowania.
        - **Historyczny VaR**: kwantyl empiryczny (inverse CDF).
          **ES**: srednia najgorszych 1-alpha strat, z czesciowa waga obserwacji granicznej.
        - **Normalny**: srednia i macierz kowariancji z proby.
          **Monte Carlo**: 30 000 skorelowanych scenariuszy normalnych, seed 42.
          Monte Carlo nie jest modelem grubych ogonow.
        - **Horyzont H**: historycznie kroczaca suma H dziennych stop zwrotu;
          normalny/MC: srednia x H, kowariancja x H (niezaleznosc w czasie).
          To liniowe, zamrozone ekspozycje, nie zlozona stopa zwrotu.
        - Dni z brakami sa usuwane wspolnie, bez forward-fill; horyzont liczy obserwacje,
          nie dni kalendarzowe. Serie Yahoo sa korygowane (auto_adjust=True).
        - Rolowania NG=F moga zaburzac stopy zwrotu. Brak krzywych forward, ryzyka bazy,
          profilu godzinowego, plynnosci i pelnej wyceny fizycznych kontraktow energii.
        - VaR/ES moga byc ujemne, gdy caly odpowiedni ogon oznacza zyski.
          Nie obcinamy ich do zera, aby zachowac addytywnosc dekompozycji.
        """
    )
