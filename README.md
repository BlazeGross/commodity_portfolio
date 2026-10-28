# Projekt 2. Commodity Portfolio Risk Engine

Interaktywna aplikacja Python + Streamlit do analizy ryzyka portfela FX, gazu
i ekspozycji zastępczej na sektor energii. Waluta bazowa: **USD**.
Rzeczywiste notowania pochodzą z **Yahoo Finance**, przez `yfinance`.

## Uruchomienie na Windows

W PowerShell, w katalogu projektu:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Aplikacja jest dostępna pod adresem http://localhost:8501.
Środowisko `.venv` zostało już przygotowane w tym projekcie; do ponownego
uruchomienia wystarczy ostatnie polecenie. Nie trzeba aktywować środowiska.
Wymagany Python 3.11+; projekt sprawdzony na Pythonie 3.13.

Domyślne źródło pobiera dane online. Bez internetu wybierz
**Yahoo Finance (zapisany przyklad)**. Projekt zawiera pobrany zestaw
750 wspólnych obserwacji z okresu **2023-10-26 – 2026-10-23**.
To zapis rzeczywistych danych, nie generator danych syntetycznych.
Pochodzenie, datę pobrania i parametry zawiera [metadane](data/yahoo_metadata.json).

Aktualizacja zapisanego przykładu:

```powershell
.\.venv\Scripts\python.exe scripts\download_sample.py
# Opcjonalnie: własny zakres (data końcowa włącznie)
.\.venv\Scripts\python.exe scripts\download_sample.py --start 2022-01-01 --end 2026-10-23
```

Błędy pobierania są widoczne. Nie ma automatycznego przełączania na inne dane.
Yahoo może ograniczyć liczbę zapytań, a dostępność i warunki wykorzystania danych
zależą od dostawcy; projekt nie jest licencjonowanym feedem produkcyjnym.

## Przykładowy portfel

| Instrument | Symbol Yahoo | Ilość | Mnożnik | Interpretacja |
|---|---|---:|---:|---|
| EUR/USD | `EURUSD=X` | +250 000 | 1 | Long EUR, cena USD za EUR |
| GBP/USD | `GBPUSD=X` | -100 000 | 1 | Short GBP, cena USD za GBP |
| Henry Hub natural gas | `NG=F` | +8 | 10 000 | Kontrakty futures, USD/MMBtu |
| Utilities Select Sector SPDR | `XLU` | +2 000 | 1 | Jednostki ETF, USD/jednostkę |

**XLU nie jest ceną energii elektrycznej.** To zaakceptowany proxy akcji
amerykańskiego sektora utilities; nie odtwarza ekspozycji na MWh, dostawę godzinową,
polską energię ani kontrakty EEX. `NG=F` reprezentuje Henry Hub, nie TTF.
Do rzeczywistych cen energii służy import własnego CSV.

Ilości i mnożniki można edytować w aplikacji. Dodatnia ilość oznacza pozycję long,
ujemna short. NAV wprowadza się osobno: notional futures nie jest wartością
aktywów netto ani wymaganym depozytem. Nie budujemy bilansu finansowania portfela.

## Funkcje

1. **Przegląd:** historyczny, normalny i Monte Carlo VaR oraz Expected Shortfall,
   wykres strat, ekspozycje i znormalizowane ceny.
2. **Dekompozycja:** kontrybucje do ES dla każdej metody, dekompozycja Eulera
   normalnego VaR, efekt dywersyfikacji i korelacje.
3. **Stress i limity:** edytowalne hipotetyczne szoki, PnL per instrument,
   limity VaR, ES, ekspozycji brutto, straty stresowej i koncentracji.
   `WARNING` od 80%, `BREACH` powyżej 100% limitu; równość nie jest naruszeniem.
4. **Backtest:** jednodniowy kroczący historyczny VaR, przekroczenia i test
   pokrycia Kupca, bez używania danych z przyszłości.
5. **Dane i eksport:** ceny, portfel i miary w CSV, raport JSON z parametrami
   i ograniczeniami. Osobne eksporty stresu, limitów i backtestu.

Wybór poziomu ufności: 95%, 97,5%, 99%. Horyzont: 1, 5, 10 lub 20 obserwacji.
Limity VaR/ES dotyczą wybranego horyzontu i metody historycznej. Backtest zawsze
dotyczy jednego dnia i jest wyraźnie oddzielony od horyzontu globalnego.

## Metodyka

Dla instrumentu `i`:

```text
e_i = quantity_i × multiplier_i × latest_price_i
L_t = -sum(e_i × r_t,i)
r_t,i = price_t,i / price_(t-1),i - 1
```

`L > 0` oznacza stratę. Bieżące ekspozycje są zamrożone we wszystkich scenariuszach.
To liniowy model ryzyka, nie odtworzenie historycznego rachunku inwestycyjnego.

### Historyczny VaR i ES

- VaR: empiryczny kwantyl `alpha`, NumPy `method="inverted_cdf"`.
- ES: średnia najgorszej masy prawdopodobieństwa `1-alpha`.
  Jeśli `N × (1-alpha)` nie jest całkowite, obserwacja graniczna ma częściową wagę.
  Przy remisach graniczna masa rozdzielana jest równo.
- Kontrybucje ES to straty poszczególnych instrumentów ważone tymi samymi wagami
  ogona; sumują się do ES portfela.
- Dla horyzontu `H` używamy **kroczącej sumy H stóp zwrotu**, bez składania:
  jest to liniowe przybliżenie przy zamrożonych ekspozycjach. Okna nakładają się.

### Model normalny

Niech `mu` będzie wektorem średnich, `Sigma` macierzą kowariancji z próby,
`e` wektorem ekspozycji, a `z = Phi^-1(alpha)`:

```text
m = -H × e' mu
s = sqrt(H × e' Sigma e)
VaR = m + z × s
ES = m + phi(z) / (1-alpha) × s
component_VaR_i = -H × e_i × mu_i + z × H × e_i × (Sigma e)_i / s
```

Dekompozycja ES jest analogiczna, z `phi(z)/(1-alpha)` zamiast `z`.
Dla zerowej zmienności część zmiennościowa kontrybucji jest zerowa.
Efekt dywersyfikacji: suma samodzielnych normalnych VaR minus normalny VaR portfela.

### Monte Carlo

30 000 skorelowanych scenariuszy z wielowymiarowego rozkładu normalnego,
średnia `H × mu`, kowariancja `H × Sigma`, ziarno `42`.
VaR i ES wyznaczamy jak dla próbki empirycznej. Wyniki są odtwarzalne.
Model normalny oraz MC zakładają niezależność zmian w czasie przy skalowaniu
horyzontu i **nie modelują grubych ogonów ani reżimów rynkowych**.

### Stress testy i limity

`PnL_i = e_i × shock_i`. Szok jest jednorazowy, niezależny od horyzontu VaR.
Scenariusze kryzysu energetycznego, recesji i słabego dolara są **hipotetyczne**,
nie historycznymi rekonstrukcjami. Dla własnych instrumentów początkowe scenariusze
to symetryczne +/-20%; należy je dostosować. Spadek poniżej -100% jest odrzucany.
Koncentracja to `max(abs(e_i)) / sum(abs(e_i))`, nie udział w NAV.

### Backtest

Okno estymacji ma 60, 125 lub 250 obserwacji. Prognoza dla dnia `t` korzysta
wyłącznie ze strat do `t-1`. Przekroczenie: `loss_t > VaR_t`.
Test Kupca sprawdza bezwarunkową częstość przekroczeń; `p < 0.05` sugeruje
odrzucenie poprawnego pokrycia. Jest asymptotyczny, nie bada niezależności,
a krótka próba i dyskretność kwantyli ograniczają jego moc.
Używamy dzisiejszych, stałych ekspozycji, nie rzeczywistych historycznych pozycji.

VaR i ES nie są obcinane do zera: mogą być ujemne, jeżeli odpowiedni ogon
obejmuje zyski. Zachowuje to addytywność kontrybucji.

## Import własnych danych

Wybierz **Import CSV** i prześlij plik UTF-8 z przecinkiem jako separatorem:

- pierwsza kolumna `Date`, daty najlepiej `YYYY-MM-DD`, jeden wiersz na dzień;
- pozostałe nagłówki to unikalne symbole instrumentów;
- ceny dodatnie, skończone, **w USD za jednostkę**;
- minimum 61 wspólnych dni cen; dla backtestu potrzeba więcej danych;
- plik [zapisanych cen](data/yahoo_prices.csv) jest działającym przykładem formatu.

Można zastąpić `XLU` kolumną `POWER_USD` z własną serią cen energii
przeliczoną do USD/MWh. Nowy instrument pojawi się jako `Custom` z zerową ilością:
ustaw ilość w MWh i mnożnik 1 albo ilość kontraktów i właściwy mnożnik.
To **samodzielna ekspozycja USD**, nie automatyczne odwzorowanie kontraktu
w EUR oraz jego dodatkowego ryzyka EUR/USD. Nie importujemy portfela z CSV;
portfel edytuje się w aplikacji.

Daty są sortowane. Dni z brakami usuwa się wspólnie, bez forward-fill,
a ich liczbę pokazuje panel. Zwrot przez lukę może obejmować kilka sesji.
Horyzont jest liczbą obserwacji, nie dni kalendarzowych.
W pobraniu online stosujemy ceny skorygowane `auto_adjust=True`.

## Struktura i użycie biblioteki

```text
app.py                       interfejs i eksport
risk_engine/market.py         instrumenty, Yahoo, walidacja CSV
risk_engine/analytics.py      VaR, ES, kontrybucje, stress, limity, backtest
scripts/download_sample.py   pobranie rzeczywistego przykładu na dysk
data/                        ceny, portfel i metadane przykładu Yahoo
tests/                       testy numeryczne, danych i aplikacji Streamlit
.streamlit/config.toml       motyw i ustawienia serwera
```

Przykład bez uruchamiania interfejsu:

```python
from pathlib import Path
from risk_engine.market import read_prices_csv, default_portfolio
from risk_engine.analytics import daily_returns, exposures, calculate_risk

prices, dropped = read_prices_csv(Path("data", "yahoo_prices.csv").read_bytes())
exposure = exposures(default_portfolio(), prices.iloc[-1])
result = calculate_risk(
    daily_returns(prices), exposure, confidence=0.99, horizon=1, method="Historical"
)
print(result.var, result.es)
print(result.component_es)
```

## Testy

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Testy jednostkowe używają kontrolowanych próbek i atrap Yahoo, nie wymagają
internetu. Obejmują formuły, znak i jednostki ekspozycji, ES z wagą graniczną,
remisy, addytywność, horyzont, Monte Carlo, pozycje short, walidację cen i CSV,
limity, stress, test Kupca oraz przejście przez wszystkie widoki aplikacji.
Pobieranie rzeczywistych danych sprawdza osobno skrypt zapisu przykładu.

## Granice zastosowania

To projekt demonstracyjny. Nie obejmuje ryzyka bazowego, krzywych forward,
wolumenu i profilu godzinowego, opcji i greków, pełnej wyceny dostaw fizycznych,
margin calls, płynności, ryzyka kontrahenta ani kosztów finansowania.
Rolowana seria `NG=F` może zawierać skoki związane ze zmianą kontraktu.
XLU jest wyłącznie equity proxy, a nie modelem rynku energii.
Dane ujemne są świadomie odrzucane: ten model procentowych zmian nie nadaje się
do instrumentów dopuszczających ceny zerowe/ujemne.
Przed zastosowaniem produkcyjnym potrzebne są odpowiednie feedy, modele kontraktów,
kalibracja, niezależna walidacja i kontrola licencji danych.
