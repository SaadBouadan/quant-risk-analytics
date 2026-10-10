# Methodology

This document defines every number QuantRisk Analytics reports. The dashboard's
*Methodology* tab renders this same file, so the documentation and the
application cannot drift apart. All statistics are **historical descriptions of
the selected sample**. They are not forecasts, and past results do not guarantee
future performance.

---

## 1. Prices and returns

**Adjusted prices.** The analysis expects *adjusted closing prices*: prices
corrected for splits and dividends. A return computed from adjusted prices
approximates a **total return** (price change plus reinvested distributions).
The Yahoo Finance adapter requests adjusted prices explicitly
(`auto_adjust=True`). Uploaded files are assumed to contain adjusted prices, and
the user is responsible for that. Mixing adjusted and unadjusted series would
make dividend-paying assets look worse than they were.

**Simple returns.** For each asset, the periodic return is

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

Simple (arithmetic) returns are used instead of log returns because they
**aggregate across assets**: a portfolio's return is the weighted sum of its
assets' simple returns, $R_{p,t} = \sum_i w_{i,t}\, r_{i,t}$. Log returns add
up across *time* but not across *assets*.

## 2. Calendar, missing data and annualisation

**Trading-day convention.** Daily data on weekdays is annualised with
**252 trading days per year**. The frequency is detected from the median gap
between dates:

| Data | Periods per year $P$ |
|---|---|
| Daily, weekdays only (equities, ETFs) | 252 |
| Daily including weekends (e.g. crypto) | 365 |
| Weekly / monthly / quarterly / annual | 52 / 12 / 4 / 1 |

The detected value can be overridden in the sidebar.

**Missing observations and holidays.** Missing prices are **never silently
replaced**:

1. *Different start dates.* The sample is trimmed to the window where every
   selected asset (and the benchmark) has prices. Earlier data is **not
   back-filled**, and the trim is reported.
2. *Gaps inside the window* (e.g. a holiday on one exchange but not another).
   - **Drop (default):** any date on which at least one selected asset has no
     price is removed. The next return then spans the gap
     ($P_{t+1}/P_{t-1} - 1$). Cumulative performance is therefore exact; only
     the number of observations falls slightly.
   - **Forward-fill (opt-in):** the last price is carried forward for at most
     *N* consecutive dates (default 3). Longer gaps are still dropped. A
     forward-filled date shows a 0% return for that asset, which slightly
     understates volatility. The number of filled cells is reported.
3. *Invalid values.* Non-numeric, zero, negative or infinite prices are
   rejected with an error naming the column and date.

Weekends are not "missing". They are simply not trading days.

## 3. Portfolio construction

**Weights.** The initial release supports **long-only** portfolios.
Negative, non-finite or all-zero weights are rejected. Weights that do not sum
to 100% are rescaled proportionally ($w_i / \sum_j w_j$), and the app says so.
Zero-weight assets are excluded.

**Rebalancing assumption.** For each period, with start-of-period weights
$w_t$:

$$
R_{p,t} = w_t^\top r_t, \qquad w^{\text{drift}}_{t} = \frac{w_t \odot (1 + r_t)}{1 + R_{p,t}}
$$

The next period starts with either the target weights (if $t$ closes a
rebalancing period) or the drifted weights.

| Rule | Behaviour |
|---|---|
| Daily | Reset to target every close (constant-mix). This is the textbook formula, but it implies daily trading. |
| Monthly / Quarterly / Annually | Weights drift; reset at the close of the last trading day of each calendar period. |
| Never (buy-and-hold) | Weights drift for the whole sample. |

The decision to rebalance at the close of $t$ uses only information known at
that close. **Turnover** at each rebalance is reported as one-way turnover
$\tfrac12 \sum_i |w_i^{\text{target}} - w_{i,t}^{\text{drift}}|$. Transaction
costs, taxes, bid–ask spreads and cash drag are **ignored**.

## 4. Performance statistics

Let $N$ be the number of return periods and $P$ the periods per year.

| Statistic | Definition |
|---|---|
| Cumulative return | $R_{\text{cum}} = \prod_{t=1}^{N} (1 + r_t) - 1$ |
| Annualised return (CAGR) | $(1 + R_{\text{cum}})^{P/N} - 1$ |
| Annualised volatility | $s \sqrt{P}$, with $s$ the sample standard deviation ($N-1$ denominator) |
| Sharpe ratio | $\dfrac{\overline{r - r_f}}{s(r - r_f)}\sqrt{P}$ |
| Maximum drawdown | $\min_t \left( \dfrac{V_t}{\max_{u \le t} V_u} - 1 \right)$, with $V_0 = 1$ |

**Cumulative vs annualised return.** The cumulative return is the total gain
over the whole sample: +100% means the money doubled. The annualised return
is the *constant yearly rate* that would produce the same cumulative result. It
makes samples of different lengths comparable. Doubling over 10 years is a
cumulative return of 100% but an annualised return of only 7.18%
($2^{1/10} - 1$). Annualising less than one year of data extrapolates a short
history, and the app warns when this happens.

**Why returns compound.** +10% followed by −10% is a cumulative return of
−1%, not 0% ($1.1 \times 0.9 = 0.99$). This is why the arithmetic average of
returns overstates the growth rate that was actually achieved.

**Volatility scaling** by $\sqrt{P}$ assumes returns are independent across
periods. Autocorrelated returns (common in illiquid assets) make this an
understatement.

**Risk-free rate convention.** The user enters an *annual effective* rate
$r_f$. It is converted to a per-period rate by compounding,
$r_{f,p} = (1 + r_f)^{1/P} - 1$, and subtracted from each period's return.
The Sharpe ratio is the ratio of the **arithmetic** mean excess return to its
standard deviation, scaled by $\sqrt{P}$. This is the standard Sharpe (1994)
convention. An alternative, $(\text{CAGR} - r_f)/\sigma$, gives slightly lower
values for volatile assets. A constant rate is a simplification, because real
T-bill yields changed substantially over most historical samples. If
volatility is zero, the Sharpe ratio is **undefined** and reported as "n/a",
never as infinity.

**Drawdown statistics.** The initial investment counts as the first peak, so
a loss in the first period is a drawdown. The app reports the peak, trough and
recovery dates of the maximum drawdown, the peak-to-trough and
peak-to-recovery lengths, and the **longest underwater period** (the longest
run of consecutive periods below a previous high). Durations are counted in
trading periods.

**Best / worst period.** These are the single best and worst period returns
and the best and worst *calendar months* (returns compounded within each month).
Partial months at the edges of the sample are included as-is.

**Benchmark-relative statistics.** Portfolio and benchmark are aligned on
identical dates. With active return $a_t = r_{p,t} - r_{b,t}$:

| Statistic | Definition |
|---|---|
| Excess cumulative / annualised return | difference of the respective returns |
| Tracking error | $s(a)\sqrt{P}$ |
| Information ratio | $\bar a P \,/\, \text{TE}$ |
| Beta | $\operatorname{cov}(r_p, r_b) / \operatorname{var}(r_b)$ |
| Jensen's alpha (annual) | $\left[\overline{r_p - r_f} - \beta\, \overline{r_b - r_f}\right] P$ |

## 5. Value at Risk and Expected Shortfall

**Sign convention.** VaR and ES are reported as **positive numbers that
represent a loss, as a fraction of portfolio value, over one return period**
(one trading day for daily data). "1-day 95% VaR = 1.20%" means that *on 95%
of the days in the sample, the portfolio lost no more than 1.20% of its value*.
The corresponding return quantile is −1.20%. Multiplying by the portfolio value
gives a monetary amount.

**Historical (non-parametric) VaR.** Sort the $n$ returns from worst to best
and let

$$
k = \left\lceil n\,(1 - c) \right\rceil
$$

be the number of observations in the loss tail at confidence $c$. Then

$$
\text{VaR}_c = -\,r_{(k)}, \qquad \text{ES}_c = -\frac{1}{k} \sum_{j=1}^{k} r_{(j)}
$$

where $r_{(j)}$ is the $j$-th smallest return. This is the empirical
("inverted CDF") quantile. No interpolation is used, so VaR is always an
observed loss and $\text{ES}_c \ge \text{VaR}_c$ always holds. The product
$n(1-c)$ is rounded before taking the ceiling, so floating-point noise
($100 \times 0.05 = 5.000000000000004$) cannot add a spurious sixth observation.

**Expected Shortfall** (Conditional VaR) answers the question VaR leaves
open: *when* a bad day happens, how bad is it on average? ES is a coherent
risk measure and is used in the Basel FRTB framework.

**Worked example** (this exact case is a unit test). There are 20 daily returns (%):

`0.5, −1.2, 0.3, 2.1, −0.4, −3.0, 0.8, 1.1, −0.7, 0.2, −2.2, 0.9, 1.5, −0.1, 0.6, −1.8, 0.4, 1.0, −0.9, 0.7`

- Sorted worst first: −3.0, −2.2, −1.8, −1.2, −0.9, …
- 90% confidence: $k = \lceil 20 \times 0.10 \rceil = 2$, so **VaR = 2.2%** (the second-worst loss) and
  **ES = (3.0 + 2.2) / 2 = 2.6%**.
- 95% confidence: $k = \lceil 20 \times 0.05 \rceil = 1$, so **VaR = ES = 3.0%**.
- On a 1,000,000 portfolio, 90% VaR = **22,000** and 90% ES = **26,000** (one day).

**Gaussian VaR (comparison only).** $\text{VaR}^{N}_c = -(\hat\mu + z_{1-c}\,\hat\sigma)$
and $\text{ES}^{N}_c = -\left(\hat\mu - \hat\sigma\,\varphi(z_{1-c})/(1-c)\right)$.
When historical VaR exceeds the Gaussian figure at high confidence, the
returns have **fatter tails** than a normal distribution.

**Horizon.** All VaR/ES figures are for **one period**. The app does not scale
them to 10 days with $\sqrt{10}$, because that rule assumes independent,
identically distributed returns.

### Limitations of historical VaR

- **It depends completely on the sample.** A calm look-back window produces a
  low VaR right before a crisis. Events larger than anything in the sample are
  invisible to it.
- **Few tail observations.** At 99% with one year of daily data, VaR rests on
  only 3 observations ($\lceil 252 \times 0.01 \rceil$). The app warns when fewer than 10
  observations define the tail.
- **VaR says nothing about losses beyond the threshold.** This is why ES is
  reported alongside it.
- **Equal weighting of history.** A loss from eight years ago counts as much as
  yesterday's. Volatility clustering is ignored. (Filtered historical simulation
  or EWMA weighting would address this. See the roadmap in the README.)
- **It is not a forecast.** VaR describes the past distribution under the
  assumption that the future resembles it.

## 6. Rolling statistics and look-ahead bias

Rolling volatility, rolling Sharpe and rolling VaR use **trailing windows
only**. The value plotted on date $t$ uses returns from $t-w+1$ to $t$, never
later data. The first $w-1$ dates are left blank instead of being computed on a
shorter window. Rolling Sharpe is shown only when the sample is at least twice
the window, because short-window Sharpe ratios are dominated by noise.

**VaR backtest.** For each date $t$, VaR is estimated from the $w$ returns
ending at $t-1$, and an **exception** is recorded if the realised loss on $t$
exceeds it. Because the forecast is lagged, it never sees the return it is
tested on. A well-calibrated 95% VaR should be exceeded on about 5% of days.
The **Kupiec (1995) proportion-of-failures test** checks this formally:

$$
\text{LR} = -2 \ln\!\left[(1-p)^{n-x} p^{x}\right] + 2 \ln\!\left[(1-\hat p)^{n-x} \hat p^{x}\right] \sim \chi^2_1
$$

where $p = 1-c$, $x$ is the number of exceptions in $n$ forecasts and
$\hat p = x/n$. A p-value below 5% means the exception rate is statistically
inconsistent with the stated confidence level. The test checks frequency only.
It does not check whether exceptions cluster together.

## 7. Diversification and concentration

| Statistic | Definition | Reading |
|---|---|---|
| Correlation matrix | Pearson correlation of aligned periodic returns | +1 = moves together, 0 = unrelated, −1 = hedge |
| Herfindahl index (HHI) | $\sum_i w_i^2$ | 1 = one asset; $1/N$ = equally weighted |
| Effective number of assets | $1 / \text{HHI}$ | "how many equal positions this resembles" |
| Diversification ratio | $\sum_i w_i \sigma_i \,/\, \sigma_p$ | 1 = no diversification benefit; higher is better |
| Risk contribution | $RC_i = w_i (\Sigma w)_i / \sigma_p$, so $\sum_i RC_i = \sigma_p$ | share of volatility each asset is responsible for |

Risk contributions (Euler decomposition) often reveal that a "balanced"
portfolio is dominated by its equity risk. An asset with a small or negative
contribution is acting as a hedge. These figures use the target weights and the
full-sample covariance matrix. They are exact for constant-mix portfolios and
approximate when weights drift.

## 8. Risk across market periods

- **Calendar years.** Return, volatility, Sharpe ratio, within-year maximum
  drawdown, historical VaR and the worst period for each year. Partial years
  are flagged.
- **Volatility regimes.** Each date is labelled *Calm* (< 15%), *Normal*
  (15–25%) or *Stressed* (≥ 25%) from the benchmark's annualised volatility over
  the **previous** 21 trading days (or the portfolio's own volatility if no
  benchmark is selected). The thresholds are fixed numbers rather than sample
  quantiles, and the volatility is lagged by one day. Both choices avoid
  look-ahead. Regime days are not contiguous, so returns within a regime are
  annualised arithmetically ($\bar r \times P$).

## 9. What this tool is not

QuantRisk Analytics describes what happened in a historical sample under
explicit assumptions. It does **not** forecast returns, does not optimise
portfolios and does not provide investment advice. Sample estimates are subject
to estimation error, which is large for mean returns. A Sharpe ratio estimated
from 10 years of data has a standard error of roughly $\sqrt{1/10} \approx 0.3$.
