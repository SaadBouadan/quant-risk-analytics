# Defending the methodology: an interview guide

Plain-language answers to the questions an interviewer is most likely to ask
about this project. The formal definitions are in [METHODOLOGY.md](METHODOLOGY.md).

---

### "Walk me through the pipeline."

Prices → validation → alignment → returns → portfolio → statistics.

1. **Load adjusted closing prices.** They include dividends and splits, so returns are total returns.
2. **Validate the data.** Reject bad dates, duplicates and non-positive prices.
3. **Align the assets** on common dates. Missing prices are dropped by default, never invented.
4. **Compute simple returns** $P_t/P_{t-1} - 1$.
5. **Build the portfolio** under an explicit rebalancing rule.
6. **Compute performance and risk statistics.** Each one is a small pure function with a unit test.

The Streamlit app only displays results. `src/analysis.py` runs the whole
pipeline, and the app, the notebook and the tests all call it.

### "Why simple returns and not log returns?"

Simple returns **add up across assets**: the portfolio return is the weighted
sum of asset returns. Log returns add up **across time** but not across assets.
I need cross-sectional aggregation to build the portfolio, so I use simple
returns. I then compound them through time with $\prod(1+r_t)$.

### "Why 252?"

There are roughly 252 trading days in a year (365 days minus weekends and
about 9 exchange holidays). Annualised volatility is $\sigma_{\text{daily}}\sqrt{252}$. The square
root comes from adding variances of independent daily returns. The app
detects the data frequency, so weekly data uses 52, monthly uses 12, and crypto
(which trades at weekends) uses 365.

### "What's the difference between cumulative and annualised return?"

Cumulative return is the total gain: 100% means the money doubled.
Annualised return (CAGR) is the constant yearly rate that gets you there:
doubling in 10 years is $2^{1/10} - 1 = 7.18\%$ per year. Annualising helps
compare periods of different lengths. Annualising a few months of data is
misleading, and the app warns about it.

### "How do you handle missing data?"

I never fill gaps silently. If an asset starts trading later, the sample starts
when **all** assets have prices, and the app reports this. Gaps inside the
sample, such as holidays on one exchange, are dropped by default. The next return
then covers both days, so the cumulative return stays exactly right.
Forward-filling is available, but only as an explicit, limited and reported
option. It creates artificial 0% returns, which understate volatility.

### "Explain your Sharpe ratio. What risk-free rate?"

Sharpe = average excess return ÷ volatility of excess returns, annualised by
$\sqrt{252}$. The user enters an **annual** risk-free rate. I convert it to a
**daily compounded** rate, $(1+r_f)^{1/252}-1$, and subtract it each day. A
constant rate is a simplification, because real T-bill yields varied a lot over
most samples. If volatility is zero, Sharpe is undefined, so I return "n/a"
rather than infinity.

### "What does a 95% VaR of 1.2% mean?"

On 95% of days in the sample, the portfolio lost **no more than 1.2%** of its
value. It is a **loss threshold, not a worst case**: on the other 5% of days
the loss was larger. I report VaR as a **positive loss**, so the matching
return quantile is −1.2%. Mixing up those two signs is a classic bug.

### "How do you compute historical VaR exactly?"

Sort the returns and take $k = \lceil n(1-c) \rceil$ observations from the
bottom. VaR is the *k*-th worst loss, and ES is the average of the *k* worst
losses. With 20 returns at 90% confidence, $k = 2$: VaR is the second-worst
loss and ES is the average of the two worst. I avoid interpolation, so VaR is
always a loss that actually happened and ES ≥ VaR is guaranteed. One subtle bug
I guard against: in floating point, `100 * (1 - 0.95)` equals `5.000000000000004`,
so a naive `ceil` would give 6 tail observations instead of 5.

### "Why report Expected Shortfall too?"

VaR ignores what happens beyond the threshold. Two portfolios can share the
same VaR while one has much worse tail losses. ES is the average loss *given*
that the VaR threshold was breached. It is also a *coherent* risk measure: it
rewards diversification, which VaR does not always do. Basel's FRTB framework
moved from VaR to ES for this reason.

### "What are the limitations of historical VaR?"

- It assumes the future looks like the sample, so it cannot see a crisis worse than anything in the data.
- A calm look-back window gives a low VaR right before a storm.
- At 99% confidence with one year of data, only about 3 observations define the tail.
- Every observation is weighted equally, so recent volatility is not emphasised.
- It is a one-day figure. Scaling it with $\sqrt{10}$ assumes independent returns.

### "How do you know your VaR is any good?" (backtesting)

Each day, I estimate VaR from the **previous** 252 days only, then check
whether that day's loss exceeded it. A good 95% VaR should be breached about 5%
of the time. The **Kupiec test** checks whether the breach rate is
statistically consistent with 5%. One caveat: Kupiec checks frequency, not
*clustering*, and breaches tend to bunch together in crises. Christoffersen's
independence test is the natural next step.

### "How did you avoid look-ahead bias?"

- Rolling windows are **trailing**: the value on day *t* uses data up to *t* only.
- VaR forecasts used in the backtest are **lagged one day**.
- Market regimes are classified from **yesterday's** trailing volatility, with **fixed**
  thresholds rather than full-sample percentiles, which would use future information.
- Rebalancing decisions at a period's close use only prices known at that close.

Unit tests enforce this. They change future returns and assert that past
rolling values are unchanged.

### "Why does the rebalancing assumption matter?"

The textbook formula $R_p = \sum w_i r_i$ with fixed weights assumes you
rebalance **every day**. With buy-and-hold, winners grow into larger weights,
so a 60/40 portfolio can drift to 75/25 and carry more equity risk than intended.
The app lets you choose the rule and reports the resulting **turnover**.
Transaction costs are not modelled.

### "What's a risk contribution and why show it?"

Volatility does not add up asset by asset, but it can be decomposed: asset
*i* contributes $w_i(\Sigma w)_i/\sigma_p$, and the contributions sum exactly to
total volatility. In the sample portfolio, US equity is 35% of the capital but
roughly two-thirds of the risk. Treasuries contribute about zero, because their
negative correlation with equities offsets their own volatility. This is the
logic behind risk-parity investing.

### "What do beta, tracking error and information ratio tell you?"

- **Beta:** how much the portfolio moves when the benchmark moves 1%.
- **Tracking error:** the volatility of the *difference* between portfolio and benchmark returns.
- **Information ratio:** average outperformance per unit of tracking error.
- **Jensen's alpha:** the return not explained by beta exposure, given the risk-free rate.

### "Why didn't you add machine-learning price predictions?"

Daily returns have a very low signal-to-noise ratio. A forecasting model would
need rigorous out-of-sample testing to be meaningful, and adding one for visual
appeal would undermine the credibility of the risk analysis. The project
deliberately **describes historical risk** and never claims to forecast.

### "How did you test the financial correctness?"

With small, hand-checkable datasets where the right answer is known in
advance. Examples:

- +10% then −10% must give −1%.
- A 20-return VaR example worked out by hand.
- A two-asset rebalancing case solved with fractions.
- Kupiec statistics calculated by hand.
- Cross-checks against an independent reference (`numpy.quantile(method="inverted_cdf")`).
- Property tests, such as ES ≥ VaR and risk contributions summing to 100%.
- Edge cases: zero volatility, a single observation and empty input.

### "What would you add next?"

- EWMA or filtered historical simulation VaR, so recent volatility counts for more.
- A Christoffersen independence test.
- Transaction costs, which turn turnover into a performance drag.
- A time-varying risk-free rate.
- Stress scenarios that replay specific historical windows.
- Bootstrap confidence intervals for Sharpe and VaR.
