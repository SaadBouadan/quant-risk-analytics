"""Plotly figure builders. Presentation only - no analytics are computed here.

Visual conventions:
* The portfolio is always blue and the benchmark always neutral gray, in
  every chart, so identity never depends on chart position.
* A second categorical hue (orange) is used only when two non-benchmark
  quantities are compared (e.g. target vs drifted weights).
* Diverging scales (correlations) run blue <-> red through a gray midpoint.
* Hairline gridlines, no dual axes, legends whenever there are 2+ series.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


@dataclass(frozen=True)
class ChartTheme:
    text: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    surface: str
    portfolio: str
    portfolio_fill: str
    benchmark: str
    accent: str
    loss: str
    diverging: tuple[tuple[float, str], ...]


LIGHT = ChartTheme(
    text="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    surface="#fcfcfb",
    portfolio="#2a78d6",
    portfolio_fill="rgba(42,120,214,0.14)",
    benchmark="#898781",
    accent="#eb6834",
    loss="#e34948",
    diverging=((0.0, "#104281"), (0.25, "#6da7ec"), (0.5, "#f0efec"), (0.75, "#f0a19c"), (1.0, "#b8302f")),
)

DARK = ChartTheme(
    text="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    surface="#1a1a19",
    portfolio="#3987e5",
    portfolio_fill="rgba(57,135,229,0.20)",
    benchmark="#898781",
    accent="#d95926",
    loss="#e66767",
    diverging=((0.0, "#5598e7"), (0.25, "#256abf"), (0.5, "#383835"), (0.75, "#a33b3a"), (1.0, "#e66767")),
)


def theme_for(mode: str | None) -> ChartTheme:
    return DARK if mode == "dark" else LIGHT


def _series_color(name: str, theme: ChartTheme) -> str:
    return theme.benchmark if name == "Benchmark" else theme.portfolio


def _layout(fig: go.Figure, theme: ChartTheme, *, title: str, yformat: str | None = None,
            ytitle: str | None = None, height: int = 380, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        template="none",
        height=height,
        margin=dict(l=8, r=16, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=theme.text_secondary),
        title=dict(text=title, x=0, xanchor="left", y=0.98, font=dict(size=14, color=theme.text)),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=theme.text_secondary)),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=theme.surface, bordercolor=theme.axis, font=dict(family=FONT, color=theme.text)),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=theme.axis, ticks="outside",
                     tickcolor=theme.axis, zeroline=False, color=theme.muted, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=theme.grid, gridwidth=1, zeroline=False, showline=False,
                     tickformat=yformat, title=ytitle, color=theme.muted, automargin=True)
    return fig


def _zero_line(fig: go.Figure, theme: ChartTheme) -> None:
    fig.add_hline(y=0, line=dict(color=theme.axis, width=1))


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def growth_chart(growth: pd.DataFrame, theme: ChartTheme, initial_value: float) -> go.Figure:
    fig = go.Figure()
    for name in growth.columns:
        s = growth[name]
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=name, mode="lines",
            line=dict(color=_series_color(name, theme), width=2),
            hovertemplate="%{y:,.0f}<extra>" + name + "</extra>",
        ))
        fig.add_annotation(x=s.index[-1], y=s.iloc[-1], text=f"{s.iloc[-1]:,.0f}", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=theme.text_secondary, size=11))
    fig.add_hline(y=initial_value, line=dict(color=theme.axis, width=1))
    return _layout(fig, theme, title=f"Growth of {initial_value:,.0f} (dividends reinvested via adjusted prices)",
                   yformat=",.0f")


def drawdown_chart(drawdowns: pd.DataFrame, theme: ChartTheme) -> go.Figure:
    fig = go.Figure()
    for name in reversed(drawdowns.columns):  # draw the benchmark first, portfolio on top
        s = drawdowns[name]
        is_port = name == "Portfolio"
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=name, mode="lines",
            line=dict(color=_series_color(name, theme), width=2 if is_port else 1.5),
            fill="tozeroy" if is_port else None,
            fillcolor=theme.portfolio_fill if is_port else None,
            hovertemplate="%{y:.1%}<extra>" + name + "</extra>",
        ))
    _zero_line(fig, theme)
    fig.update_layout(legend_traceorder="reversed")  # list the portfolio first
    return _layout(fig, theme, title="Drawdown from previous peak", yformat=".0%")


def rolling_chart(frame: pd.DataFrame, theme: ChartTheme, *, title: str, yformat: str,
                  zero_line: bool = False) -> go.Figure:
    fig = go.Figure()
    for name in frame.columns:
        s = frame[name].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=name, mode="lines",
            line=dict(color=_series_color(name, theme), width=2 if name == "Portfolio" else 1.5),
            hovertemplate="%{y:" + yformat + "}<extra>" + name + "</extra>",
        ))
    if zero_line:
        _zero_line(fig, theme)
    return _layout(fig, theme, title=title, yformat=yformat)


def annual_returns_chart(portfolio: pd.Series, benchmark: pd.Series | None, theme: ChartTheme) -> go.Figure:
    fig = go.Figure()
    series = {"Portfolio": portfolio} | ({"Benchmark": benchmark} if benchmark is not None else {})
    for name, s in series.items():
        fig.add_trace(go.Bar(
            x=[str(i) for i in s.index], y=s, name=name, marker=dict(color=_series_color(name, theme)),
            hovertemplate="%{y:.1%}<extra>" + name + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    _zero_line(fig, theme)
    return _layout(fig, theme, title="Calendar-year returns (partial years at the sample edges)",
                   yformat=".0%", hovermode="x")


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def return_distribution_chart(returns: pd.Series, var: float, es: float, confidence: float,
                              theme: ChartTheme) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=returns, nbinsx=80, name="Period returns",
        marker=dict(color=theme.portfolio, line=dict(color=theme.surface, width=1)),
        hovertemplate="Return %{x}<br>Count %{y}<extra></extra>",
    ))
    # ES lies further into the tail than VaR: label ES on its left, VaR on its right, at different heights.
    markers = ((-var, "VaR", theme.text, "dash", "left", 4, 0.98), (-es, "ES", theme.loss, "dot", "right", -4, 0.88))
    for value, label, color, dash, anchor, shift, y in markers:
        fig.add_vline(x=value, line=dict(color=color, width=1.5, dash=dash))
        fig.add_annotation(x=value, y=y, yref="paper", text=f"{label} {confidence:.1%}: {value:.2%}",
                           showarrow=False, xanchor=anchor, yanchor="top", xshift=shift,
                           bgcolor=theme.surface, font=dict(color=theme.text_secondary, size=11))
    fig.update_layout(showlegend=False, bargap=0.05)
    fig = _layout(fig, theme, title="Distribution of portfolio returns with VaR / ES thresholds",
                  yformat=None, ytitle="Number of periods", hovermode="closest")
    fig.update_xaxes(tickformat=".1%")
    return fig


def var_backtest_chart(returns: pd.Series, var_forecast: pd.Series, exceptions: pd.Series,
                       confidence: float, theme: ChartTheme) -> go.Figure:
    realised = returns.loc[var_forecast.index]
    breaches = realised[exceptions]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=realised.index, y=realised, name="Realised return", mode="lines",
        line=dict(color=theme.benchmark, width=1), hovertemplate="%{y:.2%}<extra>Return</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=var_forecast.index, y=-var_forecast, name=f"VaR {confidence:.1%} threshold (prior window)",
        mode="lines", line=dict(color=theme.portfolio, width=2),
        hovertemplate="%{y:.2%}<extra>VaR threshold</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=breaches.index, y=breaches, name=f"Exception ({len(breaches)})", mode="markers",
        marker=dict(color=theme.loss, size=8, line=dict(color=theme.surface, width=2)),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>Exception</extra>",
    ))
    return _layout(fig, theme, title="Rolling historical VaR vs realised returns (out-of-sample)",
                   yformat=".1%", hovermode="closest", height=400)


def correlation_heatmap(corr: pd.DataFrame, theme: ChartTheme) -> go.Figure:
    show_text = len(corr) <= 12
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, zmid=0, colorscale=[list(stop) for stop in theme.diverging],
        text=corr.round(2).to_numpy() if show_text else None,
        texttemplate="%{text:.2f}" if show_text else None,
        textfont=dict(size=11),
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text="ρ"), thickness=12, tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="%{y} vs %{x}<br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig = _layout(fig, theme, title="Correlation of periodic returns", hovermode="closest",
                  height=max(360, 48 * len(corr) + 120))
    fig.update_xaxes(showline=False, ticks="", tickangle=-30)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
def _paired_bars(frame: pd.DataFrame, columns: dict[str, str], theme: ChartTheme, title: str,
                 hover_format: str = ".1%") -> go.Figure:
    frame = frame.sort_values(next(iter(columns)), ascending=True)
    colors = [theme.portfolio, theme.accent]
    fig = go.Figure()
    for (col, label), color in zip(columns.items(), colors, strict=False):
        fig.add_trace(go.Bar(
            y=frame.index, x=frame[col], name=label, orientation="h", marker=dict(color=color),
            hovertemplate="%{x:" + hover_format + "}<extra>" + label + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1)
    fig = _layout(fig, theme, title=title, hovermode="y unified", height=max(320, 44 * len(frame) + 120))
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=theme.grid)
    fig.update_yaxes(showgrid=False)
    fig.add_vline(x=0, line=dict(color=theme.axis, width=1))
    return fig


def allocation_chart(target: pd.Series, end: pd.Series, theme: ChartTheme) -> go.Figure:
    frame = pd.DataFrame({"target": target, "end": end})
    return _paired_bars(frame, {"target": "Target weight", "end": "Weight at end of sample (after drift)"},
                        theme, "Asset allocation")


def risk_contribution_chart(contrib: pd.DataFrame, theme: ChartTheme) -> go.Figure:
    return _paired_bars(contrib, {"weight": "Capital weight", "pct_of_risk": "Share of portfolio volatility"},
                        theme, "Capital allocation vs risk allocation")
