# BUILD_MARKER: deploy-ready-v4 | gzip dataset support | layout and histogram fixes retained
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NIFTY 50 EDA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
ANALYSIS_START = pd.Timestamp("2011-08-22")
ANALYSIS_END = pd.Timestamp("2021-04-30")
WARMUP_START = pd.Timestamp("2010-11-04")
ANNOUNCEMENT_DATE = pd.Timestamp("2016-11-08")
EVENT_DAY_ZERO = pd.Timestamp("2016-11-09")
WIDE_EVENT_START = pd.Timestamp("2016-10-08")
WIDE_EVENT_END = pd.Timestamp("2017-01-08")
BASELINE_START = pd.Timestamp("2015-10-08")
BASELINE_END = pd.Timestamp("2016-10-07")
TRADING_DAYS_PER_YEAR = 252

BANK_STOCKS = [
    "AXISBANK",
    "HDFCBANK",
    "ICICIBANK",
    "INDUSINDBK",
    "KOTAKBANK",
    "SBIN",
]

AUTO_STOCKS = [
    "BAJAJ-AUTO",
    "EICHERMOT",
    "HEROMOTOCO",
    "M&M",
    "MARUTI",
    "TATAMOTORS",
]

FOCUS_STOCKS = BANK_STOCKS + AUTO_STOCKS

PAGES = [
    "Home / Overview",
    "Price Trends",
    "Risk & Return",
    "Comparative Analysis",
    "Demonetization Event",
]

FILE_CANDIDATES: dict[str, list[str]] = {
    "processed": [
        "NIFTY50_processed.csv",
        "NIFTY50_processed.csv.gz",
        "NIFTY50_processed(3).csv",
        "NIFTY50_processed(3).csv.gz",
        "processed_data.csv",
        "processed_data.csv.gz",
    ],
    "bull_bear": ["bull_bear_regime_summary.csv"],
    "abnormal_returns": ["demonetization_abnormal_returns.csv"],
    "demonetization_summary": ["demonetization_summary.csv"],
    "crossovers": ["golden_death_cross_events.csv"],
    "return_ranking": ["return_ranking_table.csv"],
    "risk_return": ["risk_return_summary.csv"],
    "sector_summary": ["sector_summary.csv"],
    "stock_index": ["stock_index_comparison.csv"],
}

PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "nifty50_eda_chart",
        "height": 900,
        "width": 1600,
        "scale": 2,
    },
}

PERCENT_COLUMNS = {
    "Annualized_Return",
    "Annualized_Volatility",
    "Maximum_Drawdown",
    "Historical_VaR_95_Loss",
    "Event_Cumulative_Return",
    "Event_Mean_Daily_Return",
    "Event_Daily_Volatility",
    "Event_Mean_Abnormal_Return",
    "Event_Cumulative_Abnormal_Return",
    "Baseline_Cumulative_Return",
    "Baseline_Mean_Daily_Return",
    "Baseline_Daily_Volatility",
    "Baseline_Mean_Abnormal_Return",
    "Baseline_Cumulative_Abnormal_Return",
    "Volatility_Change",
    "Abnormal_Return",
    "Cumulative_Abnormal_Return",
    "Daily_Return",
    "Index_Return",
}


# =============================================================================
# VISUAL STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            min-width: 290px;
            max-width: 340px;
        }
        .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(22, 120, 230, 0.12), rgba(0, 180, 160, 0.08));
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.15rem;
        }
        .hero p {
            margin: 0;
            font-size: 1.03rem;
            opacity: 0.9;
        }
        .section-note {
            border-left: 4px solid #1f77b4;
            padding: 0.55rem 0.8rem;
            background: rgba(31, 119, 180, 0.06);
            border-radius: 0 8px 8px 0;
            margin: 0.3rem 0 1rem 0;
        }
        .small-muted {
            opacity: 0.72;
            font-size: 0.88rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 12px;
            padding: 0.75rem;
        }
        .takeaway-card {
            border: 1px solid rgba(128, 128, 128, 0.24);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            min-height: 158px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0.7rem;
            overflow: hidden;
        }
        .takeaway-heading {
            font-size: 0.88rem;
            font-weight: 600;
            line-height: 1.25;
            opacity: 0.86;
            margin: 0;
        }
        .takeaway-value {
            font-size: clamp(1.72rem, 2.15vw, 2.20rem);
            font-weight: 500;
            line-height: 1.08;
            letter-spacing: -0.025em;
            margin: 0;
            white-space: normal;
            overflow-wrap: break-word;
        }
        .takeaway-delta {
            display: inline-block;
            align-self: flex-start;
            width: fit-content;
            max-width: 100%;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.18);
            color: #22a559;
            font-size: 0.84rem;
            font-weight: 600;
            line-height: 1.25;
            white-space: normal;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px 9px 0 0;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# FILE AND DATA HELPERS
# =============================================================================


def resolve_file(key: str) -> Path | None:
    """Return the first matching file in the app directory."""

    for filename in FILE_CANDIDATES[key]:
        candidate = APP_DIR / filename
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_csv(path_string: str, date_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load a CSV and parse selected date columns."""

    frame = pd.read_csv(path_string)
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_optional_table(key: str, date_columns: tuple[str, ...] = ()) -> pd.DataFrame | None:
    path = resolve_file(key)
    if path is None:
        return None
    return load_csv(str(path), date_columns)


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        st.error(f"{label} is missing required columns: {missing}")
        st.stop()


def safe_date_range(
    label: str,
    data: pd.DataFrame,
    key: str,
    default_start: pd.Timestamp | None = None,
    default_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Render a two-date selector and always return a valid ordered range."""

    minimum = pd.Timestamp(data["Date"].min()).normalize()
    maximum = pd.Timestamp(data["Date"].max()).normalize()
    start = max(minimum, (default_start or minimum).normalize())
    end = min(maximum, (default_end or maximum).normalize())

    selected = st.date_input(
        label,
        value=(start.date(), end.date()),
        min_value=minimum.date(),
        max_value=maximum.date(),
        key=key,
    )

    if isinstance(selected, Sequence) and len(selected) == 2:
        chosen_start = pd.Timestamp(selected[0])
        chosen_end = pd.Timestamp(selected[1])
    else:
        chosen_start = pd.Timestamp(selected)
        chosen_end = pd.Timestamp(selected)

    if chosen_start > chosen_end:
        chosen_start, chosen_end = chosen_end, chosen_start
    return chosen_start, chosen_end


def filter_dates(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return df.loc[df["Date"].between(start, end)].copy()


def format_table(
    df: pd.DataFrame,
    percent_columns: Iterable[str] = (),
    decimal_columns: Iterable[str] = (),
) -> pd.io.formats.style.Styler:
    formats: dict[str, str] = {}
    for column in percent_columns:
        if column in df.columns:
            formats[column] = "{:.2%}"
    for column in decimal_columns:
        if column in df.columns:
            formats[column] = "{:.3f}"
    return df.style.format(formats, na_rep="—")


def render_takeaway_card(
    container,
    heading: str,
    value: str,
    delta: str,
) -> None:
    """Render a compact responsive takeaway card without metric truncation."""

    with container:
        st.markdown(
            f"""
            <div class="takeaway-card">
                <p class="takeaway-heading">{escape(heading)}</p>
                <p class="takeaway-value">{escape(value)}</p>
                <span class="takeaway-delta">{escape(delta)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def place_legend_on_right(
    fig: go.Figure,
    *,
    right_margin: int,
    title: str,
    height: int,
    font_size: int = 11,
) -> go.Figure:
    """Move a dense categorical legend away from the chart title."""

    fig.update_layout(
        height=height,
        margin=dict(l=75, r=right_margin, t=100, b=70),
        title=dict(
            text=title,
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(t=8, b=16),
        ),
        legend=dict(
            title=dict(text="Sector"),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=font_size),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


# =============================================================================
# FINANCIAL CALCULATION HELPERS
# =============================================================================


def compound_return(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float((1.0 + values).prod() - 1.0)


def annualized_geometric_return(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    wealth = float((1.0 + values).prod())
    if wealth <= 0:
        return np.nan
    return float(wealth ** (TRADING_DAYS_PER_YEAR / len(values)) - 1.0)


def maximum_drawdown(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    wealth = (1.0 + values).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def calculate_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    pair = pd.concat([stock_returns, index_returns], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    variance = pair.iloc[:, 1].var(ddof=1)
    if pd.isna(variance) or variance == 0:
        return np.nan
    return float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / variance)


def normalized_price_return(
    df: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Create price-return indices rebased to 100 at the selected start."""

    work = df.loc[
        df["Symbol"].isin(symbols) & df["Date"].between(start, end),
        ["Date", "Symbol", "Daily_Return"],
    ].sort_values(["Symbol", "Date"]).copy()

    if work.empty:
        return work

    work["Adjusted_Return"] = work["Daily_Return"].fillna(0.0)
    first_observation = work.groupby("Symbol").cumcount().eq(0)
    work.loc[first_observation, "Adjusted_Return"] = 0.0
    work["Price_Return_Index"] = (1.0 + work["Adjusted_Return"]).groupby(
        work["Symbol"]
    ).cumprod() * 100.0
    return work.drop(columns="Adjusted_Return")


def relative_to_nifty(
    df: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Create stock price-return indices and relative-to-NIFTY ratios."""

    stock_indices = normalized_price_return(df, symbols, start, end)
    index = (
        df.loc[df["Date"].between(start, end), ["Date", "Index_Return"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .copy()
    )
    index_returns = index["Index_Return"].fillna(0.0).copy()
    if not index_returns.empty:
        index_returns.iloc[0] = 0.0
    index["NIFTY_Price_Return_Index"] = (1.0 + index_returns).cumprod() * 100.0

    merged = stock_indices.merge(index[["Date", "NIFTY_Price_Return_Index"]], on="Date", how="left")
    merged["Relative_to_NIFTY"] = (
        merged["Price_Return_Index"] / merged["NIFTY_Price_Return_Index"] * 100.0
    )
    return merged


# =============================================================================
# FALLBACK TABLE COMPUTATIONS
# =============================================================================


@st.cache_data(show_spinner=False)
def compute_risk_summary(processed_path: str) -> pd.DataFrame:
    df = load_csv(processed_path, ("Date",))
    df = df.loc[df["Date"].between(ANALYSIS_START, ANALYSIS_END)].copy()
    rows: list[dict[str, object]] = []

    for symbol, group in df.groupby("Symbol", sort=True):
        group = group.sort_values("Date")
        returns = group["Daily_Return"].dropna()
        annual_return = annualized_geometric_return(returns)
        annual_volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        sharpe_like = (
            annual_return / annual_volatility
            if pd.notna(annual_return) and pd.notna(annual_volatility) and annual_volatility > 0
            else np.nan
        )
        beta = calculate_beta(group["Daily_Return"], group["Index_Return"])
        rows.append(
            {
                "Symbol": symbol,
                "Sector": group["Sector"].iloc[0],
                "Start_Date": group["Date"].min(),
                "End_Date": group["Date"].max(),
                "Return_Observations": int(returns.size),
                "Annualized_Return": annual_return,
                "Annualized_Volatility": annual_volatility,
                "Sharpe_Ratio_RF0": sharpe_like,
                "Maximum_Drawdown": maximum_drawdown(returns),
                "Historical_VaR_95_Loss": -float(returns.quantile(0.05)),
                "Beta": beta,
            }
        )

    summary = pd.DataFrame(rows)
    summary["Annualized_Return_Rank"] = summary["Annualized_Return"].rank(
        ascending=False, method="min"
    )
    summary["Lowest_Volatility_Rank"] = summary["Annualized_Volatility"].rank(
        ascending=True, method="min"
    )
    return summary


@st.cache_data(show_spinner=False)
def compute_return_ranking(processed_path: str) -> pd.DataFrame:
    summary = compute_risk_summary(processed_path).copy()
    summary = summary.sort_values("Annualized_Return", ascending=False).reset_index(drop=True)
    summary["Annualized_Return_Rank"] = np.arange(1, len(summary) + 1)
    return summary[
        [
            "Annualized_Return_Rank",
            "Symbol",
            "Sector",
            "Annualized_Return",
            "Annualized_Volatility",
            "Sharpe_Ratio_RF0",
            "Maximum_Drawdown",
            "Historical_VaR_95_Loss",
            "Beta",
            "Return_Observations",
            "Start_Date",
            "End_Date",
        ]
    ]


@st.cache_data(show_spinner=False)
def compute_bull_bear(processed_path: str) -> pd.DataFrame:
    df = load_csv(processed_path, ("Date",))
    df = df.loc[df["Date"].between(ANALYSIS_START, ANALYSIS_END)].copy()
    valid = df.loc[df["Close"].notna() & df["MA_200"].notna()].copy()
    valid["Bull"] = valid["Close"] > valid["MA_200"]

    rows = []
    for symbol, group in valid.groupby("Symbol", sort=True):
        observations = len(group)
        bull_pct = float(group["Bull"].mean() * 100.0) if observations else np.nan
        rows.append(
            {
                "Symbol": symbol,
                "Bull_Pct": bull_pct,
                "Bear_Pct": 100.0 - bull_pct if pd.notna(bull_pct) else np.nan,
                "Observations": observations,
            }
        )
    return pd.DataFrame(rows).sort_values("Bull_Pct", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_crossovers(processed_path: str) -> pd.DataFrame:
    df = load_csv(processed_path, ("Date",))
    df = df.loc[df["Symbol"].isin(FOCUS_STOCKS)].sort_values(["Symbol", "Date"]).copy()
    output: list[pd.DataFrame] = []

    for symbol, group in df.groupby("Symbol", sort=False):
        group = group.copy()
        valid = group["MA_50"].notna() & group["MA_200"].notna()
        above = group["MA_50"] > group["MA_200"]
        previous_above = above.shift(1, fill_value=False).astype(bool)
        previous_valid = valid.shift(1, fill_value=False).astype(bool)
        golden = valid & previous_valid & above & ~previous_above
        death = valid & previous_valid & ~above & previous_above

        events = group.loc[golden | death, ["Symbol", "Date", "Close"]].copy()
        events["Event_Type"] = np.where(golden.loc[events.index], "Golden Cross", "Death Cross")
        events = events.loc[events["Date"].between(ANALYSIS_START, ANALYSIS_END)]
        output.append(events)

    if not output:
        return pd.DataFrame(columns=["Symbol", "Date", "Close", "Event_Type"])
    return pd.concat(output, ignore_index=True).sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_stock_index_comparison(processed_path: str) -> pd.DataFrame:
    df = load_csv(processed_path, ("Date",))
    df = df.loc[df["Date"].between(ANALYSIS_START, ANALYSIS_END)].copy()
    risk = compute_risk_summary(processed_path).set_index("Symbol")
    rows = []

    for symbol, group in df.groupby("Symbol", sort=True):
        pair = group[["Daily_Return", "Index_Return"]].dropna()
        rows.append(
            {
                "Symbol": symbol,
                "Sector": group["Sector"].iloc[0],
                "Stock_Index_Correlation": pair["Daily_Return"].corr(pair["Index_Return"]),
                "Beta": calculate_beta(group["Daily_Return"], group["Index_Return"]),
                "Paired_Observations": len(pair),
                "Annualized_Return": risk.loc[symbol, "Annualized_Return"],
                "Annualized_Volatility": risk.loc[symbol, "Annualized_Volatility"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_sector_summary(processed_path: str) -> pd.DataFrame:
    df = load_csv(processed_path, ("Date",))
    df = df.loc[df["Date"].between(ANALYSIS_START, ANALYSIS_END)].copy()

    company_counts = df.groupby("Sector")["Symbol"].nunique()
    sector_daily = (
        df.groupby(["Date", "Sector"], as_index=False)["Daily_Return"]
        .mean()
        .rename(columns={"Daily_Return": "Sector_Return"})
    )

    rows = []
    for sector, group in sector_daily.groupby("Sector", sort=True):
        returns = group["Sector_Return"].dropna()
        ann_return = annualized_geometric_return(returns)
        ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        rows.append(
            {
                "Sector": sector,
                "No_of_Companies": int(company_counts.loc[sector]),
                "Daily_Observations": int(len(returns)),
                "Annualized_Return": ann_return,
                "Annualized_Volatility": ann_vol,
                "Risk_Adjusted_Return_RF0": ann_return / ann_vol if ann_vol > 0 else np.nan,
                "Maximum_Drawdown": maximum_drawdown(returns),
                "Historical_VaR_95_Loss": -float(returns.quantile(0.05)),
                "Ranking_Status": (
                    "Eligible: multi-company sector"
                    if int(company_counts.loc[sector]) > 1
                    else "Excluded: single-stock sector"
                ),
            }
        )

    result = pd.DataFrame(rows)
    eligible = result["No_of_Companies"] > 1
    result["Sector_Return_Rank"] = np.nan
    result["Sector_Risk_Adjusted_Rank"] = np.nan
    result.loc[eligible, "Sector_Return_Rank"] = result.loc[eligible, "Annualized_Return"].rank(
        ascending=False, method="min"
    )
    result.loc[eligible, "Sector_Risk_Adjusted_Rank"] = result.loc[
        eligible, "Risk_Adjusted_Return_RF0"
    ].rank(ascending=False, method="min")
    return result.sort_values(
        ["No_of_Companies", "Annualized_Return"], ascending=[False, False]
    ).reset_index(drop=True)


def event_date_map(index_dates: pd.Series) -> dict[pd.Timestamp, int]:
    dates = pd.Series(pd.to_datetime(index_dates).dropna().unique()).sort_values().reset_index(drop=True)
    if EVENT_DAY_ZERO not in set(dates):
        raise ValueError("The 9 November 2016 market-response date is absent from the index calendar.")
    zero_position = int(dates[dates == EVENT_DAY_ZERO].index[0])
    return {pd.Timestamp(date): int(position - zero_position) for position, date in enumerate(dates)}


@st.cache_data(show_spinner=False)
def compute_demonetization_outputs(processed_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(processed_path, ("Date",))
    df = df.sort_values(["Symbol", "Date"]).copy()
    index_daily = (
        df[["Date", "Index_Return"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .rename(columns={"Index_Return": "Daily_Return"})
    )
    date_to_event_day = event_date_map(index_daily["Date"])
    index_daily["Event_Day"] = index_daily["Date"].map(date_to_event_day)

    windows: dict[str, pd.Series] = {
        "[-5,+5]": index_daily["Event_Day"].between(-5, 5),
        "[-20,+20]": index_daily["Event_Day"].between(-20, 20),
        "Wide_2016-10-08_to_2017-01-08": index_daily["Date"].between(
            WIDE_EVENT_START, WIDE_EVENT_END
        ),
    }

    baseline_stock = df.loc[df["Date"].between(BASELINE_START, BASELINE_END)].copy()
    baseline_sector = (
        baseline_stock.groupby(["Date", "Sector"], as_index=False)["Daily_Return"]
        .mean()
    )
    baseline_index = index_daily.loc[index_daily["Date"].between(BASELINE_START, BASELINE_END)]

    summary_rows: list[dict[str, object]] = []
    abnormal_rows: list[pd.DataFrame] = []

    for window_name, index_mask in windows.items():
        event_dates = index_daily.loc[index_mask, "Date"]
        if event_dates.empty:
            continue
        window_start = event_dates.min()
        window_end = event_dates.max()

        event_stock = df.loc[df["Date"].isin(event_dates)].copy()
        event_stock["Event_Day"] = event_stock["Date"].map(date_to_event_day)
        event_stock["Abnormal_Return"] = event_stock["Daily_Return"] - event_stock["Index_Return"]
        event_stock["Cumulative_Abnormal_Return"] = event_stock.groupby("Symbol")[
            "Abnormal_Return"
        ].cumsum()

        stock_abnormal = event_stock[
            [
                "Symbol",
                "Sector",
                "Date",
                "Daily_Return",
                "Index_Return",
                "Abnormal_Return",
                "Cumulative_Abnormal_Return",
                "Event_Day",
            ]
        ].copy()
        stock_abnormal.insert(0, "Analysis_Level", "Stock")
        stock_abnormal.insert(0, "Window", window_name)
        stock_abnormal = stock_abnormal.rename(columns={"Symbol": "Entity"})
        abnormal_rows.append(stock_abnormal)

        for symbol, group in event_stock.groupby("Symbol", sort=True):
            base = baseline_stock.loc[baseline_stock["Symbol"] == symbol]
            event_returns = group["Daily_Return"].dropna()
            base_returns = base["Daily_Return"].dropna()
            event_ar = group["Abnormal_Return"].dropna()
            base_ar = (base["Daily_Return"] - base["Index_Return"]).dropna()
            summary_rows.append(
                build_event_summary_row(
                    window_name,
                    window_start,
                    window_end,
                    "Stock",
                    symbol,
                    group["Sector"].iloc[0],
                    event_returns,
                    event_ar,
                    base_returns,
                    base_ar,
                )
            )

        event_sector = (
            event_stock.groupby(["Date", "Sector"], as_index=False)
            .agg(Daily_Return=("Daily_Return", "mean"), Index_Return=("Index_Return", "first"))
        )
        event_sector["Event_Day"] = event_sector["Date"].map(date_to_event_day)
        event_sector["Abnormal_Return"] = event_sector["Daily_Return"] - event_sector["Index_Return"]
        event_sector["Cumulative_Abnormal_Return"] = event_sector.groupby("Sector")[
            "Abnormal_Return"
        ].cumsum()

        sector_abnormal = event_sector.rename(columns={"Sector": "Entity"}).copy()
        sector_abnormal.insert(0, "Analysis_Level", "Sector")
        sector_abnormal.insert(0, "Window", window_name)
        sector_abnormal["Sector"] = sector_abnormal["Entity"]
        abnormal_rows.append(
            sector_abnormal[
                [
                    "Window",
                    "Analysis_Level",
                    "Entity",
                    "Sector",
                    "Date",
                    "Daily_Return",
                    "Index_Return",
                    "Abnormal_Return",
                    "Cumulative_Abnormal_Return",
                    "Event_Day",
                ]
            ]
        )

        for sector, group in event_sector.groupby("Sector", sort=True):
            base = baseline_sector.loc[baseline_sector["Sector"] == sector].copy()
            base = base.merge(
                index_daily[["Date", "Daily_Return"]].rename(
                    columns={"Daily_Return": "Index_Return"}
                ),
                on="Date",
                how="left",
            )
            summary_rows.append(
                build_event_summary_row(
                    window_name,
                    window_start,
                    window_end,
                    "Sector",
                    sector,
                    sector,
                    group["Daily_Return"].dropna(),
                    group["Abnormal_Return"].dropna(),
                    base["Daily_Return"].dropna(),
                    (base["Daily_Return"] - base["Index_Return"]).dropna(),
                )
            )

        event_index = index_daily.loc[index_mask].copy()
        summary_rows.append(
            build_event_summary_row(
                window_name,
                window_start,
                window_end,
                "Index",
                "NIFTY 50",
                "Market Index",
                event_index["Daily_Return"].dropna(),
                pd.Series(np.zeros(len(event_index)), dtype=float),
                baseline_index["Daily_Return"].dropna(),
                pd.Series(np.zeros(len(baseline_index)), dtype=float),
            )
        )

    summary = pd.DataFrame(summary_rows)
    abnormal = pd.concat(abnormal_rows, ignore_index=True) if abnormal_rows else pd.DataFrame()
    return summary, abnormal


def build_event_summary_row(
    window_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    analysis_level: str,
    entity: str,
    sector: str,
    event_returns: pd.Series,
    event_abnormal: pd.Series,
    baseline_returns: pd.Series,
    baseline_abnormal: pd.Series,
) -> dict[str, object]:
    event_volatility = event_returns.std(ddof=1)
    baseline_volatility = baseline_returns.std(ddof=1)
    return {
        "Window": window_name,
        "Window_Start": window_start,
        "Window_End": window_end,
        "Policy_Announcement_Date": ANNOUNCEMENT_DATE,
        "Market_Response_Day_Zero": EVENT_DAY_ZERO,
        "Analysis_Level": analysis_level,
        "Entity": entity,
        "Sector": sector,
        "Event_Observations": int(event_returns.size),
        "Event_Cumulative_Return": compound_return(event_returns),
        "Event_Mean_Daily_Return": event_returns.mean(),
        "Event_Daily_Volatility": event_volatility,
        "Event_Mean_Abnormal_Return": event_abnormal.mean(),
        "Event_Cumulative_Abnormal_Return": event_abnormal.sum(),
        "Baseline_Observations": int(baseline_returns.size),
        "Baseline_Cumulative_Return": compound_return(baseline_returns),
        "Baseline_Mean_Daily_Return": baseline_returns.mean(),
        "Baseline_Daily_Volatility": baseline_volatility,
        "Baseline_Mean_Abnormal_Return": baseline_abnormal.mean(),
        "Baseline_Cumulative_Abnormal_Return": baseline_abnormal.sum(),
        "Volatility_Change": event_volatility - baseline_volatility,
    }


# =============================================================================
# PLOT HELPERS
# =============================================================================


def finish_figure(
    fig: go.Figure,
    title: str,
    height: int = 520,
    hovermode: str | bool = "x unified",
    showlegend: bool = True,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        hovermode=hovermode,
        showlegend=showlegend,
        margin=dict(l=35, r=30, t=75, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def show_plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=PLOT_CONFIG,
        key=key,
    )


def add_event_lines(fig: go.Figure, annotation: bool = True) -> None:
    fig.add_vline(
        x=ANNOUNCEMENT_DATE,
        line_dash="dash",
        line_width=2,
        line_color="#ff8c00",
    )
    fig.add_vline(
        x=EVENT_DAY_ZERO,
        line_dash="dot",
        line_width=2,
        line_color="#d62728",
    )
    if annotation:
        fig.add_annotation(
            x=ANNOUNCEMENT_DATE,
            y=1.08,
            yref="paper",
            text="8 Nov: announcement",
            showarrow=False,
            font=dict(color="#ff8c00", size=11),
        )
        fig.add_annotation(
            x=EVENT_DAY_ZERO,
            y=1.02,
            yref="paper",
            text="9 Nov: trading day 0",
            showarrow=False,
            font=dict(color="#d62728", size=11),
        )


def focus_trend_figure(
    stock_data: pd.DataFrame,
    crossover_data: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    fig = go.Figure()
    line_specs = [
        ("Close", "Close", 2.0),
        ("MA_50", "MA 50", 1.6),
        ("MA_100", "MA 100", 1.6),
        ("MA_200", "MA 200", 1.8),
    ]
    for column, label, width in line_specs:
        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=stock_data[column],
                mode="lines",
                name=label,
                line=dict(width=width),
                hovertemplate=f"%{{x|%d %b %Y}}<br>{label}: ₹%{{y:,.2f}}<extra></extra>",
            )
        )

    events = crossover_data.loc[crossover_data["Symbol"] == symbol]
    for event_type, marker_symbol, marker_color in [
        ("Golden Cross", "triangle-up", "#14883b"),
        ("Death Cross", "triangle-down", "#d62728"),
    ]:
        subset = events.loc[events["Event_Type"] == event_type]
        fig.add_trace(
            go.Scatter(
                x=subset["Date"],
                y=subset["Close"],
                mode="markers",
                name=event_type,
                marker=dict(symbol=marker_symbol, size=11, color=marker_color),
                hovertemplate=(
                    "%{x|%d %b %Y}<br>Close: ₹%{y:,.2f}<br>" + event_type + "<extra></extra>"
                ),
            )
        )

    fig.update_yaxes(type="log", title="Price (₹, log scale)")
    fig.update_xaxes(title="Date", rangeslider_visible=False)
    return finish_figure(fig, f"{symbol}: Price Trend and Moving-Average Crossovers", 610)


def crossover_timeline_figure(events: pd.DataFrame) -> go.Figure:
    order = [symbol for symbol in FOCUS_STOCKS if symbol in set(events["Symbol"])]
    fig = px.scatter(
        events,
        x="Date",
        y="Symbol",
        color="Event_Type",
        symbol="Event_Type",
        category_orders={"Symbol": order[::-1]},
        color_discrete_map={"Golden Cross": "#14883b", "Death Cross": "#d62728"},
        symbol_map={"Golden Cross": "triangle-up", "Death Cross": "triangle-down"},
        hover_data={"Close": ":,.2f", "Date": "|%d %b %Y"},
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Focus stock")
    return finish_figure(fig, "Golden-Cross / Death-Cross Timeline", 600, hovermode="closest")


def bull_bear_figure(summary: pd.DataFrame, symbols: list[str]) -> go.Figure:
    work = summary.loc[summary["Symbol"].isin(symbols)].copy()
    work = work.sort_values("Bull_Pct", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=work["Symbol"],
            x=work["Bull_Pct"],
            orientation="h",
            name="Bull regime",
            marker_color="#2ca02c",
            customdata=work[["Observations"]],
            hovertemplate="%{y}<br>Bull: %{x:.2f}%<br>Observations: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=work["Symbol"],
            x=work["Bear_Pct"],
            orientation="h",
            name="Bear regime",
            marker_color="#d62728",
            customdata=work[["Observations"]],
            hovertemplate="%{y}<br>Bear: %{x:.2f}%<br>Observations: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Share of valid MA-200 observations (%)", range=[0, 100])
    fig.update_yaxes(title="")
    return finish_figure(fig, "Bull / Bear Regime Summary", max(500, 25 * len(work) + 180), hovermode="closest")


def risk_return_scatter(summary: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        summary,
        x="Annualized_Volatility",
        y="Annualized_Return",
        color="Sector",
        size="Historical_VaR_95_Loss",
        hover_name="Symbol",
        hover_data={
            "Annualized_Volatility": ":.2%",
            "Annualized_Return": ":.2%",
            "Maximum_Drawdown": ":.2%",
            "Historical_VaR_95_Loss": ":.2%",
            "Beta": ":.3f",
            "Sector": True,
        },
        labels={
            "Annualized_Volatility": "Annualized volatility",
            "Annualized_Return": "Annualized return",
        },
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")

    fig = finish_figure(
        fig,
        "Stock Risk–Return Map",
        720,
        hovermode="closest",
    )

    return place_legend_on_right(
        fig,
        right_margin=285,
        title="Stock Risk–Return Map",
        height=720,
        font_size=11,
    )


def ranking_bar(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ascending: bool,
    percent: bool = True,
) -> go.Figure:
    work = summary.sort_values(metric, ascending=ascending).head(10).sort_values(metric)
    fig = px.bar(
        work,
        x=metric,
        y="Symbol",
        orientation="h",
        color=metric,
        color_continuous_scale="Blues",
        hover_data={metric: ":.2%" if percent else ":.3f", "Sector": True},
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title=metric.replace("_", " "), tickformat=".0%" if percent else None)
    fig.update_yaxes(title="")
    return finish_figure(fig, title, 520, hovermode="closest", showlegend=False)


def stock_return_distribution(
    stock_data: pd.DataFrame,
    symbol: str,
    bins: int,
    view: str,
    limit_to_15: bool,
) -> go.Figure:
    """
    Create an interactive return-distribution chart for one stock.

    Histogram behavior:
    - Uses NumPy histogram bin edges, matching Matplotlib-style binning.
    - Keeps all valid observations in the histogram calculation.
    - Restricts only the visible x-axis to +/-15% when selected.
    - Does not add a marginal box plot.

    Violin behavior remains unchanged.
    """

    work = stock_data.loc[
        stock_data["Daily_Return"].notna()
    ].copy()

    if view == "Histogram":
        values = (
            work["Daily_Return"]
            .dropna()
            .to_numpy(dtype=float)
        )

        if values.size == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid daily-return observations are available.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_xaxes(
                title=dict(
                    text="Daily return",
                    standoff=14,
                ),
                tickformat=".1%",
                range=[-0.15, 0.15] if limit_to_15 else None,
                automargin=True,
            )
            fig.update_yaxes(
                title=dict(
                    text="Number of observations",
                    standoff=18,
                ),
                automargin=True,
                rangemode="tozero",
            )
        else:
            # NumPy calculates the exact histogram edges used by
            # Matplotlib when the same data and bin count are supplied.
            counts, edges = np.histogram(
                values,
                bins=bins,
            )

            bin_centres = (
                edges[:-1] + edges[1:]
            ) / 2
            bin_widths = np.diff(edges)

            hover_data = np.column_stack(
                [
                    edges[:-1],
                    edges[1:],
                ]
            )

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=bin_centres,
                    y=counts,
                    width=bin_widths,
                    customdata=hover_data,
                    name="Daily returns",
                    hovertemplate=(
                        "Return interval: "
                        "%{customdata[0]:.2%} to "
                        "%{customdata[1]:.2%}"
                        "<br>Number of observations: %{y:,}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_xaxes(
                title=dict(
                    text="Daily return",
                    standoff=14,
                ),
                tickformat=".1%",
                range=(
                    [-0.15, 0.15]
                    if limit_to_15
                    else None
                ),
                automargin=True,
            )
            fig.update_yaxes(
                title=dict(
                    text="Number of observations",
                    standoff=18,
                ),
                automargin=True,
                rangemode="tozero",
            )
            fig.update_layout(
                bargap=0.03,
            )
    else:
        # The violin-chart option is retained without changing its
        # position or behavior in the dashboard.
        fig = px.violin(
            work,
            y="Daily_Return",
            box=True,
            points="outliers",
            hover_data={
                "Date": "|%d %b %Y",
            },
        )
        fig.update_yaxes(
            title=dict(
                text="Daily return",
                standoff=18,
            ),
            tickformat=".1%",
            automargin=True,
        )
        fig.update_xaxes(
            title="",
            automargin=True,
        )

    fig = finish_figure(
        fig,
        f"{symbol}: Cleaned Daily-Return Distribution",
        600,
        hovermode="closest",
        showlegend=False,
    )

    # Extra spacing prevents the chart title and y-axis title from
    # overlapping with the plotting area.
    fig.update_layout(
        margin=dict(
            l=95,
            r=35,
            t=100,
            b=85,
        ),
        title=dict(
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(
                t=10,
                b=20,
            ),
        ),
    )

    return fig

def beta_correlation_figure(comparison: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        comparison,
        x="Stock_Index_Correlation",
        y="Beta",
        color="Sector",
        hover_name="Symbol",
        hover_data={
            "Stock_Index_Correlation": ":.3f",
            "Beta": ":.3f",
            "Annualized_Return": ":.2%",
            "Annualized_Volatility": ":.2%",
            "Paired_Observations": ":,",
        },
    )
    fig.update_xaxes(
        title="Correlation with NIFTY 50",
        range=[-0.05, 1.0],
    )
    fig.update_yaxes(title="Beta")

    fig = finish_figure(
        fig,
        "Beta versus NIFTY 50 Correlation",
        700,
        hovermode="closest",
    )

    return place_legend_on_right(
        fig,
        right_margin=285,
        title="Beta versus NIFTY 50 Correlation",
        height=700,
        font_size=11,
    )


def sector_risk_return_figure(sector_summary: pd.DataFrame, include_single: bool) -> go.Figure:
    work = sector_summary.copy()
    if not include_single:
        work = work.loc[work["No_of_Companies"] > 1]

    fig = px.scatter(
        work,
        x="Annualized_Volatility",
        y="Annualized_Return",
        size="No_of_Companies",
        color="No_of_Companies",
        text="No_of_Companies",
        hover_name="Sector",
        hover_data={
            "No_of_Companies": True,
            "Annualized_Return": ":.2%",
            "Annualized_Volatility": ":.2%",
            "Maximum_Drawdown": ":.2%",
            "Historical_VaR_95_Loss": ":.2%",
            "Ranking_Status": True,
        },
        color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="middle center", textfont=dict(color="white", size=11))
    fig.update_layout(coloraxis_colorbar_title="Companies")
    fig.update_xaxes(title="Annualized volatility", tickformat=".0%")
    fig.update_yaxes(title="Annualized return", tickformat=".0%")
    return finish_figure(fig, "Sample-Sector Risk–Return Map (bubble label = n)", 650, hovermode="closest")


def correlation_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(
        matrix,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        labels=dict(color="Correlation"),
    )
    fig.update_traces(
        hovertemplate="Row: %{y}<br>Column: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
    )
    fig.update_xaxes(side="bottom", tickangle=45)
    return finish_figure(
        fig,
        title,
        max(620, 28 * len(matrix) + 200),
        hovermode=False,
        showlegend=False,
    )


def nifty_event_figure(processed: pd.DataFrame) -> go.Figure:
    index = (
        processed.loc[processed["Date"].between(WIDE_EVENT_START, WIDE_EVENT_END),
                      ["Date", "Index_Return", "Index_Rolling_Vol"]]
        .drop_duplicates("Date")
        .sort_values("Date")
        .copy()
    )
    returns = index["Index_Return"].fillna(0.0)
    index["Cumulative_Return"] = (1.0 + returns).cumprod() - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Cumulative price return", "21-day rolling volatility"),
    )
    fig.add_trace(
        go.Scatter(
            x=index["Date"],
            y=index["Cumulative_Return"],
            mode="lines",
            name="NIFTY cumulative return",
            hovertemplate="%{x|%d %b %Y}<br>Cumulative return: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=index["Date"],
            y=index["Index_Rolling_Vol"],
            mode="lines",
            name="NIFTY rolling volatility",
            hovertemplate="%{x|%d %b %Y}<br>Rolling volatility: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for row in (1, 2):
        fig.add_vline(x=ANNOUNCEMENT_DATE, line_dash="dash", line_color="#ff8c00", row=row, col=1)
        fig.add_vline(x=EVENT_DAY_ZERO, line_dash="dot", line_color="#d62728", row=row, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    fig.update_xaxes(title="Date", row=2, col=1)
    return finish_figure(fig, "NIFTY 50 Around Demonetization", 720)


def candlestick_figure(stock: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock["Date"],
                open=stock["Open"],
                high=stock["High"],
                low=stock["Low"],
                close=stock["Close"],
                name=symbol,
                increasing_line_color="#14883b",
                decreasing_line_color="#d62728",
                hovertext=stock["Date"].dt.strftime("%d %b %Y"),
            )
        ]
    )
    add_event_lines(fig)
    fig.update_xaxes(title="Date", rangeslider_visible=False)
    fig.update_yaxes(title="Price (₹)")
    return finish_figure(fig, f"{symbol}: Demonetization-Window Candlestick", 570, hovermode="x")


# =============================================================================
# LOAD DATA AND FALL BACK TO ON-THE-FLY CALCULATION
# =============================================================================

processed_path = resolve_file("processed")
if processed_path is None:
    st.error(
        "Main processed dataset not found. Place `NIFTY50_processed.csv`, "
        "`NIFTY50_processed.csv.gz`, `NIFTY50_processed(3).csv`, "
        "or `processed_data.csv` beside `app.py`."
    )
    st.stop()

processed = load_csv(str(processed_path), ("Date",))
require_columns(
    processed,
    {
        "Date",
        "Symbol",
        "Company_Name",
        "Sector",
        "Open",
        "High",
        "Low",
        "Close",
        "Daily_Return",
        "MA_50",
        "MA_100",
        "MA_200",
        "Rolling_Vol_21d",
        "Index_Close",
        "Index_Return",
        "Index_Rolling_Vol",
        "Beta",
    },
    "Processed dataset",
)

processed = processed.sort_values(["Symbol", "Date"]).reset_index(drop=True)
analysis_data = processed.loc[
    processed["Date"].between(ANALYSIS_START, ANALYSIS_END)
].copy()

risk_summary = load_optional_table("risk_return", ("Start_Date", "End_Date"))
if risk_summary is None:
    risk_summary = compute_risk_summary(str(processed_path))

return_ranking = load_optional_table("return_ranking", ("Start_Date", "End_Date"))
if return_ranking is None:
    return_ranking = compute_return_ranking(str(processed_path))

bull_bear = load_optional_table("bull_bear")
if bull_bear is None:
    bull_bear = compute_bull_bear(str(processed_path))

crossovers = load_optional_table("crossovers", ("Date",))
if crossovers is None:
    crossovers = compute_crossovers(str(processed_path))

stock_index_comparison = load_optional_table("stock_index")
if stock_index_comparison is None:
    stock_index_comparison = compute_stock_index_comparison(str(processed_path))

sector_summary = load_optional_table("sector_summary")
if sector_summary is None:
    sector_summary = compute_sector_summary(str(processed_path))

demonetization_summary = load_optional_table(
    "demonetization_summary",
    ("Window_Start", "Window_End", "Policy_Announcement_Date", "Market_Response_Day_Zero"),
)
abnormal_returns = load_optional_table("abnormal_returns", ("Date",))
if demonetization_summary is None or abnormal_returns is None:
    computed_summary, computed_abnormal = compute_demonetization_outputs(str(processed_path))
    if demonetization_summary is None:
        demonetization_summary = computed_summary
    if abnormal_returns is None:
        abnormal_returns = computed_abnormal


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

if "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]


def set_page(page: str) -> None:
    st.session_state.nav_page = page


with st.sidebar:
    st.title("📈 NIFTY 50 EDA")
    st.caption("Interactive project dashboard")
    st.caption("Build: deploy-ready-v4")
    selected_page = st.radio(
        "Navigation",
        PAGES,
        key="nav_page",
    )

    st.divider()
    st.subheader("Dataset status")
    st.success(f"Main panel: {processed_path.name}")
    st.caption(
        f"{processed['Symbol'].nunique()} stocks · "
        f"{processed['Date'].nunique():,} warm-up dates · "
        f"{analysis_data['Date'].nunique():,} reported dates"
    )

    optional_status = {
        "Risk-return": resolve_file("risk_return"),
        "Sector summary": resolve_file("sector_summary"),
        "Crossovers": resolve_file("crossovers"),
        "Event summary": resolve_file("demonetization_summary"),
        "Abnormal returns": resolve_file("abnormal_returns"),
    }
    with st.expander("Input table sources", expanded=False):
        for label, path in optional_status.items():
            if path:
                st.write(f"✅ {label}: `{path.name}`")
            else:
                st.write(f"🧮 {label}: calculated from main panel")

    st.divider()
    st.caption("Analysis period")
    st.write("**22 Aug 2011 – 30 Apr 2021**")
    st.caption("Rolling-indicator warm-up begins 4 Nov 2010.")


# =============================================================================
# HOME / OVERVIEW PAGE
# =============================================================================

if selected_page == "Home / Overview":
    st.markdown(
        """
        <div class="hero">
            <h1>Historical NIFTY-50 Stocks EDA Dashboard</h1>
            <p>Interactive exploration of price trends, risk and return, comparative market behaviour, and the 2016 demonetization event.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Canonical stocks", f"{analysis_data['Symbol'].nunique()}")
    metric_columns[1].metric("Reported trading dates", f"{analysis_data['Date'].nunique():,}")
    metric_columns[2].metric("Static sectors", f"{analysis_data['Sector'].nunique()}")
    metric_columns[3].metric("Focus stocks", "12")
    metric_columns[4].metric("Demonetization windows", "3")

    st.subheader("Project overview")
    st.write(
        "The dashboard mirrors the four retained analytical modules in the final report: "
        "price trends, risk-return, comparative analysis, and demonetization-event analysis. "
        "The original project has two phases: Phase 1 builds the aligned and cleaned stock-index panel; "
        "Phase 2 reports all results over the fully MA-ready common period."
    )

    st.subheader("Key takeaways")
    top_return = return_ranking.sort_values("Annualized_Return_Rank").iloc[0]
    top_vol = risk_summary.sort_values("Annualized_Volatility", ascending=False).iloc[0]
    top_sector = sector_summary.loc[sector_summary["No_of_Companies"] > 1].sort_values(
        "Annualized_Return", ascending=False
    ).iloc[0]
    nifty_narrow = demonetization_summary.loc[
        (demonetization_summary["Analysis_Level"] == "Index")
        & (demonetization_summary["Window"] == "[-5,+5]")
    ]

    takeaway_cols = st.columns(4)
    render_takeaway_card(
        takeaway_cols[0],
        "Highest annualized return",
        str(top_return["Symbol"]),
        f"↑ {top_return['Annualized_Return']:.2%}",
    )
    render_takeaway_card(
        takeaway_cols[1],
        "Highest volatility",
        str(top_vol["Symbol"]),
        f"↑ {top_vol['Annualized_Volatility']:.2%}",
    )
    render_takeaway_card(
        takeaway_cols[2],
        "Leading multi-stock sector",
        str(top_sector["Sector"]),
        f"↑ {top_sector['Annualized_Return']:.2%}",
    )
    if not nifty_narrow.empty:
        row = nifty_narrow.iloc[0]
        render_takeaway_card(
            takeaway_cols[3],
            "NIFTY return [-5,+5]",
            f"{row['Event_Cumulative_Return']:.2%}",
            f"↑ Vol change {row['Volatility_Change']:.2%}",
        )

    st.subheader("Explore the analyses")
    nav_cols = st.columns(4)
    nav_cols[0].button(
        "📉 Price Trends",
        width="stretch",
        on_click=set_page,
        args=("Price Trends",),
    )
    nav_cols[1].button(
        "⚖️ Risk & Return",
        width="stretch",
        on_click=set_page,
        args=("Risk & Return",),
    )
    nav_cols[2].button(
        "🔗 Comparative Analysis",
        width="stretch",
        on_click=set_page,
        args=("Comparative Analysis",),
    )
    nav_cols[3].button(
        "🏦 Demonetization Event",
        width="stretch",
        on_click=set_page,
        args=("Demonetization Event",),
    )

    st.subheader("Important interpretation limits")
    col_a, col_b = st.columns(2)
    with col_a:
        st.warning(
            "**Raw-price limitation:** Close, moving averages, crossover dates and bull/bear regimes "
            "are based on raw prices that are not comprehensively adjusted for stock splits, bonus issues, "
            "rights issues, demergers or dividends. These outputs are descriptive, not trading signals."
        )
    with col_b:
        st.warning(
            "**Survivorship and constituent-selection limitation:** The 49-stock panel is not a reconstructed "
            "point-in-time NIFTY 50 membership history. Static constituent and sector labels are applied "
            "retrospectively, so results describe the supplied panel rather than an investable historical index."
        )

    st.info(
        "Normalized paths and relative-performance charts accumulate **price returns only**. "
        "They are not dividend-adjusted total-return or shareholder-wealth series."
    )


# =============================================================================
# PRICE TRENDS PAGE
# =============================================================================

elif selected_page == "Price Trends":
    st.title("Price Trends")
    st.markdown(
        '<div class="section-note">Choose a focus stock and date range. Every chart supports hover, zoom, pan and image export.</div>',
        unsafe_allow_html=True,
    )

    stock_col, range_col = st.columns([1, 2])
    with stock_col:
        focus_symbol = st.selectbox("Focus stock", FOCUS_STOCKS, index=0)
    with range_col:
        trend_start, trend_end = safe_date_range(
            "Trend date range",
            analysis_data,
            "price_trend_range",
            ANALYSIS_START,
            ANALYSIS_END,
        )

    tabs = st.tabs(
        [
            "Selected stock trend",
            "Normalized performance",
            "Crossover timeline",
            "Bull / bear regimes",
            "Exact tables",
        ]
    )

    with tabs[0]:
        selected_stock = filter_dates(
            analysis_data.loc[analysis_data["Symbol"] == focus_symbol],
            trend_start,
            trend_end,
        )
        selected_events = crossovers.loc[
            (crossovers["Symbol"] == focus_symbol)
            & crossovers["Date"].between(trend_start, trend_end)
        ]
        show_plot(
            focus_trend_figure(selected_stock, selected_events, focus_symbol),
            f"trend_{focus_symbol}_{trend_start.date()}_{trend_end.date()}",
        )
        st.caption(
            "The price axis is logarithmic. Moving averages and crossover markers use raw, unadjusted Close values."
        )

    with tabs[1]:
        default_symbols = ["BAJFINANCE", "BAJAJFINSV", "EICHERMOT", "SHREECEM", "KOTAKBANK"]
        available_symbols = sorted(analysis_data["Symbol"].unique())
        selected_symbols = st.multiselect(
            "Stocks to compare",
            available_symbols,
            default=[symbol for symbol in default_symbols if symbol in available_symbols],
            key="normalized_symbols",
        )
        if selected_symbols:
            normalized = normalized_price_return(
                analysis_data, selected_symbols, trend_start, trend_end
            )
            fig = px.line(
                normalized,
                x="Date",
                y="Price_Return_Index",
                color="Symbol",
                hover_data={"Price_Return_Index": ":.2f", "Date": "|%d %b %Y"},
            )
            fig.update_xaxes(title="Date")
            fig.update_yaxes(title="Normalized price-return index (start = 100)")
            show_plot(
                finish_figure(fig, "Normalized Cumulative Price Performance", 620),
                "normalized_price_performance",
            )
            st.caption("Dividends are excluded; this is cumulative price performance, not total return.")
        else:
            st.info("Select at least one stock.")

    with tabs[2]:
        timeline_symbols = st.multiselect(
            "Focus stocks shown in timeline",
            FOCUS_STOCKS,
            default=FOCUS_STOCKS,
            key="timeline_symbols",
        )
        timeline = crossovers.loc[
            crossovers["Symbol"].isin(timeline_symbols)
            & crossovers["Date"].between(trend_start, trend_end)
        ]
        if timeline.empty:
            st.info("No crossover events are available for the selected filters.")
        else:
            show_plot(crossover_timeline_figure(timeline), "crossover_timeline")
            counts = (
                timeline.groupby(["Symbol", "Event_Type"]).size().unstack(fill_value=0)
            )
            st.caption(
                f"Displayed events: {len(timeline)}. Counts are shown for transparency and are not used to rank stocks."
            )
            st.dataframe(counts, width="stretch")

    with tabs[3]:
        regime_scope = st.radio(
            "Regime view",
            ["12 focus stocks", "All 49 stocks"],
            horizontal=True,
            key="regime_scope",
        )
        regime_symbols = FOCUS_STOCKS if regime_scope == "12 focus stocks" else sorted(
            bull_bear["Symbol"].unique()
        )
        show_plot(bull_bear_figure(bull_bear, regime_symbols), "bull_bear_plot")

    with tabs[4]:
        st.markdown("#### Crossover event audit")
        table = crossovers.loc[crossovers["Symbol"].isin(FOCUS_STOCKS)].copy()
        table["Date"] = pd.to_datetime(table["Date"]).dt.date
        st.dataframe(table, width="stretch", hide_index=True)
        st.markdown("#### Bull / bear summary")
        st.dataframe(
            bull_bear.sort_values("Bull_Pct", ascending=False).style.format(
                {"Bull_Pct": "{:.2f}%", "Bear_Pct": "{:.2f}%"}
            ),
            width="stretch",
            hide_index=True,
        )


# =============================================================================
# RISK & RETURN PAGE
# =============================================================================

elif selected_page == "Risk & Return":
    st.title("Risk & Return")
    st.markdown(
        '<div class="section-note">All summary metrics use the common reported period, 22 Aug 2011 to 30 Apr 2021.</div>',
        unsafe_allow_html=True,
    )

    sectors = sorted(risk_summary["Sector"].dropna().unique())
    chosen_sectors = st.multiselect(
        "Sectors included in the risk-return map",
        sectors,
        default=sectors,
        key="risk_sectors",
    )
    risk_filtered = risk_summary.loc[risk_summary["Sector"].isin(chosen_sectors)]

    tabs = st.tabs(["Risk-return map", "Top 10 rankings", "Stock explorer", "Exact report tables"])

    with tabs[0]:
        show_plot(risk_return_scatter(risk_filtered), "risk_return_scatter")

    with tabs[1]:
        ranking_choice = st.radio(
            "Ranking metric",
            ["Annualized return", "Annualized volatility", "95% VaR loss"],
            horizontal=True,
            key="ranking_metric",
        )
        if ranking_choice == "Annualized return":
            figure = ranking_bar(
                risk_summary,
                "Annualized_Return",
                "Top 10 Annualized Returns",
                ascending=False,
            )
        elif ranking_choice == "Annualized volatility":
            figure = ranking_bar(
                risk_summary,
                "Annualized_Volatility",
                "Top 10 Annualized Volatilities",
                ascending=False,
            )
        else:
            figure = ranking_bar(
                risk_summary,
                "Historical_VaR_95_Loss",
                "Top 10 Historical 95% Daily VaR Losses",
                ascending=False,
            )
        show_plot(figure, f"ranking_{ranking_choice}")

    with tabs[2]:
        explorer_col, range_col = st.columns([1, 2])
        with explorer_col:
            selected_stock = st.selectbox(
                "Stock",
                sorted(analysis_data["Symbol"].unique()),
                key="risk_stock",
            )
        with range_col:
            distribution_start, distribution_end = safe_date_range(
                "Distribution date range",
                analysis_data,
                "risk_distribution_range",
                ANALYSIS_START,
                ANALYSIS_END,
            )

        row = risk_summary.loc[risk_summary["Symbol"] == selected_stock].iloc[0]
        metric_cols = st.columns(6)
        metric_cols[0].metric("Annualized return", f"{row['Annualized_Return']:.2%}")
        metric_cols[1].metric("Annualized volatility", f"{row['Annualized_Volatility']:.2%}")
        metric_cols[2].metric("Sharpe-like (RF=0)", f"{row['Sharpe_Ratio_RF0']:.3f}")
        metric_cols[3].metric("Maximum drawdown", f"{row['Maximum_Drawdown']:.2%}")
        metric_cols[4].metric("95% VaR loss", f"{row['Historical_VaR_95_Loss']:.2%}")
        metric_cols[5].metric("Beta", f"{row['Beta']:.3f}")

        control_a, control_b = st.columns(2)
        with control_a:
            distribution_view = st.radio(
                "Distribution view",
                ["Histogram", "Violin"],
                horizontal=True,
                key="distribution_view",
            )
        with control_b:
            bins = st.slider(
                "Histogram bins",
                min_value=20,
                max_value=120,
                value=45,
                step=5,
                key="histogram_bins_v2",
            )

        limit_to_15 = st.toggle(
            "Restrict histogram display to ±15% (display only)",
            value=True,
            disabled=distribution_view != "Histogram",
        )
        stock_returns = analysis_data.loc[
            (analysis_data["Symbol"] == selected_stock)
            & analysis_data["Date"].between(distribution_start, distribution_end)
        ]
        show_plot(
            stock_return_distribution(
                stock_returns, selected_stock, bins, distribution_view, limit_to_15
            ),
            f"return_distribution_{selected_stock}_{distribution_view}_{limit_to_15}",
        )

        if limit_to_15 and distribution_view == "Histogram":
            outside = stock_returns["Daily_Return"].abs().gt(0.15).sum()
            st.info(
                f"{outside:,} cleaned observations for {selected_stock} lie outside ±15% in the selected period. "
                "They remain included in every analytical calculation."
            )

    with tabs[3]:
        top_return_table = return_ranking.sort_values("Annualized_Return_Rank").head(10).copy()
        top_vol_table = risk_summary.sort_values("Annualized_Volatility", ascending=False).head(10).copy()
        top_var_table = risk_summary.sort_values("Historical_VaR_95_Loss", ascending=False).head(10).copy()

        st.markdown("#### Exact top 10 annualized returns")
        st.dataframe(
            format_table(
                top_return_table,
                percent_columns=[
                    "Annualized_Return",
                    "Annualized_Volatility",
                    "Maximum_Drawdown",
                    "Historical_VaR_95_Loss",
                ],
                decimal_columns=["Sharpe_Ratio_RF0", "Beta"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Exact top 10 annualized volatilities")
        st.dataframe(
            format_table(
                top_vol_table,
                percent_columns=[
                    "Annualized_Return",
                    "Annualized_Volatility",
                    "Maximum_Drawdown",
                    "Historical_VaR_95_Loss",
                ],
                decimal_columns=["Sharpe_Ratio_RF0", "Beta"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.markdown("#### Exact top 10 historical 95% VaR losses")
        st.dataframe(
            format_table(
                top_var_table,
                percent_columns=[
                    "Annualized_Return",
                    "Annualized_Volatility",
                    "Maximum_Drawdown",
                    "Historical_VaR_95_Loss",
                ],
                decimal_columns=["Sharpe_Ratio_RF0", "Beta"],
            ),
            width="stretch",
            hide_index=True,
        )


# =============================================================================
# COMPARATIVE ANALYSIS PAGE
# =============================================================================

elif selected_page == "Comparative Analysis":
    st.title("Comparative Analysis")
    st.markdown(
        '<div class="section-note">Compare market sensitivity, relative cumulative price performance, sample-sector risk, and return correlations.</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Beta vs correlation",
            "Relative performance",
            "Sector risk-return",
            "Correlation heatmaps",
            "Rolling volatility",
            "Exact sector table",
        ]
    )

    with tabs[0]:
        show_plot(beta_correlation_figure(stock_index_comparison), "beta_correlation")
        beta_corr = stock_index_comparison[["Beta", "Stock_Index_Correlation"]].corr().iloc[0, 1]
        st.info(
            f"Cross-sectional correlation between beta and stock-index correlation: **{beta_corr:.3f}**. "
            "The positive relationship is mathematically expected because beta also depends on relative stock and market volatility."
        )

    with tabs[1]:
        relative_start, relative_end = safe_date_range(
            "Relative-performance date range",
            analysis_data,
            "relative_range",
            ANALYSIS_START,
            ANALYSIS_END,
        )
        relative_symbols = st.multiselect(
            "Stocks",
            sorted(analysis_data["Symbol"].unique()),
            default=["BAJFINANCE", "EICHERMOT", "HDFCBANK", "TATAMOTORS"],
            key="relative_symbols",
        )
        display_mode = st.radio(
            "View",
            ["Price-return indices", "Relative to NIFTY (100 = matched performance)"],
            horizontal=True,
            key="relative_view",
        )
        if relative_symbols:
            relative = relative_to_nifty(
                analysis_data, relative_symbols, relative_start, relative_end
            )
            if display_mode == "Price-return indices":
                stock_fig = px.line(
                    relative,
                    x="Date",
                    y="Price_Return_Index",
                    color="Symbol",
                    hover_data={"Price_Return_Index": ":.2f"},
                )
                nifty_line = (
                    relative[["Date", "NIFTY_Price_Return_Index"]]
                    .drop_duplicates("Date")
                    .sort_values("Date")
                )
                stock_fig.add_trace(
                    go.Scatter(
                        x=nifty_line["Date"],
                        y=nifty_line["NIFTY_Price_Return_Index"],
                        mode="lines",
                        name="NIFTY 50",
                        line=dict(color="black", width=3, dash="dash"),
                        hovertemplate="%{x|%d %b %Y}<br>NIFTY index: %{y:.2f}<extra></extra>",
                    )
                )
                stock_fig.update_yaxes(title="Price-return index (start = 100)")
                title = "Stock and NIFTY 50 Cumulative Price Performance"
            else:
                stock_fig = px.line(
                    relative,
                    x="Date",
                    y="Relative_to_NIFTY",
                    color="Symbol",
                    hover_data={"Relative_to_NIFTY": ":.2f"},
                )
                stock_fig.add_hline(y=100, line_dash="dash", line_color="black")
                stock_fig.update_yaxes(title="Relative performance index")
                title = "Stock Price Performance Relative to NIFTY 50"
            stock_fig.update_xaxes(title="Date")
            show_plot(finish_figure(stock_fig, title, 650), "relative_performance")
            st.caption("Both stock and benchmark paths use price returns and exclude dividends.")
        else:
            st.info("Select at least one stock.")

    with tabs[2]:
        include_single = st.toggle(
            "Include single-stock sector labels",
            value=False,
            key="include_single_sectors",
        )
        show_plot(
            sector_risk_return_figure(sector_summary, include_single),
            "sector_risk_return",
        )
        eligible = sector_summary.loc[sector_summary["No_of_Companies"] > 1].sort_values(
            "Annualized_Return", ascending=False
        )
        if not eligible.empty:
            st.write(
                "**Top multi-company sample sectors:** "
                + ", ".join(
                    f"{row.Sector} ({row.Annualized_Return:.2%})"
                    for row in eligible.head(3).itertuples()
                )
            )

    with tabs[3]:
        heatmap_start, heatmap_end = safe_date_range(
            "Correlation date range",
            analysis_data,
            "heatmap_range",
            ANALYSIS_START,
            ANALYSIS_END,
        )
        heatmap_type = st.radio(
            "Heatmap",
            ["Stocks", "Sample sectors"],
            horizontal=True,
            key="heatmap_type",
        )
        period = filter_dates(analysis_data, heatmap_start, heatmap_end)

        if heatmap_type == "Stocks":
            available = sorted(period["Symbol"].unique())
            default_heatmap = [
                "AXISBANK",
                "HDFCBANK",
                "ICICIBANK",
                "KOTAKBANK",
                "SBIN",
                "BAJAJ-AUTO",
                "EICHERMOT",
                "MARUTI",
                "TATAMOTORS",
                "BAJFINANCE",
            ]
            heatmap_symbols = st.multiselect(
                "Stocks in heatmap (2–25 recommended)",
                available,
                default=[symbol for symbol in default_heatmap if symbol in available],
                key="heatmap_stocks",
            )
            if len(heatmap_symbols) >= 2:
                matrix = (
                    period.loc[period["Symbol"].isin(heatmap_symbols)]
                    .pivot(index="Date", columns="Symbol", values="Daily_Return")
                    .corr(min_periods=100)
                )
                show_plot(correlation_heatmap(matrix, "Stock Daily-Return Correlation"), "stock_heatmap")
            else:
                st.info("Select at least two stocks.")
        else:
            sector_daily = (
                period.groupby(["Date", "Sector"], as_index=False)["Daily_Return"]
                .mean()
            )
            matrix = sector_daily.pivot(
                index="Date", columns="Sector", values="Daily_Return"
            ).corr(min_periods=100)
            show_plot(
                correlation_heatmap(matrix, "Sample-Sector Daily-Return Correlation"),
                "sector_heatmap",
            )

    with tabs[4]:
        rolling_start, rolling_end = safe_date_range(
            "Rolling-volatility date range",
            analysis_data,
            "rolling_range",
            ANALYSIS_START,
            ANALYSIS_END,
        )
        rolling_symbols = st.multiselect(
            "Stocks",
            sorted(analysis_data["Symbol"].unique()),
            default=["VEDL", "TATAMOTORS", "ZEEL", "HINDALCO", "INDUSINDBK"],
            key="rolling_symbols",
        )
        period = filter_dates(analysis_data, rolling_start, rolling_end)
        fig = go.Figure()
        index_vol = (
            period[["Date", "Index_Rolling_Vol"]]
            .drop_duplicates("Date")
            .sort_values("Date")
        )
        fig.add_trace(
            go.Scatter(
                x=index_vol["Date"],
                y=index_vol["Index_Rolling_Vol"],
                mode="lines",
                name="NIFTY 50",
                line=dict(color="black", width=3),
                hovertemplate="%{x|%d %b %Y}<br>NIFTY vol: %{y:.2%}<extra></extra>",
            )
        )
        for symbol in rolling_symbols:
            stock = period.loc[period["Symbol"] == symbol]
            fig.add_trace(
                go.Scatter(
                    x=stock["Date"],
                    y=stock["Rolling_Vol_21d"],
                    mode="lines",
                    name=symbol,
                    hovertemplate=f"%{{x|%d %b %Y}}<br>{symbol} vol: %{{y:.2%}}<extra></extra>",
                )
            )
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title="21-day rolling daily volatility", tickformat=".1%")
        show_plot(finish_figure(fig, "Rolling Volatility: Stocks versus NIFTY 50", 650), "rolling_volatility")

    with tabs[5]:
        st.dataframe(
            format_table(
                sector_summary.sort_values(
                    ["No_of_Companies", "Annualized_Return"], ascending=[False, False]
                ),
                percent_columns=[
                    "Annualized_Return",
                    "Annualized_Volatility",
                    "Maximum_Drawdown",
                    "Historical_VaR_95_Loss",
                ],
                decimal_columns=["Risk_Adjusted_Return_RF0"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Sector groups are static, equal-weighted sample portfolios. Single-stock labels are not directly ranked with multi-company groups."
        )


# =============================================================================
# DEMONETIZATION PAGE
# =============================================================================

elif selected_page == "Demonetization Event":
    st.title("Demonetization Event Analysis")
    st.markdown(
        '<div class="section-note">Announcement: 8 Nov 2016 after market hours. Trading event day zero: 9 Nov 2016.</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "NIFTY & sectors",
            "Volatility increases",
            "Focus-stock CAR",
            "Candlesticks",
            "Exact event tables",
        ]
    )

    wide_label = "Wide_2016-10-08_to_2017-01-08"

    with tabs[0]:
        show_plot(nifty_event_figure(processed), "nifty_event")

        wide_sectors = demonetization_summary.loc[
            (demonetization_summary["Window"] == wide_label)
            & (demonetization_summary["Analysis_Level"] == "Sector")
        ].sort_values("Event_Cumulative_Return")
        sector_fig = px.bar(
            wide_sectors,
            x="Event_Cumulative_Return",
            y="Entity",
            orientation="h",
            color="Event_Cumulative_Return",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            hover_data={
                "Event_Cumulative_Return": ":.2%",
                "Event_Daily_Volatility": ":.2%",
                "Volatility_Change": ":.2%",
            },
        )
        sector_fig.update_layout(coloraxis_showscale=False)
        sector_fig.update_xaxes(title="Wide-window compounded return", tickformat=".0%")
        sector_fig.update_yaxes(title="")
        show_plot(
            finish_figure(
                sector_fig,
                "Sample-Sector Returns: Wide Demonetization Window",
                620,
                hovermode="closest",
                showlegend=False,
            ),
            "demonetization_sector_returns",
        )

    with tabs[1]:
        wide_stocks = demonetization_summary.loc[
            (demonetization_summary["Window"] == wide_label)
            & (demonetization_summary["Analysis_Level"] == "Stock")
        ].sort_values("Volatility_Change", ascending=False).head(10)
        vol_fig = px.bar(
            wide_stocks.sort_values("Volatility_Change"),
            x="Volatility_Change",
            y="Entity",
            orientation="h",
            color="Volatility_Change",
            color_continuous_scale="Reds",
            hover_data={
                "Event_Daily_Volatility": ":.2%",
                "Baseline_Daily_Volatility": ":.2%",
                "Volatility_Change": ":.2%",
                "Event_Cumulative_Return": ":.2%",
            },
        )
        vol_fig.update_layout(coloraxis_showscale=False)
        vol_fig.update_xaxes(title="Event volatility − baseline volatility", tickformat=".1%")
        vol_fig.update_yaxes(title="")
        show_plot(
            finish_figure(
                vol_fig,
                "Largest Stock Volatility Increases: Wide Window",
                560,
                hovermode="closest",
                showlegend=False,
            ),
            "top_volatility_increases",
        )

    with tabs[2]:
        car_window_display = st.radio(
            "Event window",
            ["[-5,+5]", "[-20,+20]", "Wide"],
            horizontal=True,
            key="car_window",
        )
        selected_window = wide_label if car_window_display == "Wide" else car_window_display
        car_symbols = st.multiselect(
            "Focus stocks",
            FOCUS_STOCKS,
            default=FOCUS_STOCKS,
            key="car_symbols",
        )
        car = abnormal_returns.loc[
            (abnormal_returns["Window"] == selected_window)
            & (abnormal_returns["Analysis_Level"] == "Stock")
            & (abnormal_returns["Entity"].isin(car_symbols))
        ].copy()
        if car.empty:
            st.info("No abnormal-return rows match the selected filters.")
        else:
            fig = px.line(
                car,
                x="Event_Day",
                y="Cumulative_Abnormal_Return",
                color="Entity",
                hover_data={
                    "Date": "|%d %b %Y",
                    "Abnormal_Return": ":.2%",
                    "Cumulative_Abnormal_Return": ":.2%",
                },
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#d62728")
            fig.add_hline(y=0, line_dash="dot", line_color="black")
            fig.update_xaxes(title="Trading day relative to 9 Nov 2016")
            fig.update_yaxes(title="Cumulative abnormal return", tickformat=".0%")
            fig = finish_figure(
                fig,
                f"Focus-Stock CAR: {car_window_display}",
                710,
                hovermode="closest",
            )
            fig.update_layout(
                height=710,
                margin=dict(l=80, r=205, t=100, b=75),
                title=dict(
                    text=f"Focus-Stock CAR: {car_window_display}",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    pad=dict(t=8, b=16),
                ),
                legend=dict(
                    title=dict(text="Focus stock"),
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            fig.update_xaxes(automargin=True)
            fig.update_yaxes(automargin=True)

            show_plot(
                fig,
                f"car_{selected_window}",
            )

    with tabs[3]:
        candle_view = st.radio(
            "Candlestick display",
            ["Selected focus stock", "All 12 separate charts"],
            horizontal=True,
            key="candle_view",
        )
        candle_start, candle_end = safe_date_range(
            "Candlestick date range",
            processed.loc[processed["Date"].between(WIDE_EVENT_START, WIDE_EVENT_END)],
            "candle_range",
            WIDE_EVENT_START,
            WIDE_EVENT_END,
        )

        if candle_view == "Selected focus stock":
            candle_symbol = st.selectbox("Focus stock", FOCUS_STOCKS, key="candle_symbol")
            candle_data = processed.loc[
                (processed["Symbol"] == candle_symbol)
                & processed["Date"].between(candle_start, candle_end)
            ].copy()
            show_plot(candlestick_figure(candle_data, candle_symbol), f"candle_{candle_symbol}")
        else:
            st.caption("Each focus stock is rendered as a separate interactive Plotly candlestick chart.")
            for row_start in range(0, len(FOCUS_STOCKS), 2):
                chart_columns = st.columns(2)
                for offset, symbol in enumerate(FOCUS_STOCKS[row_start : row_start + 2]):
                    with chart_columns[offset]:
                        candle_data = processed.loc[
                            (processed["Symbol"] == symbol)
                            & processed["Date"].between(candle_start, candle_end)
                        ].copy()
                        show_plot(candlestick_figure(candle_data, symbol), f"all_candle_{symbol}")

    with tabs[4]:
        st.markdown("#### NIFTY 50 event statistics")
        nifty_table = demonetization_summary.loc[
            demonetization_summary["Analysis_Level"] == "Index"
        ].copy()
        st.dataframe(
            format_table(
                nifty_table,
                percent_columns=[
                    "Event_Cumulative_Return",
                    "Event_Mean_Daily_Return",
                    "Event_Daily_Volatility",
                    "Baseline_Daily_Volatility",
                    "Volatility_Change",
                ],
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Focus-stock cumulative abnormal returns")
        focus_car = demonetization_summary.loc[
            (demonetization_summary["Analysis_Level"] == "Stock")
            & (demonetization_summary["Entity"].isin(FOCUS_STOCKS)),
            ["Window", "Entity", "Event_Cumulative_Abnormal_Return"],
        ].pivot(index="Entity", columns="Window", values="Event_Cumulative_Abnormal_Return")
        ordered_columns = [column for column in ["[-5,+5]", "[-20,+20]", wide_label] if column in focus_car.columns]
        focus_car = focus_car.reindex(FOCUS_STOCKS)[ordered_columns]
        st.dataframe(
            focus_car.style.format("{:.2%}", na_rep="—"),
            width="stretch",
        )

        st.markdown("#### Wide-window sample-sector statistics")
        wide_sector_table = demonetization_summary.loc[
            (demonetization_summary["Window"] == wide_label)
            & (demonetization_summary["Analysis_Level"] == "Sector")
        ].sort_values("Event_Cumulative_Return", ascending=False)
        st.dataframe(
            format_table(
                wide_sector_table,
                percent_columns=[
                    "Event_Cumulative_Return",
                    "Event_Cumulative_Abnormal_Return",
                    "Event_Daily_Volatility",
                    "Volatility_Change",
                ],
            ),
            width="stretch",
            hide_index=True,
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption(
    "Exploratory dashboard for an academic EDA project. Historical results are descriptive and are not investment advice."
)
