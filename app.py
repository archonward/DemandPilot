from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROCESSED_HISTORY_PATH = Path("data/processed/future_sales_daily.csv")
SAMPLE_HISTORICAL_PATH = Path("data/sample/sample_sales_series.csv")
SAMPLE_FORECAST_PATH = Path("data/processed/sample_forecast.csv")
HISTORICAL_WINDOW_DAYS = 180


st.set_page_config(
    page_title="DemandPilot: Retail Demand Forecasting",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_processed_history(path_str: str) -> pd.DataFrame:
  path = Path(path_str)
  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
  if not path.exists():
    st.error(f"Missing {label} CSV: `{path}`")
    return None

  return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(
      drop=True
  )


def format_date_range(df: pd.DataFrame) -> str:
  return (
      f"{df['date'].min().date().isoformat()} to "
      f"{df['date'].max().date().isoformat()}"
  )


def build_history_chart(historical_df: pd.DataFrame) -> go.Figure:
  history_plot_df = historical_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=history_plot_df["date"],
          y=history_plot_df["value"],
          mode="lines",
          name="Historical demand",
          line={"color": "#1f77b4", "width": 2},
      )
  )
  fig.update_layout(
      title="Historical Demand for Selected Series",
      xaxis_title="Date",
      yaxis_title="Sales quantity",
      hovermode="x unified",
      height=450,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )
  return fig


def build_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
  historical_plot_df = historical_df.tail(HISTORICAL_WINDOW_DAYS).copy()

  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=historical_plot_df["date"],
          y=historical_plot_df["value"],
          mode="lines",
          name="Historical sales",
          line={"color": "#1f77b4", "width": 2},
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["upper_90"],
          mode="lines",
          line={"width": 0},
          hoverinfo="skip",
          showlegend=False,
          name="90% upper",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["lower_90"],
          mode="lines",
          line={"width": 0},
          fill="tonexty",
          fillcolor="rgba(255, 127, 14, 0.15)",
          name="90% prediction interval",
          hovertemplate="90% interval<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["upper_80"],
          mode="lines",
          line={"width": 0},
          hoverinfo="skip",
          showlegend=False,
          name="80% upper",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["lower_80"],
          mode="lines",
          line={"width": 0},
          fill="tonexty",
          fillcolor="rgba(255, 127, 14, 0.30)",
          name="80% prediction interval",
          hovertemplate="80% interval<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=forecast_df["date"],
          y=forecast_df["forecast"],
          mode="lines",
          name="Forecast",
          line={"color": "#ff7f0e", "width": 3},
      )
  )

  fig.update_layout(
      title="Historical Sales and TimesFM Forecast",
      xaxis_title="Date",
      yaxis_title="Sales quantity",
      hovermode="x unified",
      legend_title_text="Series",
      height=520,
      margin={"l": 20, "r": 20, "t": 60, "b": 20},
  )

  return fig


def main() -> None:
  st.title("DemandPilot: Retail Demand Forecasting")
  st.caption(
      "This dashboard uses TimesFM-generated forecasts to help estimate "
      "future product demand."
  )

  with st.sidebar:
    st.header("Instructions")
    st.caption(
      "Current status: MVP dashboard with product-store historical exploration "
      "and precomputed sample TimesFM forecast."
    )
    st.write("Step 1: Prepare dataset")
    st.write("Step 2: Run TimesFM forecast script")
    st.write("Step 3: View forecast in dashboard")
    st.caption(
        "Phase 2 allows historical demand exploration for a selected "
        "product-store series. TimesFM forecasting is still using the "
        "precomputed sample forecast."
    )

  if not PROCESSED_HISTORY_PATH.exists():
    st.error(f"Missing processed historical CSV: `{PROCESSED_HISTORY_PATH}`")
    return

  processed_history_df = load_processed_history(str(PROCESSED_HISTORY_PATH))

  shop_options = sorted(processed_history_df["shop_id"].dropna().unique().tolist())
  selected_shop_id = st.sidebar.selectbox("shop_id", shop_options)

  item_options = sorted(
      processed_history_df.loc[
          processed_history_df["shop_id"] == selected_shop_id, "item_id"
      ]
      .dropna()
      .unique()
      .tolist()
  )
  selected_item_id = st.sidebar.selectbox("item_id", item_options)

  historical_df = processed_history_df.loc[
      (processed_history_df["shop_id"] == selected_shop_id)
      & (processed_history_df["item_id"] == selected_item_id)
  ].copy()

  if historical_df.empty:
    st.error(
        "No historical rows found for the selected shop_id and item_id "
        "combination."
    )
    return

  historical_df = historical_df.sort_values("date").reset_index(drop=True)
  series_id = str(historical_df["series_id"].iloc[0])

  st.subheader("Selected product-store series")
  st.write(series_id)
  st.write(f"Historical data: {format_date_range(historical_df)}")

  if len(historical_df) < HISTORICAL_WINDOW_DAYS:
    st.warning("This series has limited history, so forecast quality may be weaker.")

  top_metrics = st.columns(3)
  top_metrics[0].metric("Historical rows", f"{len(historical_df):,}")
  top_metrics[1].metric(
      "Average historical daily sales",
      f"{historical_df['value'].mean():.2f}",
  )
  top_metrics[2].metric("Max daily sales", f"{historical_df['value'].max():.0f}")

  lower_metrics = st.columns(2)
  lower_metrics[0].metric(
      "Total historical sales",
      f"{historical_df['value'].sum():.0f}",
  )
  lower_metrics[1].metric(
      "Average item price",
      f"{historical_df['avg_item_price'].mean():.2f}",
  )

  st.subheader("Historical Demand")
  st.caption("The chart shows the most recent 180 rows of historical demand.")
  history_chart = build_history_chart(historical_df)
  st.plotly_chart(history_chart, use_container_width=True)

  sample_historical_df = load_csv(SAMPLE_HISTORICAL_PATH, "sample historical data")
  forecast_df = load_csv(SAMPLE_FORECAST_PATH, "forecast data")

  st.subheader("Sample TimesFM Forecast")
  st.caption(
      "This forecast is currently precomputed for the sample series only. "
      "It does not change when you select a different shop_id or item_id."
  )

  if sample_historical_df is None or forecast_df is None:
    return

  forecast_horizon = len(forecast_df)
  forecast_csv = SAMPLE_FORECAST_PATH.read_bytes()
  expected_demand = forecast_df["forecast"].sum()
  conservative_demand = forecast_df["lower_80"].sum()
  high_demand = forecast_df["upper_80"].sum()
  recommended_stock = math.ceil(high_demand)
  safety_buffer = math.ceil(high_demand - expected_demand)

  st.write(f"Forecast period: {format_date_range(forecast_df)}")

  sample_top_metrics = st.columns(3)
  sample_top_metrics[0].metric("Forecast horizon", f"{forecast_horizon} days")
  sample_top_metrics[1].metric(
      "Expected forecast demand",
      f"{round(expected_demand):.0f} units",
  )
  sample_top_metrics[2].metric(
      "Conservative demand estimate",
      f"{round(conservative_demand):.0f} units",
  )

  sample_lower_metrics = st.columns(2)
  sample_lower_metrics[0].metric(
      "High-demand estimate",
      f"{round(high_demand):.0f} units",
  )
  sample_lower_metrics[1].metric(
      "Sample historical rows",
      f"{len(sample_historical_df):,}",
  )

  st.subheader("Inventory Planning Recommendation")
  recommendation_columns = st.columns(2)
  recommendation_columns[0].metric(
      "Recommended stock for next 30 days",
      f"{recommended_stock} units",
  )
  recommendation_columns[1].metric(
      "Safety buffer above expected demand",
      f"{safety_buffer} units",
  )
  st.caption(
      "The recommendation uses the upper 80% forecast range to reduce stockout "
      "risk while avoiding the more aggressive 90% worst-case estimate."
  )
  st.caption(
      "Inventory recommendation is currently based on the precomputed sample "
      "TimesFM forecast. In the next phase, this will be generated for the "
      "selected series."
  )

  st.subheader("Forecast Chart")
  st.caption(
      "The chart compares recent historical sales with the next 30 days of "
      "TimesFM forecasted demand."
  )
  chart = build_forecast_chart(sample_historical_df, forecast_df)
  st.plotly_chart(chart, use_container_width=True)

  st.subheader("Forecast Details")
  st.download_button(
      label="Download forecast CSV",
      data=forecast_csv,
      file_name=SAMPLE_FORECAST_PATH.name,
      mime="text/csv",
  )

  display_forecast_df = forecast_df.copy()
  numeric_columns = display_forecast_df.select_dtypes(include="number").columns
  display_forecast_df[numeric_columns] = display_forecast_df[numeric_columns].round(2)
  st.dataframe(display_forecast_df, use_container_width=True)


if __name__ == "__main__":
  main()
